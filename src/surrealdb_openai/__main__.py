import os
import logging
import contextlib
import datetime
import string
from typing import AsyncGenerator
import zipfile
import ast

from dotenv import load_dotenv
import fastapi
from fastapi import Form
from fastapi.responses import JSONResponse
from fastapi import templating, responses, staticfiles
import surrealdb
import pandas as pd
import tqdm
import wget

# Load environment variables
load_dotenv()

# Configuration
SURREALDB_URL = os.getenv('SURREALDB_URL', 'ws://localhost:8080')
SURREALDB_USERNAME = os.getenv('SURREALDB_USERNAME', 'root')
SURREALDB_PASSWORD = os.getenv('SURREALDB_PASSWORD', 'root')
SURREALDB_NAMESPACE = os.getenv('SURREALDB_NAMESPACE', 'test')
SURREALDB_DATABASE = os.getenv('SURREALDB_DATABASE', 'test')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

FORMATTED_RECORD_FOR_INSERT_WIKI_EMBEDDING = string.Template(
    """{url: "$url", title: s"$title", text: s"$text", title_vector: $title_vector, content_vector: $content_vector}"""
)

INSERT_WIKI_EMBEDDING_QUERY = string.Template(
    """
    INSERT INTO wiki_embedding [\n $records\n];
    """
)

TOTAL_ROWS = 25000
CHUNK_SIZE = 100

# Set up FastAPI app
app = fastapi.FastAPI()
templates = templating.Jinja2Templates(directory="templates")
app.mount("/static", staticfiles.StaticFiles(directory="static"), name="static")

# Global variables
life_span = {}

def extract_id(surrealdb_id: str) -> str:
    return surrealdb_id.split(":")[1]

def convert_timestamp_to_date(timestamp: str) -> str:
    parsed_timestamp = datetime.datetime.fromisoformat(timestamp.rstrip("Z"))
    return parsed_timestamp.strftime("%B %d %Y, %H:%M")

templates.env.filters["extract_id"] = extract_id
templates.env.filters["convert_timestamp_to_date"] = convert_timestamp_to_date

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = setup_logger(__name__)

@contextlib.asynccontextmanager
async def lifespan(_: fastapi.FastAPI) -> AsyncGenerator:
    """FastAPI lifespan to create and destroy objects."""
    connection = surrealdb.AsyncSurrealDB(url="ws://localhost:8080/rpc")
    await connection.connect()
    await connection.signin(data={"username": "root", "password": "root"})
    await connection.use_namespace("test")
    await connection.use_database("test")
    life_span["surrealdb"] = connection
    yield
    life_span.clear()

@app.get("/", response_class=responses.HTMLResponse)
def index(request: fastapi.Request) -> responses.HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/create-chat")
async def create_chat(request: fastapi.Request) -> JSONResponse:
        chat_record = await life_span["surrealdb"].query("RETURN fn::create_chat();")
        return JSONResponse(content={
            "chat_id": chat_record[0].get("id"),
            "chat_title": chat_record[0].get("title"),
        })

@app.get("/load-chat/{chat_id}")
async def load_chat(request: fastapi.Request, chat_id: str) -> JSONResponse:
        message_records = await life_span["surrealdb"].query("RETURN fn::load_chat($chat_id)", {"chat_id": chat_id})
        return JSONResponse(content={
            "messages": message_records[0],
            "chat_id": chat_id,
        })

@app.get("/chats")
async def chats(request: fastapi.Request) -> JSONResponse:
        chat_records = await life_span["surrealdb"].query("RETURN fn::load_all_chats();")
        return JSONResponse(content={"chats": chat_records[0]})

@app.post("/send-user-message")
async def send_user_message(
    request: fastapi.Request,
    chat_id: str = Form(...),
    content: str = Form(...),
) -> JSONResponse:
        user_message = await life_span["surrealdb"].query(
                "RETURN fn::create_user_message($chat_id, $content)",
                {
                    "chat_id": chat_id,
                    "content": content,
                }
            )
        system_response = await life_span["surrealdb"].query(
                "RETURN fn::create_system_message($chat_id, $token)",
                {
                    "chat_id": chat_id,
                    "token": OPENAI_API_KEY,
                }
            )
        return JSONResponse(content={
            "chat_id": chat_id,
            "user_message": user_message[0],
            "system_message": system_response[0],
        })


@app.get("/send-system-message/{chat_id}")
async def send_system_message(request: fastapi.Request, chat_id: str) -> JSONResponse:
        message = await life_span["surrealdb"].query(
                "RETURN fn::create_system_message($chat_id, $token)",
                {
                    "chat_id": chat_id,
                    "token": OPENAI_API_KEY,
                }
            )
        title = await life_span["surrealdb"].query(
                "RETURN fn::get_chat_title($chat_id)",
                {"chat_id": chat_id}
            )
        return JSONResponse(content={
            "content": message[0].get("content"),
            "timestamp": message[0].get("timestamp"),
            "create_title": title[0] == "Untitled chat",
            "chat_id": chat_id,
        })


@app.get("/create-title/{chat_id}")
async def create_title(chat_id: str) -> JSONResponse:
        title = await life_span["surrealdb"].query(
                "RETURN fn::generate_chat_title($chat_id, $token)",
                {
                    "chat_id": chat_id,
                    "token": OPENAI_API_KEY,
                }
            )
        return JSONResponse(content={"title": title[0].strip('"')})

def get_data() -> None:
    logger = setup_logger("get-data")
    logger.info("Downloading Wikipedia")
    wget.download(
        url="https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip",
        out="data/vector_database_wikipedia_articles_embedded.zip",
    )
    logger.info("Extracting")
    with zipfile.ZipFile("data/vector_database_wikipedia_articles_embedded.zip", "r") as zip_ref:
        zip_ref.extractall("data")
    logger.info("Extracted file successfully. Please check the data folder")

def surreal_insert() -> None:
    """Main entrypoint to insert Wikipedia embeddings into SurrealDB."""
    logger = setup_logger("surreal_insert")

    total_chunks = TOTAL_ROWS // CHUNK_SIZE + (
        1 if TOTAL_ROWS % CHUNK_SIZE else 0
    )

    logger.info("Connecting to SurrealDB")
    connection = surrealdb.SurrealDB("ws://localhost:8080/rpc")
    connection.signin(data={"username": "root", "password": "root"})
    connection.use_namespace("test")
    connection.use_database("test")

    logger.info("Inserting rows into SurrealDB")
    with tqdm.tqdm(total=total_chunks, desc="Inserting") as pbar:
        for chunk in tqdm.tqdm(
            pd.read_csv(
                "data/vector_database_wikipedia_articles_embedded.csv",
                usecols=[
                    "url",
                    "title",
                    "text",
                    "title_vector",
                    "content_vector",
                ],
                chunksize=CHUNK_SIZE,
            ),
        ):
            formatted_rows = [
                FORMATTED_RECORD_FOR_INSERT_WIKI_EMBEDDING.substitute(
                    url=row["url"],
                    title=row["title"]
                    .replace("\\", "\\\\")
                    .replace('"', '\\"'),
                    text=row["text"].replace("\\", "\\\\").replace('"', '\\"'),
                    title_vector=ast.literal_eval(row["title_vector"]),
                    content_vector=ast.literal_eval(row["content_vector"]),
                )
                for _, row in chunk.iterrows()  # type: ignore
            ]
            connection.query(
                query=INSERT_WIKI_EMBEDDING_QUERY.substitute(
                    records=",\n ".join(formatted_rows)
                )
            )
            pbar.update(1)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)