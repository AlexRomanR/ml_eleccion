# app/db.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Variable de entorno DB_URL (recomendado para Cloud Run)
# Formato PostgreSQL: postgresql+psycopg2://<USER>:<PASS>@<HOST>:<PORT>/<DB>
DB_URL = os.environ.get(
    "DB_URL",
    "postgresql+psycopg2://postgres:password@localhost:5432/railway"
)

def get_engine():
    return create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_recycle=280
    )
