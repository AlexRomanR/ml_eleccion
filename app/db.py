# app/db.py
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

# Para Cloud SQL en Cloud Run, el formato ideal usando Sockets Unix es:
# postgresql+psycopg2://<USER>:<PASS>@/<DB>?host=/cloudsql/<PROJECT_ID>:<REGION>:<INSTANCE_NAME>
DB_URL = os.environ.get(
    "DB_URL",
    "postgresql+psycopg2://postgres:password@localhost:5432/railway"
)

def get_engine():
    # Eliminamos el argumento 'ssl' vacío ya que Cloud Run a Cloud SQL en la misma red
    # usando sockets Unix o VPC Serverless maneja la seguridad/encriptación nativamente.
    return create_engine(
        DB_URL,
        pool_pre_ping=True,
        pool_recycle=280
    )
