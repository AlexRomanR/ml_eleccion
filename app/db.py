# db.py
import os
from sqlalchemy import create_engine

def get_engine():
    # Usa variable de entorno DB_URL si existe; si no, usa tu BD local
    url = os.getenv(
        "DB_URL",
        "mysql+pymysql://<USER>:<PASS>@<PUBLIC_HOST>:<PORT>/<DB>?charset=utf8mb4"
    )
    return create_engine(url, pool_pre_ping=True)
