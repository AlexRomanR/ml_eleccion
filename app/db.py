# app/db.py
from urllib.parse import quote_plus
from sqlalchemy import create_engine

# ✅ Datos PÚBLICOS (Railway proxy)
RAILWAY_HOST = "centerbeam.proxy.rlwy.net"
RAILWAY_PORT = 14209
RAILWAY_DB   = "railway"
RAILWAY_USER = "root"
RAILWAY_PASS = "EDXzpjNKykmenwtSNlnlXnpGmckkIgRx"

def get_engine():
    pwd = quote_plus(RAILWAY_PASS)

    url = (
        f"mysql+pymysql://{RAILWAY_USER}:{pwd}"
        f"@{RAILWAY_HOST}:{RAILWAY_PORT}/{RAILWAY_DB}"
        f"?charset=utf8mb4"
    )

    # Si tu MySQL de Railway exige TLS, esto ayuda a que PyMySQL conecte por SSL.
    return create_engine(
        url,
        pool_pre_ping=True,
        pool_recycle=280,
        connect_args={"ssl": {}},
    )
