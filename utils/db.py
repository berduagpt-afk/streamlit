# utils/db.py
import streamlit as st
from sqlalchemy import create_engine, text
import pandas as pd

def get_connection():
    """Membuat koneksi SQLAlchemy ke database Postgres."""
    cfg = st.secrets["connections"]["postgres"]
    url = f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    engine = create_engine(url)
    return engine

def save_dataframe(df: pd.DataFrame, table_name: str, schema: str = "public", if_exists: str = "replace"):
    """Simpan DataFrame ke tabel PostgreSQL."""
    engine = get_connection()
    df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False)
    engine.dispose()

def load_dataframe(table_name: str, schema: str = "public") -> pd.DataFrame:
    """Ambil seluruh data dari tabel."""
    engine = get_connection()
    query = text(f"SELECT * FROM {schema}.{table_name}")
    df = pd.read_sql(query, engine)
    engine.dispose()
    return df
