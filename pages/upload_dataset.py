# pages/upload_dataset.py — SIMPLIFIED (Tanpa Bangun 7 Kolom Final)

import io
import csv
import pandas as pd
import streamlit as st
from datetime import datetime
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ==============================================================
# 1) AUTH
# ==============================================================

if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

st.markdown("""
<style>
.header {color:#0B3A82;font-weight:800;font-size:1.75rem;margin-bottom:-0.25rem;}
.subtext {color:#64748B;font-size:0.95rem;}
</style>
<div class="header">📂 Upload Dataset Tiket Insiden</div>
<div class="subtext">
Format: <b>CSV</b> / <b>Excel</b><br>
Kolom minimal: <code>Detailed_Decription</code>, <code>Incident_Number</code>, <code>tgl_submit</code><br>
<b>Disimpan (7 kolom):</b> <code>tgl_submit, incident_number, site, assignee, modul, sub_modul, detailed_decription</code>
</div>
""", unsafe_allow_html=True)
st.divider()

# ==============================================================
# 2) UTIL DB
# ==============================================================

def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    driver = cfg.get("dialect", "postgresql")
    if "+" not in driver:
        driver = f"{driver}+psycopg2"
    url = URL.create(
        drivername=driver,
        username=cfg["username"],
        password=cfg["password"],
        host=cfg["host"],
        port=int(cfg["port"]),
        database=cfg["database"],
    )
    return create_engine(url, pool_pre_ping=True)

def ensure_schema(engine, schema):
    if not schema or schema.lower()=="public":
        return
    with engine.begin() as con:
        con.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}";'))

def save_dataframe(df, table_name, schema="lasis_djp", if_exists="replace"):
    engine = get_connection()
    try:
        ensure_schema(engine, schema)
        df.to_sql(
            name=table_name,
            con=engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            chunksize=50_000,
            method="multi",
        )
    finally:
        engine.dispose()

# ==============================================================
# 3) IO Helpers
# ==============================================================

def detect_delimiter(b):
    sample = b[:10000].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","|","\t"])
        return dialect.delimiter
    except:
        return max([",",";","|","\t"], key=sample.count)

def try_encodings(b, encs=("utf-8","cp1252","latin-1")):
    for e in encs:
        try:
            b.decode(e)
            return e
        except: pass
    return "latin-1"

@st.cache_data(show_spinner=False)
def read_csv_safe(b, sep=None, encoding="auto"):
    if encoding=="auto":
        encoding = try_encodings(b)
    if sep is None:
        sep = detect_delimiter(b)
    df = pd.read_csv(io.BytesIO(b), sep=sep, encoding=encoding,
                     engine="python", dtype=str, keep_default_na=False)
    return df

@st.cache_data(show_spinner=False)
def read_excel_bytes(b, sheet_name):
    return pd.read_excel(io.BytesIO(b), sheet_name=sheet_name, dtype=str)

# ==============================================================
# 4) Alias Kolom
# ==============================================================

def remap_aliases(df):
    alias = {
        "Detailed_Decription": ["detailed_decription","detailed_description","detailed desc","uraian","detail"],
        "Incident_Number": ["incident_number","incident","ticket_id","no_tiket"],
        "tgl_submit": ["tgl_submit","tanggal","created_date","submit_date"],
        "site": ["site","lokasi","area"],
        "assignee": ["assignee","petugas","assigned_to"],
        "modul": ["modul","module"],
        "sub_modul": ["sub_modul","submodule"],
    }
    lower_map = {c.lower(): c for c in df.columns}
    for target, alist in alias.items():
        if target in df.columns:
            continue
        for a in alist:
            if a in df.columns:
                df.rename(columns={a: target}, inplace=True); break
            if a.lower() in lower_map:
                df.rename(columns={lower_map[a.lower()]: target}, inplace=True); break
    return df

# ==============================================================
# 5) UPLOAD FILE
# ==============================================================

uploaded = st.file_uploader("Pilih file dataset", type=["csv","xlsx"])
if not uploaded:
    st.stop()

b = uploaded.read()
ext = uploaded.name.lower().split(".")[-1]

if ext=="csv":
    df = read_csv_safe(b)
else:
    sheets = pd.ExcelFile(io.BytesIO(b)).sheet_names
    sheet = st.selectbox("Pilih sheet", sheets, index=0)
    df = read_excel_bytes(b, sheet)

df = remap_aliases(df)

# ==============================================================
# 6) Validasi Minimal
# ==============================================================

required = {"Detailed_Decription","Incident_Number","tgl_submit"}
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Kolom wajib hilang: {', '.join(missing)}")
    st.stop()

st.success("File berhasil dibaca & kolom minimal tersedia.")

# ==============================================================
# 7) Bentuk 7 Kolom Final (langsung dipakai)
# ==============================================================

# Pastikan kolom ada, jika tidak → isi None
for c in ["site","assignee","modul","sub_modul"]:
    if c not in df.columns:
        df[c] = None

df7 = df[[
    "tgl_submit",
    "Incident_Number",
    "site",
    "assignee",
    "modul",
    "sub_modul",
    "Detailed_Decription"
]].rename(columns={
    "Incident_Number": "incident_number",
    "Detailed_Decription": "detailed_decription"
})

# Trim spasi
for c in df7.columns:
    df7[c] = df7[c].astype(str).str.strip()

# ==============================================================
# 8) Preview
# ==============================================================

with st.expander("Preview (maks 200 baris)", expanded=True):
    st.dataframe(df7.head(200), use_container_width=True)

# ==============================================================
# 9) Simpan ke Database
# ==============================================================

if st.button("💾 Simpan ke Database", type="primary"):
    try:
        with st.spinner("Menyimpan ke PostgreSQL..."):
            save_dataframe(
                df7,
                table_name="incident_raw",
                schema="lasis_djp",
                if_exists="replace"
            )
        st.success("Data berhasil disimpan (replace).")
    except Exception as e:
        st.error(f"Error menyimpan: {e}")

# ==============================================================
# 10) Download 7 Kolom
# ==============================================================

buf = io.BytesIO()
df7.to_csv(buf, index=False)
st.download_button(
    "⬇️ Unduh CSV (7 kolom)",
    data=buf.getvalue(),
    file_name="incident_7kolom.csv",
    mime="text/csv"
)
