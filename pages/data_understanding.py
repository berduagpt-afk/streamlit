# pages/upload_dataset.py — versi aman untuk file besar & CSV tidak konsisten
import io
import re
import pandas as pd
import streamlit as st
from datetime import datetime
from sqlalchemy import create_engine

# ==============================================================
# 1️⃣ AUTENTIKASI & KONFIGURASI
# ==============================================================

if not st.session_state.get("logged_in", False):
    st.error("Silakan login dulu untuk mengakses halaman ini.")
    st.stop()

# Tema halaman
st.markdown(
    """
    <style>
    .header {color:#0B3A82;font-weight:800;font-size:1.75rem;margin-bottom:-0.25rem;}
    .subtext {color:#64748B;font-size:0.95rem;}
    </style>
    <div class="header">📂 Upload Dataset Tiket Insiden</div>
    <div class="subtext">
        Format didukung: <b>CSV</b> / <b>Excel</b><br>
        Kolom wajib: <code>Detailed_Decription</code>, <code>Incident_Number</code>, dan <code>tgl_submit</code>.
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ==============================================================
# 2️⃣ HELPER FUNCTIONS
# ==============================================================

def detect_delimiter(sample_bytes: bytes) -> str:
    """Deteksi delimiter umum dari beberapa baris pertama file."""
    text = sample_bytes.decode(errors="ignore")
    sample = text[:5000]  # ambil sebagian awal
    candidates = [",", ";", "|", "\t"]
    counts = {c: sample.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best

@st.cache_data(show_spinner=False)
def read_csv_safe(b: bytes, sep: str | None = None, encoding: str = "utf-8") -> tuple[pd.DataFrame, int]:
    """
    Membaca CSV besar dengan toleransi tinggi terhadap error baris.
    Mengembalikan (DataFrame, skipped_lines)
    """
    # deteksi delimiter otomatis jika belum diberikan
    if sep is None:
        sep = detect_delimiter(b)

    skipped = 0
    bad_rows = []

    def on_bad_lines_callback(line):
        nonlocal skipped
        skipped += 1
        bad_rows.append(line)
        return None  # lewati baris

    try:
        df = pd.read_csv(
            io.BytesIO(b),
            sep=sep,
            encoding=encoding,
            engine="python",
            on_bad_lines=on_bad_lines_callback,
        )
    except Exception as e:
        raise RuntimeError(f"Gagal membaca CSV: {e}")

    return df, skipped

@st.cache_data(show_spinner=False)
def read_excel_bytes(b: bytes, sheet_name: str | int | None) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(b), sheet_name=sheet_name)

def split_detailed_description(text: str) -> dict:
    parts = {
        "modul": None,
        "sub_modul": None,
        "jenis_masalah": None,
        "judul_masalah": None,
        "data": None,
        "jenis_data": None,
        "isi_permasalahan": None,
    }
    pattern = (
        r"MODUL:\s*(.*?)\s*SUB MODUL:\s*(.*?)\s*"
        r"JENIS MASALAH:\s*(.*?)\s*JUDUL MASALAH:\s*(.*?)\s*"
        r"DATA:\s*(.*?)\s*JENIS DATA:\s*(.*?)\s*"
        r"ISI PERMASALAHAN:\s*(.*)"
    )
    match = re.search(pattern, str(text), flags=re.DOTALL | re.IGNORECASE)
    if match:
        for i, key in enumerate(parts.keys()):
            parts[key] = match.groups()[i].strip()
    return parts

# ---------- PostgreSQL Utilities ----------
def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url)

def save_dataframe(df: pd.DataFrame, table_name: str, schema: str = "lasis_djp", if_exists: str = "replace"):
    engine = get_connection()
    df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False)
    engine.dispose()

# ==============================================================
# 3️⃣ UPLOAD FILE
# ==============================================================

uploaded_file = st.file_uploader("Pilih file dataset insiden", type=["csv", "xlsx"])
if not uploaded_file:
    st.info("Belum ada file yang diunggah.")
    st.stop()

file_bytes = uploaded_file.read()
file_size_mb = len(file_bytes) / (1024 * 1024)
MAX_MB = 400

if file_size_mb > MAX_MB:
    st.error(f"Ukuran file {file_size_mb:.1f} MB > {MAX_MB} MB. Mohon unggah file lebih kecil.")
    st.stop()

st.success(f"File diterima: {uploaded_file.name} ({file_size_mb:.1f} MB)")

# ==============================================================
# 4️⃣ BACA FILE
# ==============================================================

ext = uploaded_file.name.lower().split(".")[-1]
df = None
skipped_lines = 0

if ext == "csv":
    with st.expander("⚙️ Pengaturan CSV (opsional)", expanded=False):
        auto_detect = st.checkbox("Deteksi delimiter otomatis", value=True)
        sep = None if auto_detect else st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
        encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)
    with st.spinner("📖 Membaca file CSV..."):
        df, skipped_lines = read_csv_safe(file_bytes, sep=sep, encoding=encoding)
else:
    with pd.ExcelFile(io.BytesIO(file_bytes)) as xls:
        sheets = xls.sheet_names
    with st.expander("⚙️ Pengaturan Excel", expanded=False):
        sheet = st.selectbox("Pilih sheet", sheets, index=0)
    with st.spinner("📖 Membaca file Excel..."):
        df = read_excel_bytes(file_bytes, sheet_name=sheet)

if df is None or df.empty:
    st.error("Gagal membaca file atau file kosong.")
    st.stop()

if skipped_lines > 0:
    st.warning(f"{skipped_lines:,} baris dilewati karena rusak atau jumlah kolom tidak sesuai.")

# ==============================================================
# 5️⃣ RINGKASAN DATA
# ==============================================================

st.subheader("Ringkasan Data")
col1, col2, col3 = st.columns(3)
col1.metric("Jumlah Baris", f"{df.shape[0]:,}")
col2.metric("Jumlah Kolom", f"{df.shape[1]:,}")
col3.markdown("**Kolom Awal:**")
col3.code(", ".join(df.columns), language="text")

with st.expander("👀 Preview Dataset (maks 200 baris)"):
    st.dataframe(df.head(200), use_container_width=True)

# ==============================================================
# 6️⃣ VALIDASI KOLOM WAJIB & PEMECAHAN
# ==============================================================

required_cols = {"Detailed_Decription", "Incident_Number", "tgl_submit"}
if required_cols.issubset(df.columns):
    st.success("✅ Dataset memiliki kolom wajib.")
    st.info("🔍 Memecah kolom `Detailed_Decription` menjadi 7 bagian…")
    parsed = df["Detailed_Decription"].apply(split_detailed_description).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)
    st.dataframe(df[["Incident_Number", "modul", "sub_modul", "jenis_masalah", "judul_masalah", "jenis_data"]].head(10))
else:
    st.warning("⚠️ Kolom wajib belum lengkap.")
    st.stop()

# ==============================================================
# 7️⃣ SIMPAN KE DATABASE
# ==============================================================

col_save, _ = st.columns([1, 3])
if col_save.button("💾 Simpan Dataset ke Database", type="primary"):
    st.session_state["df_raw"] = df
    st.session_state["last_update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        with st.spinner("Menyimpan ke PostgreSQL..."):
            save_dataframe(df, table_name="incident_raw", schema="lasis_djp", if_exists="replace")
        st.toast("Dataset berhasil disimpan ✅")
        st.success("Dataset tersimpan ke PostgreSQL & session state.")
    except Exception as e:
        st.error(f"Gagal menyimpan ke database: {e}")

# ==============================================================
# 8️⃣ OPSIONAL: DOWNLOAD SALINAN
# ==============================================================

if "df_raw" in st.session_state:
    buf = io.BytesIO()
    st.session_state["df_raw"].to_csv(buf, index=False)
    st.download_button(
        "⬇️ Unduh salinan CSV (hasil session)",
        data=buf.getvalue(),
        file_name="dataset_session.csv",
        mime="text/csv",
    )
