import io
import pandas as pd
import streamlit as st

st.title("📂 Upload Dataset")
st.caption("Format yang didukung: CSV / Excel. Kolom minimal: **isi** (deskripsi), **id_bugtrack** (bisa diekstrak tanggal).")

# ---------- Helpers ----------
@st.cache_data(show_spinner=False)
def read_csv_bytes(b: bytes, sep: str, encoding: str) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(b), sep=sep, encoding=encoding)

@st.cache_data(show_spinner=False)
def read_excel_bytes(b: bytes, sheet_name: str | int | None) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(b), sheet_name=sheet_name)

def extract_timestamp_from_id_bugtrack(s: pd.Series) -> pd.Series:
    """
    id_bugtrack contoh: 20230102074234Mon -> 2023-01-02 07:42:34
    Ambil 14 digit pertama sebagai datetime; abaikan suffix huruf.
    """
    out = pd.to_datetime(s.astype(str).str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    return out

# ---------- UI upload ----------
uploaded_file = st.file_uploader("Pilih file dataset", type=["csv", "xlsx"])

if not uploaded_file:
    st.info("Belum ada file diupload.")
    st.stop()

file_bytes = uploaded_file.read()
file_size_mb = len(file_bytes) / (1024 * 1024)

# (opsional) batasi ukuran file 25 MB
MAX_MB = 25
if file_size_mb > MAX_MB:
    st.error(f"Ukuran file {file_size_mb:.1f} MB > {MAX_MB} MB. Mohon unggah file yang lebih kecil.")
    st.stop()

# ---------- Parameter parsing ----------
ext = uploaded_file.name.lower().split(".")[-1]
df = None

if ext == "csv":
    with st.expander("Pengaturan CSV (opsional)", expanded=False):
        sep = st.selectbox("Delimiter", [",", ";", "\t", "|"], index=0)
        encoding = st.selectbox("Encoding", ["utf-8", "latin-1", "cp1252"], index=0)
    with st.spinner("Membaca CSV..."):
        df = read_csv_bytes(file_bytes, sep=sep, encoding=encoding)
else:
    # Excel
    with pd.ExcelFile(io.BytesIO(file_bytes)) as xls:
        sheets = xls.sheet_names
    with st.expander("Pengaturan Excel", expanded=False):
        sheet = st.selectbox("Pilih sheet", sheets, index=0)
    with st.spinner("Membaca Excel..."):
        df = read_excel_bytes(file_bytes, sheet_name=sheet)

if df is None or df.empty:
    st.error("Gagal membaca file atau file kosong.")
    st.stop()

# ---------- Ringkasan ----------
st.subheader("Ringkasan Data")
col1, col2, col3 = st.columns(3)
col1.metric("Baris", f"{df.shape[0]:,}")
col2.metric("Kolom", f"{df.shape[1]:,}")
col3.write("**Kolom:**")
col3.write(list(df.columns))

with st.expander("Preview (maks 200 baris)"):
    st.dataframe(df.head(200), use_container_width=True)

# ---------- Validasi kolom minimal ----------
required_cols = {"isi", "id_bugtrack"}
has_required = required_cols.issubset(df.columns)

if has_required:
    st.success("✅ Dataset memiliki kolom wajib: `isi` dan `id_bugtrack`.")
    # Tawarkan ekstraksi timestamp dari id_bugtrack
    with st.expander("Ekstraksi Tanggal dari `id_bugtrack`", expanded=False):
        want_extract = st.checkbox("Buat kolom `timestamp` dari `id_bugtrack`", value=True)
        if want_extract:
            tmp = df.copy()
            tmp["timestamp"] = extract_timestamp_from_id_bugtrack(tmp["id_bugtrack"])
            n_na = tmp["timestamp"].isna().sum()
            if n_na > 0:
                st.warning(f"Sebanyak {n_na} baris gagal diparse datetime dari `id_bugtrack`.")
            st.dataframe(tmp[["id_bugtrack", "timestamp"]].head(20), use_container_width=True)
            df = tmp
else:
    st.warning(
        "⚠️ Kolom wajib belum lengkap. "
        "Harus ada **`isi`** (deskripsi teks) dan **`id_bugtrack`** (ID berisi timestamp). "
        "Silakan sesuaikan dataset Anda."
    )

# ---------- Simpan ke session ----------
apply_col1, apply_col2 = st.columns([1, 3])
if apply_col1.button("Gunakan dataset ini", type="primary"):
    st.session_state["df_raw"] = df  # simpan mentah/hasil minimal parsing
    st.toast("Dataset disimpan ke sesi ✅")
    st.success("Dataset siap dipakai di halaman Preprocessing / Analisis.")

# (opsional) Download versi bersih sebagai bukti
if "df_raw" in st.session_state:
    buf = io.BytesIO()
    st.session_state["df_raw"].to_csv(buf, index=False)
    st.download_button(
        "Unduh salinan CSV (session)",
        data=buf.getvalue(),
        file_name="dataset_session.csv",
        mime="text/csv",
    )
