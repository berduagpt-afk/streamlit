# pages/data_preparation/text_processing/data_normalization.py
# Text Normalization — Data Insiden DJP — Data Preparation
# Source : lasis_djp.incident_kelayakan
# Output : lasis_djp.incident_normalized (kolom tetap: text_norm, tgl_normalized)
# Catatan: setiap run akan REPLACE isi text_norm (bukan versi kolom)
#
# Versi FINAL:
# - Placeholder replacement bertipe (mis: __PH_NPWP__, __PH_EMAIL__, __PH_URL__, __PH_IP__, __PH_LONGNUM__)
# - Urutan pemrosesan paling stabil via stable_preprocess_order() dari pages/ref_normalization.py
# - Progress bar tetap ada
# - Simpan REPLACE table incident_normalized

import re
import unicodedata
from datetime import datetime, timezone, timedelta

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# ======================================================
# 🧠 OPSIONAL: ftfy untuk mojibake
# ======================================================
try:
    from ftfy import fix_text as _ftfy_fix  # type: ignore
    _FTFY_AVAILABLE = True
except Exception:
    _FTFY_AVAILABLE = False

    def _ftfy_fix(x):
        return x

# ======================================================
# 📚 Kamus & Redaction (pages/ref_normalization.py)
# ======================================================
# Import utama: gunakan stable_preprocess_order + placeholder type-aware stats
try:
    from pages.ref_normalization import (
        # placeholder
        replace_non_informative,
        count_placeholders,
        remove_single_char_tokens,
        replace_and_count_placeholders,   # ✅ untuk stats lengkap
        # stable order
        stable_preprocess_order,          # ✅ urutan pemrosesan stabil
        # lexicon
        WORD_NORMALIZATION_MAP,
        LEXICAL_NORMALIZATION_MAP,
    )
    _REDACT_OK = True
    _LEXICON_OK = True
    _STABLE_OK = True
    _COUNT_OK = True
except Exception:
    # fallback minimal: halaman tetap jalan, tapi tanpa stable order & type-aware stats
    _REDACT_OK = False
    _LEXICON_OK = False
    _STABLE_OK = False
    _COUNT_OK = False

    def replace_non_informative(text: str, *args, **kwargs):
        return text

    def count_placeholders(text: str):
        return {"total": 0, "by_type": {}, "raw_matches": []}

    def replace_and_count_placeholders(text: str, *args, **kwargs):
        return (text or ""), {"total": 0, "by_type": {}, "raw_matches": [], "by_regex": {}}

    def remove_single_char_tokens(text: str, *args, **kwargs):
        return text

    def stable_preprocess_order(text: str, **kwargs):
        return text

    WORD_NORMALIZATION_MAP = {"npwpd": "npwp", "pph21": "pph 21"}
    LEXICAL_NORMALIZATION_MAP = {r"\bwp\b": "wajib_pajak", r"\blh\b": "lebih_bayar"}


# ======================================================
# 🔐 DB CONFIG
# ======================================================
SCHEMA = "lasis_djp"
SRC_TABLE = "incident_kelayakan"
OUT_TABLE = "incident_normalized"

# Kolom output FIX (tidak versi)
TEXT_COL = "text_norm"
TS_COL = "tgl_normalized"


# ======================================================
# 🔌 DB ENGINE (SQLAlchemy 2.x safe)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True, future=True)


@st.cache_data(show_spinner=False)
def table_exists(schema: str, table: str) -> bool:
    eng = get_engine()
    q = text(
        """
        SELECT EXISTS (
          SELECT 1
          FROM information_schema.tables
          WHERE table_schema = :schema AND table_name = :table
        ) AS ok
        """
    )
    with eng.connect() as conn:
        return bool(conn.execute(q, {"schema": schema, "table": table}).scalar())


@st.cache_data(show_spinner=False)
def load_source_from_db(table: str = SRC_TABLE, schema: str = SCHEMA) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql_table(table, con=conn, schema=schema)


@st.cache_data(show_spinner=False)
def load_existing_normalized(schema: str = SCHEMA, table: str = OUT_TABLE) -> pd.DataFrame:
    eng = get_engine()
    with eng.connect() as conn:
        return pd.read_sql(text(f'SELECT * FROM "{schema}"."{table}"'), con=conn)


def save_replace(df: pd.DataFrame, table_name: str = OUT_TABLE, schema: str = SCHEMA):
    """Replace output table with latest normalized result."""
    eng = get_engine()
    df.to_sql(
        table_name,
        eng,
        schema=schema,
        if_exists="replace",
        index=False,
        chunksize=10_000,
        method="multi",
    )


# ======================================================
# 🔧 FALLBACK FUNCTIONS (dipakai jika stable_preprocess_order tidak tersedia)
# ======================================================
def strip_weird_chars(text_in: str, force_ascii: bool = False) -> str:
    if not isinstance(text_in, str):
        text_in = "" if pd.isna(text_in) else str(text_in)

    t = unicodedata.normalize("NFKC", text_in)
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)   # zero-width chars
    t = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", t)   # control chars
    if force_ascii:
        t = t.encode("ascii", "ignore").decode("ascii", "ignore")
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def apply_word_normalization(text_in: str) -> str:
    t = text_in
    for wrong, canon in WORD_NORMALIZATION_MAP.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"
        t = re.sub(pattern, canon, t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


def apply_lexical_normalization(text_in: str) -> str:
    t = text_in
    for pattern, repl in LEXICAL_NORMALIZATION_MAP.items():
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


def data_normalization(
    text_in: str,
    *,
    to_lower: bool = True,
    remove_punct: bool = True,
    remove_digits: bool = False,
    use_ftfy: bool = False,
    force_ascii: bool = False,
    use_word_norm: bool = True,
    use_lexical_norm: bool = True,
    use_redaction: bool = False,
    remove_single_chars: bool = False,
) -> str:
    """
    FINAL:
    - Jika stable_preprocess_order tersedia → pakai urutan paling stabil + placeholder type-aware.
    - Jika tidak tersedia → fallback ke urutan aman lokal (mendekati stable).
    """
    if _STABLE_OK:
        return stable_preprocess_order(
            text_in,
            use_ftfy=use_ftfy and _FTFY_AVAILABLE,
            ftfy_fix_func=_ftfy_fix if _FTFY_AVAILABLE else None,
            force_ascii=force_ascii,
            use_redaction=use_redaction and _REDACT_OK,
            to_lower=to_lower,
            use_word_norm=use_word_norm,
            use_lexical_norm=use_lexical_norm,
            remove_punct=remove_punct,
            remove_digits=remove_digits,
            remove_single_chars=remove_single_chars,
        )

    # -------- fallback (kalau helper tidak ada) --------
    if not isinstance(text_in, str):
        text_in = "" if pd.isna(text_in) else str(text_in)

    # 1) ftfy
    if use_ftfy and _FTFY_AVAILABLE:
        text_in = _ftfy_fix(text_in)

    # 2) weird chars
    text_in = strip_weird_chars(text_in, force_ascii=force_ascii)

    # 3) HTML
    text_in = re.sub(r"<[^>]+>", " ", text_in)

    # 4) placeholder replacement (fallback: single token jika ref belum dipatch)
    if use_redaction and _REDACT_OK:
        text_in = replace_non_informative(text_in)

    # 5) lower
    if to_lower:
        text_in = text_in.lower()

    # 6) word / lexical dulu (lebih stabil untuk pola seperti 4(2))
    if use_word_norm:
        text_in = apply_word_normalization(text_in)
    if use_lexical_norm:
        text_in = apply_lexical_normalization(text_in)

    # 7) punct/digits
    if remove_punct:
        text_in = re.sub(r"[^\w\s]", " ", text_in)
    if remove_digits:
        text_in = re.sub(r"\d+", " ", text_in)

    # 8) remove single char
    if remove_single_chars:
        text_in = remove_single_char_tokens(text_in, keep={"e"})

    return re.sub(r"\s+", " ", text_in).strip()


# ======================================================
# 🧾 UI HEADER
# ======================================================
st.title("🧹 Text Normalization — Insiden DJP")
st.caption(
    f"Sumber: `{SCHEMA}.{SRC_TABLE}` → Output: `{SCHEMA}.{OUT_TABLE}` "
    f"(kolom tetap `{TEXT_COL}`, `{TS_COL}`; setiap run akan overwrite)."
)

if not _STABLE_OK:
    st.info(
        "ℹ️ `stable_preprocess_order()` tidak terdeteksi. "
        "Halaman tetap berjalan dengan fallback. "
        "Untuk urutan paling stabil + placeholder bertipe, pastikan `pages/ref_normalization.py` versi terbaru sudah terpasang."
    )

# ======================================================
# 📦 LOAD SOURCE
# ======================================================
with st.spinner(f"📦 Memuat data dari `{SCHEMA}.{SRC_TABLE}` ..."):
    try:
        df_src = load_source_from_db()
        st.success(f"Berhasil memuat {len(df_src):,} baris.")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

if df_src.empty:
    st.warning("Dataset kosong.")
    st.stop()

# ======================================================
# 🔍 DETEKSI KOLOM TEKS
# ======================================================
candidate_text_cols = [
    "isi_permasalahan",
    "detailed_description",
    "detailed_decription",
    "deskripsi",
]
SOURCE_TEXT_COL = next((c for c in candidate_text_cols if c in df_src.columns), None)

if SOURCE_TEXT_COL is None:
    st.error(
        "Tidak menemukan kolom deskripsi teks. "
        "Kolom yang dicari: isi_permasalahan, detailed_description/detailed_decription, atau deskripsi."
    )
    st.stop()

if "incident_number" not in df_src.columns:
    st.error("Kolom 'incident_number' tidak ditemukan.")
    st.stop()

# ======================================================
# 🧭 SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Text Normalization")

    with st.form("norm_form"):
        st.markdown("### ✂️ Replace / Redaction (Placeholder)")
        use_redaction = st.checkbox(
            "Aktifkan replace/redaction (NPWP, NIK, NOP, URL, EMAIL, IP, dst)",
            value=_REDACT_OK,
        )
        if not _REDACT_OK:
            st.caption(
                "⚠️ Fungsi redaction tidak ditemukan (pages.ref_normalization). "
                "Mode redaction akan di-skip."
            )

        st.markdown("---")
        st.markdown("### Normalization")
        to_lower = st.checkbox("Lowercasing", True)

        # default remove_punct True, tapi urutan pemrosesan stabil memastikan
        # word/lexical dilakukan sebelum remove_punct (via stable_preprocess_order).
        remove_punct = st.checkbox("Hapus tanda baca", True)

        remove_digits = st.checkbox("Hapus angka", False)
        remove_single_chars = st.checkbox("Hapus token 1 karakter (mis: a, n)", True)

        use_ftfy = st.checkbox("Perbaiki mojibake (ftfy)", value=_FTFY_AVAILABLE)
        force_ascii = st.checkbox("Paksa ASCII only", False)

        use_word_norm = st.checkbox("Word Normalization (kamus)", True)
        use_lexical_norm = st.checkbox("Lexical Normalization (kamus)", True)

        if not _LEXICON_OK:
            st.caption("ℹ️ Kamus tidak ditemukan, menggunakan fallback.")

        st.markdown("---")
        run = st.form_submit_button("🚀 Jalankan Normalisasi", use_container_width=True)

# ======================================================
# 👀 PREVIEW BEFORE
# ======================================================
st.subheader("Preview Data (Sebelum)")
st.dataframe(df_src[["incident_number", SOURCE_TEXT_COL]].head(10), use_container_width=True)

if not run:
    st.info("Atur parameter di sidebar lalu klik **Jalankan Normalisasi**.")
    st.stop()

# ======================================================
# 🧮 NORMALIZE WITH PROGRESS BAR
# ======================================================
WIB = timezone(timedelta(hours=7))
run_dt = datetime.now(WIB).replace(tzinfo=None)  # simpan tanpa tz (lebih kompatibel)

texts = df_src[SOURCE_TEXT_COL].fillna("").astype(str).tolist()
n = len(texts)

progress = st.progress(0, text="Menyiapkan proses normalisasi...")
status = st.empty()
status.info("⚙️ Menjalankan normalisasi teks...")

out = [""] * n
update_every = max(200, n // 200)  # update ~200 kali

for i, raw_txt in enumerate(texts):
    out[i] = data_normalization(
        raw_txt,
        to_lower=to_lower,
        remove_punct=remove_punct,
        remove_digits=remove_digits,
        use_ftfy=use_ftfy,
        force_ascii=force_ascii,
        use_word_norm=use_word_norm,
        use_lexical_norm=use_lexical_norm,
        use_redaction=use_redaction,
        remove_single_chars=remove_single_chars,
    )

    if (i + 1) % update_every == 0 or (i + 1) == n:
        p = int(((i + 1) / n) * 100)
        progress.progress((i + 1) / n, text=f"Normalisasi berjalan... {p}% ({i+1:,}/{n:,})")

status.success("✅ Normalisasi selesai.")

# ======================================================
# 🧩 BUILD OUTPUT (KEEP META + text_norm + tgl_normalized)
# ======================================================
df_out = df_src.copy()
df_out[TEXT_COL] = out
df_out[TS_COL] = pd.Timestamp(run_dt)

# ======================================================
# 📈 STATISTIK REDACTION (type-aware; aman utk data besar)
# ======================================================
if use_redaction and _REDACT_OK:
    st.subheader("📊 Statistik Replace/Redaction (Placeholder)")

    max_scan = 50_000
    scan_series = df_src[SOURCE_TEXT_COL].fillna("").astype(str)
    if len(scan_series) > max_scan:
        scan_series = scan_series.sample(max_scan, random_state=42)
        st.caption(f"Perhitungan statistik memakai sample {max_scan:,} baris (dari {len(df_src):,}).")

    agg = {}

    if _COUNT_OK:
        # ✅ stats lengkap: angle placeholder + URL/EMAIL/IP/LONGNUM
        for raw_txt in scan_series.tolist():
            _, stats = replace_and_count_placeholders(raw_txt)
            by_type = stats.get("by_type") or {}
            by_regex = stats.get("by_regex") or {}

            for k, v in by_type.items():
                agg[k] = agg.get(k, 0) + int(v)
            for k, v in by_regex.items():
                agg[k] = agg.get(k, 0) + int(v)
    else:
        # fallback: hanya hitung angle placeholder
        for raw_txt in scan_series.tolist():
            stats = count_placeholders(raw_txt)
            by_type = stats.get("by_type") or {}
            for k, v in by_type.items():
                agg[k] = agg.get(k, 0) + int(v)

    if agg:
        stats_df = (
            pd.DataFrame([{"token": k, "jumlah": v} for k, v in agg.items()])
            .sort_values("jumlah", ascending=False, ignore_index=True)
        )
        st.dataframe(stats_df, use_container_width=True)
    else:
        st.info("Tidak ada placeholder terdeteksi (atau fungsi redaction sedang fallback).")

# ======================================================
# 💾 SAVE OUTPUT (REPLACE)
# ======================================================
try:
    with st.spinner(f"💾 Menyimpan ke `{SCHEMA}.{OUT_TABLE}` (replace) ..."):
        save_replace(df_out, OUT_TABLE, SCHEMA)
    st.success(f"Berhasil menyimpan ke `{SCHEMA}.{OUT_TABLE}`. Kolom output: `{TEXT_COL}`, `{TS_COL}`.")
except Exception as e:
    st.error(f"Gagal menyimpan ke database: {e}")
    st.stop()

# ======================================================
# 📊 PREVIEW AFTER + STATS
# ======================================================
st.subheader("Preview Hasil Normalisasi (Sesudah)")
st.dataframe(df_out[["incident_number", SOURCE_TEXT_COL, TEXT_COL, TS_COL]].head(10), use_container_width=True)

st.markdown("### Statistik Panjang Teks Normalized")
length_series = df_out[TEXT_COL].astype(str).str.len()
st.write(length_series.describe()[["min", "mean", "max"]])

st.markdown("#### Bar Chart Panjang Teks (sample 1.000)")
sample_for_bar = length_series.sample(min(1000, len(length_series)), random_state=42)
st.bar_chart(sample_for_bar)

st.markdown("#### Histogram Distribusi Panjang Teks")
len_df = pd.DataFrame({"length": length_series})
hist = alt.Chart(len_df).mark_bar().encode(
    x=alt.X("length:Q", bin=alt.Bin(maxbins=50), title="Panjang Teks (karakter)"),
    y=alt.Y("count()", title="Jumlah Tiket"),
)
st.altair_chart(hist, use_container_width=True)

# ======================================================
# 📥 DOWNLOAD CSV
# ======================================================
csv_bytes = df_out[["incident_number", TEXT_COL, TS_COL]].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Output Normalisasi (CSV)",
    data=csv_bytes,
    file_name="incident_normalized.csv",
    mime="text/csv",
)
