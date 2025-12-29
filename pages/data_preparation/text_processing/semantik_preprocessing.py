# pages/preprocessing.py
# Preprocessing Data Insiden DJP — sesuai pipeline Data Preparation

import re
import unicodedata
from datetime import datetime, timezone, timedelta
from collections import Counter

import pandas as pd
import streamlit as st
import altair as alt
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sqlalchemy import create_engine

# ======================================================
# 🔐 Guard login (opsional, ikuti pola halaman lain)
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
# 📚 Kamus & Redaction (semua di pages/ref_normalization.py)
# ======================================================

# --- 1) FUNGSI REDACTION (NPWP, NIK, NOP, NAMA_WP, dll) ---
try:
    from pages.ref_normalization import (
        replace_non_informative,
        count_placeholders,
        remove_single_char_tokens,
    )

    _REDACT_OK = True
except Exception:
    _REDACT_OK = False

    def replace_non_informative(text: str, *args, **kwargs):
        return text

    def count_placeholders(text: str):
        return {"total": 0, "by_type": {}, "raw_matches": []}

    def remove_single_char_tokens(text: str, *args, **kwargs):
        return text

# --- 2) KAMUS WORD & LEXICAL NORMALIZATION ---
try:
    from pages.ref_normalization import WORD_NORMALIZATION_MAP, LEXICAL_NORMALIZATION_MAP

    _LEXICON_OK = True
except Exception:
    _LEXICON_OK = False
    WORD_NORMALIZATION_MAP = {
        "npwpd": "npwp",
        "pph21": "pph 21",
    }
    LEXICAL_NORMALIZATION_MAP = {
        r"\bwp\b": "wajib pajak",
        r"\blh\b": "lebih bayar",
    }

# ======================================================
# 🔐 KONEKSI DATABASE
# ======================================================
def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(show_spinner=False)
def load_from_db(table="incident_kelayakan", schema="lasis_djp") -> pd.DataFrame:
    eng = get_connection()
    try:
        df = pd.read_sql_table(table, con=eng, schema=schema)
    finally:
        eng.dispose()
    return df


def save_dataframe(df: pd.DataFrame, table_name="incident_clean", schema="lasis_djp"):
    eng = get_connection()
    try:
        df.to_sql(
            table_name,
            eng,
            schema=schema,
            if_exists="replace",  # overwrite
            index=False,
            chunksize=10_000,
        )
    finally:
        eng.dispose()


# ======================================================
# ⚙️ SETUP HALAMAN
# ======================================================
st.title("🧹 Preprocessing Teks Insiden")
st.caption(
    "Pipeline Data Preparation: Data Normalization → Pendekatan Sintaksis (TF-IDF) "
    "dan Pendekatan Semantik (IndoBERT). Sumber data: `lasis_djp.incident_kelayakan`, "
    "hasil disimpan ke `lasis_djp.incident_clean`."
)

# ======================================================
# 📦 LOAD DATA
# ======================================================
with st.spinner("📦 Memuat data dari incident_kelayakan..."):
    try:
        df = load_from_db()
        st.success(f"Berhasil memuat {len(df):,} baris.")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

if df.empty:
    st.warning("Dataset kosong.")
    st.stop()

BASE_COLUMNS = df.columns.tolist()

# ======================================================
# 🔍 DETEKSI KOLOM TEKS
# ======================================================
candidate_text_cols = ["isi_permasalahan", "detailed_decription", "deskripsi"]
SOURCE_TEXT_COL = next((c for c in candidate_text_cols if c in df.columns), None)

if SOURCE_TEXT_COL is None:
    st.error(
        "Tidak menemukan kolom deskripsi teks. "
        "Kolom yang dicari: isi_permasalahan, detailed_decription, atau deskripsi."
    )
    st.stop()

if "incident_number" not in df.columns:
    st.error("Kolom 'incident_number' tidak ditemukan.")
    st.stop()

# ======================================================
# 🧩 ALAT NLP: Stopword & Stemmer (Sastrawi)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_nlp_tools():
    stop_factory = StopWordRemoverFactory()
    stem_factory = StemmerFactory()
    stopwords = set(stop_factory.get_stop_words())
    stemmer = stem_factory.create_stemmer()
    return stopwords, stemmer


STOPWORDS_SET, stemmer = get_nlp_tools()

# ======================================================
# 🔧 FUNGSI NORMALISASI
# ======================================================
def strip_weird_chars(text: str, force_ascii: bool = False) -> str:
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    t = unicodedata.normalize("NFKC", text)
    t = re.sub(r"[\u200B-\u200D\uFEFF]", "", t)
    t = re.sub(r"[\x00-\x1F\x7F-\x9F]", " ", t)
    if force_ascii:
        t = t.encode("ascii", "ignore").decode("ascii", "ignore")
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def apply_word_normalization(text: str) -> str:
    t = text
    for wrong, canon in WORD_NORMALIZATION_MAP.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"
        t = re.sub(pattern, canon, t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


def apply_lexical_normalization(text: str) -> str:
    t = text
    for pattern, repl in LEXICAL_NORMALIZATION_MAP.items():
        t = re.sub(pattern, repl, t, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", t).strip()


def data_normalization(
    text: str,
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
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    if use_ftfy and _FTFY_AVAILABLE:
        text = _ftfy_fix(text)

    text = strip_weird_chars(text, force_ascii=force_ascii)

    if use_redaction and _REDACT_OK:
        text = replace_non_informative(text)

    # hapus HTML
    text = re.sub(r"<[^>]+>", " ", text)

    if to_lower:
        text = text.lower()

    if remove_punct:
        text = re.sub(r"[^\w\s]", " ", text)
    if remove_digits:
        text = re.sub(r"\d+", " ", text)

    if use_word_norm:
        text = apply_word_normalization(text)

    if use_lexical_norm:
        text = apply_lexical_normalization(text)

    # ✅ hapus token 1 karakter (mis: a, n)
    if remove_single_chars:
        # keep={"e"} supaya token 'e' tidak hilang bila muncul sendiri
        text = remove_single_char_tokens(text, keep={"e"})

    text = re.sub(r"\s+", " ", text).strip()
    return text


# ======================================================
# 🌿 Syntactic & Semantic Pipelines
# ======================================================
def syntactic_pipeline(norm_text: str, *, do_stopword: bool = True, do_stemming: bool = True):
    tokens = re.findall(r"\b\w+\b", norm_text)

    if do_stopword:
        tokens = [t for t in tokens if t.lower() not in STOPWORDS_SET]

    # ✅ pastikan token 1 karakter dibuang untuk TF-IDF
    tokens = [t for t in tokens if len(t) > 1]

    if do_stemming:
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens


def semantic_pipeline(norm_text: str):
    return norm_text.split()


# ======================================================
# 🧭 SIDEBAR PENGATURAN
# ======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Data Normalization")

    with st.form("prep_form"):
        st.markdown("### ✂️ Text Redaction (Non-informatif)")
        use_redaction = st.checkbox(
            "Aktifkan text redaction (NPWP, NIK, NOP, Nama WP, dll)",
            value=_REDACT_OK,
        )
        if not _REDACT_OK:
            st.caption(
                "⚠️ Fungsi `replace_non_informative()` / `count_placeholders()` "
                "tidak ditemukan. Redaction akan di-skip."
            )

        st.markdown("---")
        st.markdown("### Normalization")
        to_lower = st.checkbox("Lowercasing", True)
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
        st.markdown("### Pendekatan Sintaksis")
        syn_stopword = st.checkbox("Stopword Removal (Sintaksis)", True)
        syn_stemming = st.checkbox("Stemming (Sastrawi)", True)

        st.markdown("### Pendekatan Semantik")
        sem_show_tokens = st.checkbox("Tampilkan preview token semantik", False)

        run = st.form_submit_button("🚀 Jalankan Preprocessing", use_container_width=True)

# ======================================================
# 👀 PREVIEW DATA SEBELUM PROSES
# ======================================================
st.subheader("Preview Data (Sebelum)")
st.dataframe(df[["incident_number", SOURCE_TEXT_COL]].head(10), use_container_width=True)

if not run:
    st.info("Silakan atur parameter di sidebar lalu klik **Jalankan Preprocessing**.")
    st.stop()

# ======================================================
# 🧮 JALANKAN PIPELINE
# ======================================================
with st.spinner("⚙️ Menjalankan pipeline preprocessing..."):
    df["text_normalized"] = df[SOURCE_TEXT_COL].apply(
        lambda txt: data_normalization(
            txt,
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
    )

    # Sintaksis
    df["tokens_sintaksis"] = df["text_normalized"].apply(
        lambda t: syntactic_pipeline(t, do_stopword=syn_stopword, do_stemming=syn_stemming)
    )
    df["text_sintaksis"] = df["tokens_sintaksis"].apply(lambda toks: " ".join(toks))

    # Semantik
    df["text_semantik"] = df["text_normalized"]
    if sem_show_tokens:
        df["tokens_semantik"] = df["text_semantik"].apply(semantic_pipeline)

    WIB = timezone(timedelta(hours=7))
    df["tgl_preprocessed"] = datetime.now(WIB)

# ======================================================
# 📈 STATISTIK TEXT REDACTION (opsional, untuk tesis)
# ======================================================
if use_redaction and _REDACT_OK:
    st.subheader("📊 Statistik Text Redaction (Placeholder)")
    agg_counts = Counter()

    for raw_txt in df[SOURCE_TEXT_COL].fillna(""):
        stats = count_placeholders(str(raw_txt))
        for token, cnt in (stats.get("by_type") or {}).items():
            agg_counts[token] += cnt

    if agg_counts:
        stats_df = (
            pd.DataFrame([{"token": k, "jumlah": v} for k, v in agg_counts.items()])
            .sort_values("jumlah", ascending=False, ignore_index=True)
        )
        st.dataframe(stats_df, use_container_width=True)

# ======================================================
# 💾 SIMPAN KE DATABASE
# ======================================================
meta_cols = [c for c in BASE_COLUMNS if c in df.columns]
cols_to_save = meta_cols + [
    "text_normalized",
    "text_sintaksis",
    "text_semantik",
    "tgl_preprocessed",
]

df_save = df[cols_to_save].copy()

try:
    with st.spinner("💾 Menyimpan hasil preprocessing ke lasis_djp.incident_clean..."):
        save_dataframe(df_save, "incident_clean", "lasis_djp")
    st.success("Berhasil menyimpan hasil preprocessing ke `lasis_djp.incident_clean`.")
except Exception as e:
    st.error(f"Gagal menyimpan ke database: {e}")

# ======================================================
# 📊 OUTPUT SESUDAH
# ======================================================
st.subheader("Preview Hasil Preprocessing")

st.markdown("#### Pendekatan Sintaksis (untuk TF-IDF)")
st.dataframe(
    df[["incident_number", "text_sintaksis", "tokens_sintaksis"]].head(10),
    use_container_width=True,
)

st.markdown("#### Pendekatan Semantik (untuk IndoBERT)")
cols_sem = ["incident_number", "text_semantik"]
if "tokens_semantik" in df.columns:
    cols_sem.append("tokens_semantik")
st.dataframe(df[cols_sem].head(10), use_container_width=True)

c1, c2 = st.columns(2)
c1.metric("Jumlah baris", f"{df.shape[0]:,}")
c2.metric("Jumlah kolom (termasuk kolom hasil)", f"{df.shape[1]:,}")

# ======================================================
# 📏 STATISTIK PANJANG TEKS NORMALIZED
# ======================================================
st.markdown("### Statistik Panjang Teks Normalized")

length_series = df["text_normalized"].str.len()
st.write(length_series.describe()[["min", "mean", "max"]])

st.markdown("#### Bar Chart Panjang Teks per Tiket (sample 1.000)")
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
csv_bytes = df_save.to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇️ Download Dataset Preprocessed (CSV)",
    data=csv_bytes,
    file_name="dataset_preprocessed_pipeline.csv",
    mime="text/csv",
)
