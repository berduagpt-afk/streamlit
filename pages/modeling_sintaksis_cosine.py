# pages/modeling_sintaksis_cosine.py
# Proses Modeling – Pendekatan Sintaksis (Cosine Similarity)

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()

# ======================================================
# 🔌 Koneksi PostgreSQL
# ======================================================
def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(show_spinner=False)
def load_incident_clean(schema="lasis_djp", table="incident_clean") -> pd.DataFrame:
    eng = get_connection()
    try:
        df = pd.read_sql_table(table, con=eng, schema=schema)
    finally:
        eng.dispose()
    return df


# ======================================================
# 🧭 Setup halaman
# ======================================================
st.title("🧩 Proses Modeling – Pendekatan Sintaksis (Cosine Similarity)")
st.caption(
    "Mengukur kemiripan antar tiket insiden berdasarkan representasi TF-IDF "
    "dari kolom **text_sintaksis** pada `lasis_djp.incident_clean`."
)

# ======================================================
# 📦 Load data
# ======================================================
with st.spinner("📦 Memuat data incident_clean..."):
    try:
        df = load_incident_clean()
    except Exception as e:
        st.error(f"Gagal memuat data dari database: {e}")
        st.stop()

if df.empty:
    st.warning("Dataset incident_clean kosong. Jalankan dulu tahap preprocessing.")
    st.stop()

required_cols = ["incident_number", "text_sintaksis"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Kolom wajib tidak ditemukan: {', '.join(missing)}")
    st.stop()

# Pastikan teks string
df["text_sintaksis"] = df["text_sintaksis"].fillna("").astype(str)

# ======================================================
# 🧹 Cleaning ringan supaya kata aneh (ðÿ, œdata, dst.) hilang
# ======================================================
def clean_for_tfidf(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    # hanya huruf/angka/spasi
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["text_sintaksis_clean"] = df["text_sintaksis"].apply(clean_for_tfidf)

# Buang dokumen yang benar-benar kosong setelah cleaning
df = df[df["text_sintaksis_clean"].str.strip() != ""].copy()
if df.empty:
    st.error("Semua teks menjadi kosong setelah proses cleaning. Periksa kembali preprocessing.")
    st.stop()

# ======================================================
# 🔍 Filter modul & rentang tanggal (opsional)
# ======================================================
st.subheader("Filter Data Sebelum Modeling (Opsional)")

with st.expander("Tampilkan filter data (modul / rentang waktu)", expanded=False):

    # Filter modul (jika ada)
    if "modul" in df.columns:
        all_modul = (
            df["modul"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        selected_modul = st.multiselect(
            "Pilih modul (kosongkan jika ingin semua modul)",
            options=all_modul,
        )
        if selected_modul:
            df = df[df["modul"].astype(str).isin(selected_modul)].copy()
    else:
        st.info("Kolom **modul** tidak ditemukan, filter modul dilewati.")

    # Filter tanggal (jika ada tgl_submit)
    if "tgl_submit" in df.columns:
        df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
        valid_dates = df["tgl_submit"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            start_date, end_date = st.date_input(
                "Rentang tanggal berdasarkan kolom tgl_submit",
                value=(min_date, max_date),
            )
            if start_date > end_date:
                st.warning("Tanggal awal > tanggal akhir, rentang tidak diterapkan.")
            else:
                mask = df["tgl_submit"].dt.date.between(start_date, end_date)
                df = df[mask].copy()
        else:
            st.info("Semua nilai tgl_submit tidak valid/NaT, filter tanggal dilewati.")
    else:
        st.info("Kolom **tgl_submit** tidak ditemukan, filter tanggal dilewati.")

if df.empty:
    st.error("Tidak ada data tersisa setelah filter. Atur ulang filter Anda.")
    st.stop()

st.write(f"Jumlah tiket setelah filter: **{len(df):,}**")

# ======================================================
# ⚙️ Parameter TF-IDF & Cosine Similarity (Sidebar)
# ======================================================
with st.sidebar:
    st.header("⚙️ Parameter Modeling Sintaksis")

    max_features = st.number_input(
        "Maksimal jumlah fitur TF-IDF",
        min_value=1000,
        max_value=50_000,
        value=5_000,
        step=1000,
    )

    ngram_max = st.selectbox(
        "N-gram maksimum",
        options=[1, 2],
        index=0,
        format_func=lambda x: "Unigram" if x == 1 else "Unigram + Bigram",
    )

    min_df = st.number_input(
        "min_df (dokumen minimal suatu term muncul)",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
    )

    max_df_ratio = st.slider(
        "max_df (proporsi dokumen maksimal term muncul)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )

    top_k = st.slider(
        "Jumlah tiket paling mirip yang ditampilkan",
        min_value=5,
        max_value=100,
        value=20,
        step=5,
    )

    threshold = st.slider(
        "Threshold Cosine Similarity (untuk penyaringan)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

# ======================================================
# 🔢 Hitung TF-IDF (sekali untuk seluruh data terfilter)
# ======================================================
with st.spinner("🔢 Menghitung TF-IDF untuk pendekatan sintaksis..."):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df_ratio,
        use_idf=True,
        norm="l2",
        lowercase=False,  # sudah dilowercase di cleaning
        token_pattern=r"(?u)\b[a-z0-9]{2,}\b",  # hanya token >=2
    )

    X = vectorizer.fit_transform(df["text_sintaksis_clean"].tolist())
    n_docs, n_terms = X.shape

st.success(f"TF-IDF berhasil dihitung: {n_docs} dokumen × {n_terms} fitur aktif.")

# ======================================================
# 🎯 Pilih tiket anchor
# ======================================================
st.subheader("Pilih Tiket Anchor untuk Mengukur Kemiripan")

# Dropdown berdasarkan incident_number + (optional) judul/teks pendek
df["display_label"] = df["incident_number"].astype(str)
if "judul_masalah" in df.columns:
    df["display_label"] = df["display_label"] + " – " + df["judul_masalah"].fillna("").str.slice(0, 60)

anchor_label = st.selectbox(
    "Pilih tiket anchor",
    options=df["display_label"].tolist(),
)

anchor_idx = df.index[df["display_label"] == anchor_label][0]

st.markdown("**Teks anchor (text_sintaksis):**")
st.write(df.loc[anchor_idx, "text_sintaksis"])

# ======================================================
# 🧮 Hitung Cosine Similarity anchor vs lainnya
# ======================================================
if st.button("🚀 Hitung Kemiripan dengan Cosine Similarity", use_container_width=True):
    with st.spinner("Menghitung kemiripan anchor dengan seluruh tiket..."):
        # cosine_similarity antara 1 dokumen vs seluruh dokumen
        anchor_vec = X[ df.index.get_loc(anchor_idx) ]  # posisi baris di X
        sims = cosine_similarity(anchor_vec, X).ravel()

        # susun DataFrame hasil
        result_df = df.copy()
        result_df["cosine_similarity"] = sims

        # urutkan dari yang paling mirip
        result_df = result_df.sort_values("cosine_similarity", ascending=False)

        # buang dirinya sendiri (cosine = 1)
        result_df = result_df[result_df["incident_number"] != df.loc[anchor_idx, "incident_number"]]

        # ambil top_k
        top_df = result_df.head(top_k).copy()

        # terapkan threshold untuk display
        above_th = top_df[top_df["cosine_similarity"] >= threshold].copy()

        st.subheader("Distribusi Cosine Similarity (Anchor vs Semua Dokumen)")
        hist_chart = alt.Chart(result_df).mark_bar().encode(
            x=alt.X("cosine_similarity:Q", bin=alt.Bin(maxbins=30), title="Cosine Similarity"),
            y=alt.Y("count():Q", title="Jumlah Dokumen"),
            tooltip=["count()"]
        ).properties(height=300)
        st.altair_chart(hist_chart, use_container_width=True)

        st.markdown(
            f"**Top {top_k} tiket paling mirip** (tanpa filter threshold). "
            f"Baris dengan Cosine ≥ {threshold:.2f} dianggap *cukup mirip* menurut parameter saat ini."
        )

        show_cols = ["incident_number", "cosine_similarity"]
        for c in ["modul", "site", "judul_masalah", "text_sintaksis"]:
            if c in top_df.columns:
                show_cols.append(c)

        st.dataframe(
            top_df[show_cols],
            use_container_width=True,
        )

        st.subheader(f"Tiket Paling Mirip dengan Cosine ≥ {threshold:.2f}")
        if above_th.empty:
            st.info("Tidak ada tiket dalam Top-N yang melampaui threshold saat ini. Coba turunkan threshold.")
        else:
            st.dataframe(
                above_th[show_cols],
                use_container_width=True,
            )

        st.markdown(
            "Catatan: Threshold Cosine Similarity inilah yang nanti bisa kamu bahas sebagai "
            "**parameter evaluasi** pada pendekatan sintaksis, misalnya apakah 0,6 atau 0,7 "
            "memberikan pasangan tiket yang secara substantif benar-benar mirip menurut pakar DJP."
        )
else:
    st.info("Pilih tiket anchor lalu klik **Hitung Kemiripan dengan Cosine Similarity**.")
