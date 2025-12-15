# pages/evaluasi_cosine_threshold.py
# Evaluasi Variasi Cosine Similarity Threshold (Pendekatan Sintaksis)

import re
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
st.title("📏 Evaluasi Variasi Cosine Similarity Threshold")
st.caption(
    "Halaman ini digunakan untuk mengevaluasi beberapa nilai ambang "
    "Cosine Similarity sekaligus, agar dapat menentukan threshold yang "
    "paling sesuai untuk pendekatan sintaksis."
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

# pastikan string
df["text_sintaksis"] = df["text_sintaksis"].fillna("").astype(str)

# ======================================================
# 🧹 Cleaning ringan untuk TF-IDF
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
df = df[df["text_sintaksis_clean"].str.strip() != ""].copy()

if df.empty:
    st.error("Semua teks menjadi kosong setelah proses cleaning. Periksa kembali preprocessing.")
    st.stop()

# ======================================================
# 🔍 Filter modul & rentang tanggal (opsional)
# ======================================================
st.subheader("Filter Data untuk Evaluasi")

col_f1, col_f2 = st.columns(2)

with col_f1:
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
            "Pilih modul (kosongkan = semua modul)",
            options=all_modul,
        )
        if selected_modul:
            df = df[df["modul"].astype(str).isin(selected_modul)].copy()
            st.info(
                "Catatan: jika ingin evaluasi threshold per modul, "
                "pilih hanya satu modul di sini lalu jalankan evaluasi."
            )
    else:
        st.info("Kolom **modul** tidak ditemukan, filter modul dilewati.")

with col_f2:
    # Filter tanggal (jika ada tgl_submit)
    if "tgl_submit" in df.columns:
        df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
        valid_dates = df["tgl_submit"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            start_date, end_date = st.date_input(
                "Rentang tanggal berdasarkan tgl_submit",
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
# ⚙️ Parameter Evaluasi (Sidebar)
# ======================================================
with st.sidebar:
    st.header("⚙️ Parameter Evaluasi Cosine Threshold")

    max_docs = st.number_input(
        "Maksimum tiket yang dianalisis (sampling acak)",
        min_value=200,
        max_value=min(50_000, len(df)),
        value=min(5_000, len(df)),
        step=200,
        help="Jika data sangat besar, analisis akan memakai sampel acak."
    )

    n_anchors = st.number_input(
        "Jumlah tiket sample sebagai anchor",
        min_value=20,
        max_value=1_000,
        value=200,
        step=10,
        help="Semakin banyak anchor, evaluasi makin stabil namun perhitungan lebih berat."
    )

    max_features = st.number_input(
        "Maksimal fitur TF-IDF",
        min_value=1_000,
        max_value=50_000,
        value=10_000,
        step=1_000,
    )

    ngram_max = st.selectbox(
        "N-gram maksimum",
        options=[1, 2],
        index=1,
        format_func=lambda x: "Unigram saja" if x == 1 else "Unigram + Bigram",
    )

    min_df = st.number_input(
        "min_df (dokumen minimal term muncul)",
        min_value=1,
        max_value=100,
        value=3,
        step=1,
    )

    max_df_ratio = st.slider(
        "max_df (proporsi maksimum dokumen term muncul)",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
    )

    # Daftar threshold yang akan dievaluasi
    default_thresholds = [0.40, 0.50, 0.60, 0.70, 0.80]
    selected_thresholds = st.multiselect(
        "Daftar threshold yang dievaluasi",
        options=[round(x, 2) for x in np.arange(0.30, 0.95, 0.05)],
        default=default_thresholds,
        help="Pilih satu atau beberapa nilai ambang cosine untuk dibandingkan."
    )

    selected_thresholds = sorted(selected_thresholds)

# ======================================================
# 🎲 Sampling data (kalau perlu)
# ======================================================
if len(df) > max_docs:
    df_sample = df.sample(n=int(max_docs), random_state=42).copy()
    st.info(
        f"Dataset berisi {len(df):,} tiket. "
        f"Diambil sampel acak sebanyak {len(df_sample):,} tiket untuk evaluasi."
    )
else:
    df_sample = df.copy()

n_docs = len(df_sample)
if n_anchors > n_docs:
    st.warning(
        f"Jumlah anchor ({n_anchors}) melebihi jumlah dokumen ({n_docs}). "
        f"Jumlah anchor disesuaikan menjadi {n_docs}."
    )
    n_anchors = n_docs

# ======================================================
# 🔢 TF-IDF
# ======================================================
with st.spinner("🔢 Menghitung representasi TF-IDF..."):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df_ratio,
        use_idf=True,
        norm="l2",
        lowercase=False,
        token_pattern=r"(?u)\b[a-z0-9]{2,}\b",
    )
    X = vectorizer.fit_transform(df_sample["text_sintaksis_clean"].tolist())
    n_docs, n_terms = X.shape

st.success(f"TF-IDF selesai: {n_docs} dokumen × {n_terms} fitur aktif.")

# ======================================================
# 🧪 Evaluasi Threshold
# ======================================================
if st.button("🚀 Jalankan Evaluasi Threshold", use_container_width=True):
    if not selected_thresholds:
        st.error("Pilih minimal satu nilai threshold terlebih dahulu.")
        st.stop()

    with st.spinner("Menghitung distribusi cosine similarity dan evaluasi threshold..."):

        # pilih beberapa dokumen sebagai anchor secara acak
        anchor_indices = np.random.choice(n_docs, size=int(n_anchors), replace=False)
        # cosine anchor vs semua dokumen
        sims = cosine_similarity(X[anchor_indices], X)  # shape: (n_anchors, n_docs)

        # buang self-similarity (anchor ke dirinya sendiri)
        for i, idx in enumerate(anchor_indices):
            sims[i, idx] = 0.0

        # ringkasan distribusi keseluruhan (bisa dipakai untuk narasi)
        flat_sims = sims.ravel()
        desc = pd.Series(flat_sims).describe(
            percentiles=[0.25, 0.5, 0.75, 0.9, 0.95]
        )
        st.subheader("Ringkasan Distribusi Cosine Similarity (Anchor vs Semua Tiket)")
        st.write(desc)

        # histogram distribusi cosine
        hist_df = pd.DataFrame({"cosine_similarity": flat_sims})
        hist_chart = (
            alt.Chart(hist_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "cosine_similarity:Q",
                    bin=alt.Bin(maxbins=40),
                    title="Cosine Similarity"
                ),
                y=alt.Y("count():Q", title="Jumlah Pasangan"),
                tooltip=["count()"]
            )
            .properties(height=300)
        )
        st.altair_chart(hist_chart, use_container_width=True)

        # ==============================================
        # Hitung metrik untuk tiap threshold yang dipilih
        # ==============================================
        rows = []
        for th in selected_thresholds:
            mask = sims >= th

            total_pairs = int(mask.sum())
            if total_pairs > 0:
                mean_cos = float(sims[mask].mean())
            else:
                mean_cos = 0.0

            # berapa banyak anchor yang minimal punya 1 pasangan di atas threshold
            anchors_with_match = int((mask.sum(axis=1) > 0).sum())
            coverage = anchors_with_match / sims.shape[0]

            rows.append(
                {
                    "threshold": th,
                    "total_pairs": total_pairs,
                    "mean_cosine": round(mean_cos, 4),
                    "anchors_with_match": anchors_with_match,
                    "coverage_anchor": round(coverage, 4),
                }
            )

        eval_df = pd.DataFrame(rows).sort_values("threshold")

        st.subheader("Hasil Evaluasi Variasi Cosine Similarity Threshold")
        st.dataframe(
            eval_df.style.format(
                {
                    "threshold": "{:.2f}",
                    "mean_cosine": "{:.4f}",
                    "coverage_anchor": "{:.2%}",
                }
            ),
            use_container_width=True,
        )

        # ==============================================
        # Visualisasi: coverage & total_pairs vs threshold
        # ==============================================
        st.subheader("Grafik Evaluasi Threshold")

        chart_data = eval_df.copy()
        chart_data["coverage_percent"] = chart_data["coverage_anchor"] * 100

        # coverage vs threshold
        coverage_chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title="Threshold Cosine Similarity"),
                y=alt.Y("coverage_percent:Q", title="% Anchor yang Punya Pasangan ≥ Threshold"),
                tooltip=["threshold", "coverage_percent"]
            )
            .properties(height=300, title="Coverage Anchor vs Threshold")
        )

        # total pairs vs threshold
        pairs_chart = (
            alt.Chart(chart_data)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title="Threshold Cosine Similarity"),
                y=alt.Y("total_pairs:Q", title="Jumlah Pasangan di Atas Threshold"),
                tooltip=["threshold", "total_pairs"]
            )
            .properties(height=300, title="Jumlah Pasangan vs Threshold")
        )

        st.altair_chart(coverage_chart, use_container_width=True)
        st.altair_chart(pairs_chart, use_container_width=True)

        st.markdown(
            """
**Interpretasi singkat:**

- Kolom `coverage_anchor` menunjukkan seberapa banyak tiket (anchor) yang memiliki
  minimal satu pasangan mirip pada threshold tertentu.
- Kolom `total_pairs` menunjukkan jumlah semua pasangan (anchor, tiket lain)
  yang memiliki nilai cosine ≥ threshold.
- Nilai threshold yang terlalu rendah → coverage tinggi, tetapi `total_pairs` sangat besar
  dan mengandung banyak pasangan yang kurang relevan.
- Nilai threshold yang terlalu tinggi → coverage menurun tajam, hanya sedikit tiket yang
  memiliki pasangan mirip.

Untuk evaluasi per modul, jalankan halaman ini dengan memilih **satu modul saja**
pada filter di atas, lalu bandingkan hasil threshold antar modul.
"""
        )
else:
    st.info(
        "Atur parameter di sidebar lalu klik **Jalankan Evaluasi Threshold** "
        "untuk melihat pengaruh berbagai nilai Cosine Similarity Threshold."
    )
