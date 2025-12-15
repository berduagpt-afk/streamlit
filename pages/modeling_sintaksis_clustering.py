# pages/modeling_sintaksis_cluster.py
# Modeling Sintaksis – Clustering TF-IDF (tanpa anchor, per modul & waktu)

import re
from datetime import datetime

import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans  # bisa diganti MiniBatchKMeans kalau dataset sangat besar

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
st.title("🧩 Modeling Sintaksis – Clustering Tiket Insiden")
st.caption(
    "Clustering tiket insiden berbasis kemiripan sintaksis (TF-IDF) pada kolom "
    "**text_sintaksis** dari `lasis_djp.incident_clean`, tanpa anchor text."
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

# pastikan tipe string
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
# 🔍 Filter modul & rentang tanggal
# ======================================================
st.subheader("Filter Data (Modul & Waktu)")

col_f1, col_f2 = st.columns(2)

# Filter modul (jika ada)
with col_f1:
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
    else:
        st.info("Kolom **modul** tidak ditemukan, filter modul dilewati.")

# Filter tanggal (jika ada tgl_submit)
with col_f2:
    if "tgl_submit" in df.columns:
        df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
        valid_dates = df["tgl_submit"].dropna()
        if not valid_dates.empty:
            min_date = valid_dates.min().date()
            max_date = valid_dates.max().date()
            start_date, end_date = st.date_input(
                "Rentang tanggal (tgl_submit)",
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
# ⚙️ Parameter TF-IDF & Clustering (Sidebar)
# ======================================================
with st.sidebar:
    st.header("⚙️ Parameter Modeling Sintaksis (Clustering)")

    max_rows = st.number_input(
        "Maksimum tiket yang di-cluster (sampling acak)",
        min_value=100,
        max_value=min(50_000, len(df)),
        value=min(5_000, len(df)),
        step=100,
    )

    n_clusters = st.slider(
        "Jumlah cluster (K)",
        min_value=2,
        max_value=20,
        value=5,
        step=1,
    )

    max_features = st.number_input(
        "Maksimal fitur TF-IDF",
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

    top_terms = st.slider(
        "Jumlah kata kunci pembentuk topik cluster",
        min_value=3,
        max_value=15,
        value=5,
        step=1,
    )

# ======================================================
# 🎲 Sampling (kalau data besar)
# ======================================================
if len(df) > max_rows:
    df_sample = df.sample(n=int(max_rows), random_state=42).copy()
    st.info(
        f"Dataset berisi {len(df):,} tiket. "
        f"Diambil sampel acak sebanyak {len(df_sample):,} tiket untuk proses clustering."
    )
else:
    df_sample = df.copy()

# ======================================================
# 🔢 Hitung TF-IDF
# ======================================================
with st.spinner("🔢 Menghitung TF-IDF..."):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
        max_df=max_df_ratio,
        use_idf=True,
        norm="l2",
        lowercase=False,  # sudah dilowercase di cleaning
        token_pattern=r"(?u)\b[a-z0-9]{2,}\b",
    )

    X = vectorizer.fit_transform(df_sample["text_sintaksis_clean"].tolist())
    n_docs, n_terms = X.shape

st.success(f"TF-IDF selesai: {n_docs} dokumen × {n_terms} fitur aktif.")

# ======================================================
# 🧮 Clustering KMeans
# ======================================================
if st.button("🚀 Jalankan Clustering Sintaksis", use_container_width=True):
    with st.spinner("Mengelompokkan tiket insiden..."):
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init="auto",
        )
        labels = kmeans.fit_predict(X)

        df_sample = df_sample.reset_index(drop=True)
        df_sample["cluster_label"] = labels

        # ==================================================
        # 🔑 Tentukan "cluster topic" dari top terms TF-IDF
        # ==================================================
        feature_names = np.array(vectorizer.get_feature_names_out())
        centers = kmeans.cluster_centers_  # shape: (k, n_terms)

        topics = []
        sizes = []
        for k in range(n_clusters):
            # ukuran cluster
            size_k = (df_sample["cluster_label"] == k).sum()
            sizes.append(size_k)

            # kata dengan bobot tertinggi di centroid cluster
            center_k = centers[k]
            if n_terms > 0:
                top_idx = center_k.argsort()[::-1][:top_terms]
                top_words = feature_names[top_idx]
                topic_str = ", ".join(top_words)
            else:
                topic_str = "-"
            topics.append(topic_str)

        summary_df = pd.DataFrame(
            {
                "cluster_label": list(range(n_clusters)),
                "cluster_size": sizes,
                "cluster_topic": topics,
            }
        ).sort_values("cluster_size", ascending=False)

        st.subheader("Ringkasan Cluster (Sintaksis)")
        st.dataframe(summary_df, use_container_width=True)

        # ==================================================
        # 📈 Bubble chart: Cluster vs Waktu
        # ==================================================
        if "tgl_submit" in df_sample.columns and df_sample["tgl_submit"].notna().any():
            # agregasi per bulan
            tmp = df_sample.dropna(subset=["tgl_submit"]).copy()
            tmp["month"] = tmp["tgl_submit"].dt.to_period("M").dt.to_timestamp()

            # gabungkan label topik
            topic_map = dict(zip(summary_df["cluster_label"], summary_df["cluster_topic"]))
            tmp["cluster_topic"] = tmp["cluster_label"].map(topic_map)

            agg = (
                tmp.groupby(["month", "cluster_label", "cluster_topic"])
                .size()
                .reset_index(name="ticket_count")
            )

            st.subheader("Distribusi Tiket per Cluster dan Waktu (Bubble Chart)")
            bubble_chart = (
                alt.Chart(agg)
                .mark_circle()
                .encode(
                    x=alt.X("month:T", title="Waktu (bulan)"),
                    y=alt.Y("cluster_topic:N", title="Cluster Topic"),
                    size=alt.Size("ticket_count:Q", title="Jumlah Tiket", legend=None),
                    color=alt.Color("cluster_label:N", title="Cluster Label"),
                    tooltip=[
                        alt.Tooltip("month:T", title="Bulan"),
                        alt.Tooltip("cluster_label:N", title="Cluster"),
                        alt.Tooltip("cluster_topic:N", title="Topik"),
                        alt.Tooltip("ticket_count:Q", title="Jumlah Tiket"),
                    ],
                )
                .properties(height=350)
            )
            st.altair_chart(bubble_chart, use_container_width=True)
        else:
            st.info("Kolom tgl_submit tidak tersedia / seluruhnya NaT, bubble chart waktu tidak dapat ditampilkan.")

        # ==================================================
        # 📋 Tabel Detail Tiket per Cluster
        # ==================================================
        st.subheader("Detail Tiket per Cluster")

        # pilih cluster tertentu (seperti bagian bawah contoh gambar)
        selected_cluster = st.multiselect(
            "Filter cluster label (kosongkan = semua cluster)",
            options=summary_df["cluster_label"].tolist(),
        )

        df_detail = df_sample.copy()
        if selected_cluster:
            df_detail = df_detail[df_detail["cluster_label"].isin(selected_cluster)]

        # tambahkan cluster_topic ke detail
        topic_map = dict(zip(summary_df["cluster_label"], summary_df["cluster_topic"]))
        df_detail["cluster_topic"] = df_detail["cluster_label"].map(topic_map)

        show_cols = ["incident_number", "cluster_label", "cluster_topic"]
        for c in ["modul", "site", "tgl_submit", "judul_masalah", "text_sintaksis"]:
            if c in df_detail.columns:
                show_cols.append(c)

        st.dataframe(df_detail[show_cols], use_container_width=True)

        st.markdown(
            "Keterangan:\n"
            "- **Cluster Label**: nomor indeks cluster dari algoritma KMeans.\n"
            "- **Cluster Topic**: ringkasan topik berdasarkan kata-kata dengan bobot TF-IDF tertinggi di tiap cluster.\n"
            "- **Bubble Chart**: menggambarkan distribusi jumlah tiket per cluster untuk setiap bulan."
        )
else:
    st.info("Atur parameter di sidebar lalu klik **Jalankan Clustering Sintaksis** untuk melihat hasil.")
