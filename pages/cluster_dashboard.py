# pages/cluster_dashboard.py
# Analisis Clustering TF-IDF + KMeans dari PostgreSQL (schema: lasis_djp)
# Versi dengan 2 mode pengolahan data + datepicker horizontal yang selalu muncul

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from datetime import datetime, date
from sqlalchemy import create_engine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# ======================================================
# 🔐 KONEKSI DATABASE
# ======================================================
def get_connection():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url)

@st.cache_data(show_spinner=False)
def load_from_db(table_name="incident_clean", schema="lasis_djp", limit=None):
    engine = get_connection()
    query = f'SELECT * FROM "{schema}"."{table_name}"'
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, con=engine)
    engine.dispose()
    return df

def save_to_db(df, table_name="incident_cluster", schema="lasis_djp"):
    engine = get_connection()
    df.to_sql(table_name, engine, schema=schema, if_exists="replace", index=False)
    engine.dispose()

# ======================================================
# 🧭 PAGE SETUP
# ======================================================
st.title("📊 Incident Ticket Clusters (TF-IDF + KMeans)")
st.caption("Analisis topik otomatis dari tabel `lasis_djp.incident_clean` menggunakan TF-IDF dan K-Means clustering.")

# ======================================================
# ⚙️ SIDEBAR PARAMETER
# ======================================================
with st.sidebar:
    st.header("⚙️ Pengaturan Clustering")

    process_mode = st.radio(
        "Mode Pengolahan Data",
        ["Berdasarkan Jumlah Tiket", "Berdasarkan Range Waktu (Tanggal Submit)"],
        index=0,
    )

    sample_limit = st.number_input("Ambil maksimal data", 0, 50000, 10000, step=5000, help="0 = ambil semua baris")
    k_clusters = st.slider("Jumlah cluster (KMeans)", 2, 15, 5, 1)
    ngram_choice = st.selectbox("N-gram TF-IDF", ["1", "1–2"], index=1)
    ngram = (1, 1) if ngram_choice == "1" else (1, 2)
    min_df = st.number_input("min_df (dokumen minimal)", 1, 100, 2, 1)
    max_df = st.slider("max_df (proporsi dokumen maks)", 0.5, 1.0, 0.95, 0.01)
    top_terms = st.number_input("Top terms/cluster", 3, 15, 5, 1)
    save_db = st.checkbox("Simpan hasil ke PostgreSQL", True)
    st.markdown("---")

# ======================================================
# 📦 LOAD DATA
# ======================================================
with st.spinner("📦 Mengambil data dari database..."):
    try:
        df = load_from_db("incident_clean", "lasis_djp", limit=sample_limit or None)
        st.success(f"Berhasil memuat {len(df):,} baris.")
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        st.stop()

if df.empty:
    st.warning("Dataset kosong. Pastikan tabel `lasis_djp.incident_clean` memiliki data.")
    st.stop()

# ======================================================
# 🧮 VALIDASI KOLOM
# ======================================================
text_col = "tokens_str" if "tokens_str" in df.columns else "Deskripsi_Bersih" if "Deskripsi_Bersih" in df.columns else None
if not text_col:
    st.error("Kolom teks tidak ditemukan (harus ada `tokens_str` atau `Deskripsi_Bersih`).")
    st.stop()

id_col = "Incident_Number" if "Incident_Number" in df.columns else None
date_col = "tgl_submit" if "tgl_submit" in df.columns else None

# ======================================================
# 📅 FILTER RANGE WAKTU (SELALU TAMPIL SAAT MODE RANGE)
# ======================================================
if process_mode == "Berdasarkan Range Waktu (Tanggal Submit)":
    st.sidebar.markdown("### 📅 Rentang Waktu Analisis")

    # Styling biru DJP
    st.markdown(
        """
        <style>
        div[data-testid="stDateInput"] label {
            color: white !important;
            background-color: #0B3A82;
            padding: 3px 8px;
            border-radius: 6px;
            font-weight: 600;
            margin-bottom: 4px;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input("Start date", value=date(2023, 1, 1))
    with col2:
        end_date = st.date_input("End date", value=date.today())

    if start_date > end_date:
        st.error("❌ Tanggal mulai tidak boleh setelah tanggal selesai.")
        st.stop()

    # Filter dataset bila kolom tanggal ada
    if date_col and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df[(df[date_col].dt.date >= start_date) & (df[date_col].dt.date <= end_date)]
        st.info(f"📆 Analisis data dari **{start_date}** hingga **{end_date}** ({len(df):,} tiket).")
    else:
        st.warning("Kolom `tgl_submit` tidak ditemukan, filter waktu dilewati.")

if df.empty:
    st.warning("Tidak ada data yang sesuai rentang waktu.")
    st.stop()

# ======================================================
# 🧼 PRA-PROSES TEKS
# ======================================================
df[text_col] = df[text_col].fillna("").astype(str)
df["text_len"] = df[text_col].str.len()
df_valid = df[df["text_len"] >= 2].copy()

if df_valid.empty:
    st.error("❌ Semua dokumen kosong atau terlalu pendek untuk diproses.")
    st.stop()

texts = df_valid[text_col].tolist()
st.info(f"Menjalankan TF-IDF pada {len(texts):,} dokumen valid ...")

# ======================================================
# 🔠 TF-IDF + K-MEANS
# ======================================================
try:
    vec = TfidfVectorizer(
        ngram_range=ngram,
        min_df=min_df,
        max_df=(None if max_df >= 0.9999 else max_df),
        sublinear_tf=True,
        use_idf=True,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
        lowercase=False,
    )
    X = vec.fit_transform(texts)
except ValueError as e:
    st.error(f"Gagal membentuk TF-IDF: {e}")
    st.stop()

if X.shape[1] == 0:
    st.error("❌ Tidak ada fitur yang terbentuk. Mungkin semua teks hanya berisi stopword.")
    st.stop()

km = KMeans(n_clusters=k_clusters, n_init="auto", random_state=42)
labels = km.fit_predict(X)
df_valid["cluster_label"] = labels

# ======================================================
# 📋 RINGKASAN CLUSTER
# ======================================================
feature_names = vec.get_feature_names_out()
def top_terms_of_cluster(cidx, n=top_terms):
    center = km.cluster_centers_[cidx]
    idx = np.argsort(center)[-n:][::-1]
    return ", ".join(feature_names[idx])

cluster_topics = {c: top_terms_of_cluster(c) for c in range(k_clusters)}
df_valid["cluster_topic"] = df_valid["cluster_label"].map(cluster_topics)

st.subheader("📋 Ringkasan Cluster")
summary = (
    df_valid.groupby("cluster_label")
    .size()
    .rename("Jumlah Tiket")
    .reset_index()
    .sort_values("Jumlah Tiket", ascending=False)
)
summary["Cluster Topic"] = summary["cluster_label"].map(cluster_topics)
st.dataframe(summary, use_container_width=True, hide_index=True)

# ======================================================
# 📊 DISTRIBUSI CLUSTER (BAR CHART)
# ======================================================
st.subheader("📊 Distribusi Cluster Berdasarkan Jumlah Tiket")
dist = (
    df_valid.groupby(["cluster_label", "cluster_topic"])
    .size()
    .reset_index(name="Jumlah Tiket")
)
bar_chart = (
    alt.Chart(dist)
    .mark_bar()
    .encode(
        x=alt.X("Jumlah Tiket:Q", title="Jumlah Tiket"),
        y=alt.Y("cluster_topic:N", sort="-x", title="Cluster Topic"),
        color="cluster_label:N",
        tooltip=["cluster_label:N", "cluster_topic:N", "Jumlah Tiket:Q"],
    )
)
st.altair_chart(bar_chart, use_container_width=True)

# ======================================================
# 🕒 TIMELINE CLUSTER PER BULAN
# ======================================================
if date_col and pd.to_datetime(df_valid[date_col], errors="coerce").notna().any():
    st.subheader("📈 Timeline Cluster per Bulan (Berdasarkan Tanggal Submit)")
    df_valid["month"] = pd.to_datetime(df_valid[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    g = df_valid.groupby(["month", "cluster_label"]).size().reset_index(name="Jumlah Tiket")
    g["Cluster Topic"] = g["cluster_label"].map(cluster_topics)

    chart = (
        alt.Chart(g)
        .mark_circle()
        .encode(
            x=alt.X("month:T", title="Bulan"),
            y=alt.Y("Cluster Topic:N", title="Cluster Topic"),
            size="Jumlah Tiket:Q",
            color="cluster_label:N",
            tooltip=["month:T", "Cluster Topic:N", "Jumlah Tiket:Q"],
        )
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Kolom tanggal tidak tersedia atau kosong, timeline tidak ditampilkan.")

# ======================================================
# 💾 SIMPAN & DOWNLOAD
# ======================================================
df_valid["tgl_clustered"] = datetime.now()
if "tgl_submit" in df_valid.columns:
    df_valid["tgl_submit"] = pd.to_datetime(df_valid["tgl_submit"], errors="coerce")

if save_db:
    try:
        with st.spinner("💾 Menyimpan hasil clustering ke database..."):
            save_to_db(df_valid, "incident_cluster", "lasis_djp")
        st.success("✅ Hasil clustering berhasil disimpan ke database.")
    except Exception as e:
        st.error(f"Gagal menyimpan ke database: {e}")

csv = df_valid.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Download Hasil Clustering (CSV)",
    data=csv,
    file_name="incident_clusters.csv",
    mime="text/csv",
)
