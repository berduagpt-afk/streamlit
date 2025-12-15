# pages/exec_summary.py
# Executive Summary dari hasil clustering di PostgreSQL (schema: lasis_djp)

import pandas as pd
import streamlit as st
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
def load_from_db(table_name="incident_cluster", schema="lasis_djp", limit=None):
    engine = get_connection()
    query = f'SELECT * FROM "{schema}"."{table_name}"'
    if limit:
        query += f" LIMIT {limit}"
    df = pd.read_sql_query(query, con=engine)
    engine.dispose()
    return df

# ======================================================
# 🧭 PAGE SETUP
# ======================================================
st.title("📊 Executive Summary of Incident Clustering")
st.caption("Ringkasan tiket insiden dari lasis_djp.incident_cluster — KPI utama, topik cluster, dan tabel aplikasi.")

# ======================================================
# 📦 AMBIL DATA DARI DATABASE
# ======================================================
with st.spinner("📦 Mengambil data clustering dari database..."):
    try:
        df = load_from_db("incident_cluster", "lasis_djp")
        st.success(f"Berhasil memuat {len(df):,} baris dari lasis_djp.incident_cluster")
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        st.stop()

if df.empty:
    st.warning("Dataset kosong. Pastikan tabel lasis_djp.incident_cluster sudah diisi dari proses clustering.")
    st.stop()

# ======================================================
# 🧮 VALIDASI KOLOM
# ======================================================
text_col = "Deskripsi_Bersih" if "Deskripsi_Bersih" in df.columns else (
    "tokens_str" if "tokens_str" in df.columns else None
)
if not text_col:
    st.error("Kolom teks tidak ditemukan (harus ada Deskripsi_Bersih atau tokens_str).")
    st.stop()

app_col = "modul" if "modul" in df.columns else (
    "kategori" if "kategori" in df.columns else None
)
if not app_col:
    st.error("Tidak ditemukan kolom aplikasi (misal: modul/kategori).")
    st.stop()

id_col = "Incident_Number" if "Incident_Number" in df.columns else None
cluster_col = "cluster_label" if "cluster_label" in df.columns else None
topic_col = "cluster_topic" if "cluster_topic" in df.columns else None

# ======================================================
# ⚙️ SIDEBAR FILTER
# ======================================================
with st.sidebar:
    st.header("⚙️ Filter & Parameter")
    limit = st.number_input("Ambil maksimal data (0=semua)", 0, 50000, 0, step=5000)
    app_selected = st.selectbox(
        "Filter berdasarkan Application (modul/kategori)",
        ["(Semua)"] + sorted(df[app_col].dropna().astype(str).unique().tolist())
    )
    top_rows = st.number_input("Top baris per tabel", 3, 50, 10, 1)
    run = st.button("🚀 Tampilkan Ringkasan", use_container_width=True)

if not run:
    st.info("Atur parameter di sidebar lalu klik **Tampilkan Ringkasan**.")
    st.stop()

if limit and limit > 0:
    df = df.head(limit)

if app_selected != "(Semua)":
    df = df[df[app_col].astype(str) == app_selected]
    st.info(f"Menampilkan data untuk application **{app_selected}** ({len(df):,} tiket).")

if df.empty:
    st.warning("Tidak ada data yang sesuai filter.")
    st.stop()

# ======================================================
# 🧾 KPI Cards
# ======================================================
total_tickets = len(df)
n_clusters = df[cluster_col].nunique() if cluster_col in df.columns else 0
avg_tickets_per_cluster = (
    df.groupby(cluster_col).size().mean() if cluster_col in df.columns else 0
)

col1, col2, col3 = st.columns(3)
col1.metric("Total Incident Tickets", f"{total_tickets:,}")
col2.metric("Clusters Identified", f"{n_clusters:,}")
col3.metric("Avg Tickets / Cluster", f"{avg_tickets_per_cluster:.1f}")

# ======================================================
# 🧩 TABLE 1: Clusters with Most Tickets
# ======================================================
if cluster_col in df.columns:
    cluster_counts = (
        df.groupby(cluster_col).size().rename("Cluster Size").reset_index()
        .sort_values("Cluster Size", ascending=False)
    )
    if topic_col in df.columns:
        cluster_counts["Cluster Topic"] = cluster_counts[cluster_col].map(
            df.drop_duplicates(cluster_col).set_index(cluster_col)[topic_col]
        )
    table_left = cluster_counts.head(top_rows)
else:
    table_left = pd.DataFrame(columns=["Cluster", "Cluster Size", "Cluster Topic"])

# ======================================================
# 🧩 TABLE 2: Applications with Most Clusters
# ======================================================
if app_col in df.columns and cluster_col in df.columns:
    app_cluster_count = (
        df.groupby(app_col)[cluster_col].nunique().rename("Number of Clusters").reset_index()
        .sort_values("Number of Clusters", ascending=False)
    )
    table_right = app_cluster_count.head(top_rows)
else:
    table_right = pd.DataFrame(columns=["Application", "Number of Clusters"])

# ======================================================
# 📊 DISPLAY TABLES
# ======================================================
st.subheader("📈 Executive Summary Tables")
left, right = st.columns(2)

with left:
    st.markdown("**Clusters with the Most Incidents**")
    st.dataframe(table_left, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download (left table)",
        data=table_left.to_csv(index=False).encode("utf-8"),
        file_name="clusters_with_most_incidents.csv",
        mime="text/csv",
    )

with right:
    st.markdown("**Applications with the Most Clusters**")
    st.dataframe(table_right, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download (right table)",
        data=table_right.to_csv(index=False).encode("utf-8"),
        file_name="applications_with_most_clusters.csv",
        mime="text/csv",
    )

# ======================================================
# 💾 SIMPAN KE SESSION
# ======================================================
st.session_state["exec_summary_tables"] = {
    "clusters_with_most_incidents": table_left,
    "applications_with_most_clusters": table_right,
}

st.success("✅ Executive summary berhasil dibuat dari database.")
