# pages/reporting_summary.py
# Dashboard ringkas visual dari hasil clustering (lasis_djp.incident_cluster)

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine
from datetime import datetime

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
def load_from_db(table_name="incident_cluster", schema="lasis_djp"):
    """Ambil data hasil clustering dari PostgreSQL."""
    engine = get_connection()
    query = f'SELECT * FROM "{schema}"."{table_name}"'
    df = pd.read_sql_query(query, con=engine)
    engine.dispose()
    return df

# ======================================================
# 🧭 PAGE SETUP (tanpa set_page_config karena sudah di app.py)
# ======================================================
st.title("📊 Executive Reporting Summary")
st.caption("Visualisasi interaktif dari hasil clustering insiden — sumber data: `lasis_djp.incident_cluster`.")

# ======================================================
# 📦 LOAD DATA
# ======================================================
with st.spinner("📦 Mengambil data dari database..."):
    try:
        df = load_from_db("incident_cluster", "lasis_djp")
        st.success(f"Berhasil memuat {len(df):,} baris dari lasis_djp.incident_cluster.")
    except Exception as e:
        st.error(f"Gagal mengambil data: {e}")
        st.stop()

if df.empty:
    st.warning("Dataset kosong. Jalankan proses clustering terlebih dahulu.")
    st.stop()

# ======================================================
# 🧮 VALIDASI KOLOM
# ======================================================
# Gunakan tanggal submit sebagai prioritas, fallback ke tanggal clustered
date_col = None
if "tgl_submit" in df.columns:
    date_col = "tgl_submit"
elif "tgl_clustered" in df.columns:
    date_col = "tgl_clustered"

text_col = "Deskripsi_Bersih" if "Deskripsi_Bersih" in df.columns else "tokens_str"
cluster_col = "cluster_label" if "cluster_label" in df.columns else None
topic_col = "cluster_topic" if "cluster_topic" in df.columns else None
app_col = "modul" if "modul" in df.columns else ("kategori" if "kategori" in df.columns else None)

if cluster_col is None:
    st.error("Kolom `cluster_label` tidak ditemukan. Pastikan hasil clustering sudah tersimpan dengan benar.")
    st.stop()

# ======================================================
# ⚙️ SIDEBAR FILTER
# ======================================================
with st.sidebar:
    st.header("⚙️ Filter Data")

    app_selected = None
    if app_col:
        apps = ["(Semua)"] + sorted(df[app_col].dropna().astype(str).unique().tolist())
        app_selected = st.selectbox("Filter Application", apps, index=0)

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].notna().any():
            min_date, max_date = df[date_col].min(), df[date_col].max()
            date_range = st.date_input(
                "Rentang tanggal",
                (min_date.date(), max_date.date()) if min_date and max_date else None
            )
            if len(date_range) == 2:
                df = df[
                    (df[date_col].dt.date >= date_range[0])
                    & (df[date_col].dt.date <= date_range[1])
                ]

    st.markdown("---")
    show_detail = st.checkbox("Tampilkan tabel detail", False)

# ======================================================
# 🔍 FILTER DATA
# ======================================================
if app_col and app_selected and app_selected != "(Semua)":
    df = df[df[app_col].astype(str) == app_selected]
    st.info(f"Menampilkan data untuk **{app_selected}** ({len(df):,} tiket).")

if df.empty:
    st.warning("Tidak ada data sesuai filter yang dipilih.")
    st.stop()

# ======================================================
# 🧾 INFO SUMBER TANGGAL
# ======================================================
if date_col:
    st.caption(f"📅 Menggunakan tanggal: **{date_col.replace('_', ' ').title()}** untuk visualisasi timeline.")
else:
    st.caption("📅 Tidak ditemukan kolom tanggal (`tgl_submit` / `tgl_clustered`). Timeline tidak ditampilkan.")

# ======================================================
# 🧾 KPI CARDS
# ======================================================
total_tickets = len(df)
total_clusters = df[cluster_col].nunique()
avg_ticket_per_cluster = round(total_tickets / total_clusters, 1) if total_clusters else 0

# cluster terbesar
top_cluster = df[cluster_col].value_counts().idxmax() if not df[cluster_col].empty else None
top_cluster_topic = (
    df[df[cluster_col] == top_cluster][topic_col].iloc[0]
    if topic_col and top_cluster is not None
    else "-"
)
top_cluster_size = df[cluster_col].value_counts().max() if not df[cluster_col].empty else 0

# tampilkan KPI
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Incident Tickets", f"{total_tickets:,}")
c2.metric("Clusters Identified", f"{total_clusters:,}")
c3.metric("Avg Tickets / Cluster", f"{avg_ticket_per_cluster:,}")
c4.metric("Top Cluster", f"{top_cluster_topic}", f"{top_cluster_size:,} Tickets")

# ======================================================
# 📈 TREN INSIDEN PER BULAN
# ======================================================
if date_col and df[date_col].notna().any():
    st.markdown("### 📈 Tren Jumlah Tiket per Bulan")

    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("month").size().reset_index(name="Jumlah Tiket")

    chart = (
        alt.Chart(monthly)
        .mark_line(point=True, color="#0B3A82")
        .encode(
            x=alt.X("month:T", title="Bulan"),
            y=alt.Y("Jumlah Tiket:Q", title="Jumlah Tiket"),
            tooltip=["month:T", "Jumlah Tiket:Q"],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Kolom tanggal tidak tersedia — tren bulanan tidak dapat ditampilkan.")

# ======================================================
# 🧩 DISTRIBUSI CLUSTER
# ======================================================
st.markdown("### 🔘 Distribusi Cluster Berdasarkan Topik")

cluster_dist = (
    df.groupby([cluster_col, topic_col])[cluster_col]
    .count()
    .rename("Jumlah Tiket")
    .reset_index()
    .sort_values("Jumlah Tiket", ascending=False)
)

bar_chart = (
    alt.Chart(cluster_dist)
    .mark_bar()
    .encode(
        x=alt.X("Jumlah Tiket:Q", title="Jumlah Tiket"),
        y=alt.Y("cluster_topic:N", sort="-x", title="Cluster Topic"),
        color=alt.Color("cluster_label:N", legend=None, scale=alt.Scale(scheme="goldorange")),
        tooltip=["cluster_label:N", "cluster_topic:N", "Jumlah Tiket:Q"],
    )
    .properties(height=400)
)
st.altair_chart(bar_chart, use_container_width=True)

# ======================================================
# 🧾 DETAIL TABEL (OPSIONAL)
# ======================================================
if show_detail:
    st.markdown("### 🧾 Detail Tiket per Cluster")
    display_cols = [c for c in [app_col, text_col, cluster_col, topic_col, date_col] if c in df.columns]
    st.dataframe(df[display_cols].head(300), use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Data (CSV)",
        data=csv,
        file_name="reporting_summary.csv",
        mime="text/csv",
    )

# ======================================================
# ✅ CATATAN
# ======================================================
st.markdown("---")
st.caption(
    "💡 Dashboard ini membaca data dari tabel **`lasis_djp.incident_cluster`** hasil proses clustering otomatis. "
    "Gunakan filter di sidebar untuk menelusuri subset data. "
    "Tema warna mengikuti palet DJP (biru–emas)."
)

# ======================================================
# 🎨 GAYA TAMBAHAN (THEME DJP)
# ======================================================
st.markdown(
    """
    <style>
    .stMetricLabel {
        color: #0B3A82 !important;
        font-weight: 600 !important;
    }
    .stMetricValue {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
