# pages/cluster_dashboard.py
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# st.set_page_config(page_title="Incident Ticket Clusters", page_icon=":bar_chart:", layout="wide")
st.title("Incident Ticket Clusters by Application")
st.caption("Prototype ringkas: filter aplikasi → clustering TF-IDF → topik cluster → bubble timeline → tabel tiket.")

# ---------- Ambil data dari session ----------
df = st.session_state.get("df_clean")  # hasil preprocessing
if df is None:
    df = st.session_state.get("df_raw") or st.session_state.get("dataset")

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("Dataset belum tersedia. Upload & jalankan preprocessing terlebih dahulu.")
    st.stop()

# ---------- Pastikan kolom-kolom yang dibutuhkan ----------
text_col = "tokens_str" if "tokens_str" in df.columns else ("Deskripsi_Bersih" if "Deskripsi_Bersih" in df.columns else "isi")
if text_col not in df.columns:
    st.error("Tidak menemukan kolom teks. Harus ada salah satu dari: tokens_str / Deskripsi_Bersih / isi.")
    st.stop()

# kolom tanggal: timestamp dari preprocessing paling diprioritaskan
if "timestamp" in df.columns:
    date_col = "timestamp"
else:
    # coba ekstrak dari id_bugtrack (14 digit pertama) bila ada
    date_col = None
    if "id_bugtrack" in df.columns:
        dt = pd.to_datetime(df["id_bugtrack"].astype(str).str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
        if dt.notna().any():
            df = df.copy()
            df["timestamp"] = dt
            date_col = "timestamp"

# kolom ID tiket & pembuat
id_col = "no_tiket" if "no_tiket" in df.columns else ("id_bugtrack" if "id_bugtrack" in df.columns else None)
creator_col = "nama_kpp_pengirim" if "nama_kpp_pengirim" in df.columns else None
svc_col = "seksi_kategori" if "seksi_kategori" in df.columns else None
prio_col = "priority" if "priority" in df.columns else None  # mungkin belum ada

# "Application" = kategori (fallback ke seksi_kategori)
app_base = "kategori" if "kategori" in df.columns else (svc_col if svc_col in df.columns else None)
if app_base is None:
    st.error("Tidak ada kolom 'kategori' atau 'seksi_kategori' untuk dijadikan filter aplikasi.")
    st.stop()

# ---------- Sidebar parameter ----------
with st.sidebar:
    st.header("⚙️ Pengaturan")
    apps = ["(Semua)"] + sorted([x for x in df[app_base].astype(str).unique() if x and x != "nan"])
    app_selected = st.selectbox("Application Name", apps, index=0)
    k_clusters = st.slider("Jumlah cluster (KMeans)", 2, 12, 4, 1)
    ngram_choice = st.selectbox("N-gram TF-IDF", ["1", "1–2"], index=1)
    ngram = (1, 1) if ngram_choice == "1" else (1, 2)
    min_df = st.number_input("min_df (≥ dokumen)", min_value=1, value=2, step=1)
    max_df = st.slider("max_df (≤ proporsi dokumen)", 0.5, 1.0, 0.95, 0.01)
    top_terms = st.number_input("Top terms/cluster", 3, 15, 3, 1)
    run = st.button("🚀 Jalankan")

# ---------- Filter aplikasi ----------
if app_selected != "(Semua)":
    df_view = df[df[app_base].astype(str) == app_selected].copy()
else:
    df_view = df.copy()

st.write(f"**Dataset aktif:** {len(df_view):,} tiket | Teks: `{text_col}` | Aplikasi: `{app_selected}`")

if not run:
    st.info("Atur parameter di sidebar lalu klik **Jalankan**.")
    st.stop()

if len(df_view) < k_clusters:
    st.warning(f"Baris data ({len(df_view)}) lebih kecil dari jumlah cluster (k={k_clusters}). Kurangi K atau perbesar data.")
    st.stop()

# ---------- TF-IDF + KMeans clustering ----------
texts = df_view[text_col].fillna("").astype(str).tolist()
vec = TfidfVectorizer(
    ngram_range=ngram,
    min_df=min_df,
    max_df=(None if max_df >= 0.9999 else max_df),
    sublinear_tf=True,
    use_idf=True,
    norm="l2",
    token_pattern=r"(?u)\b\w+\b",
    lowercase=False,  # sudah dibersihkan di preprocessing
)
X = vec.fit_transform(texts)
km = KMeans(n_clusters=k_clusters, n_init="auto", random_state=42)
labels = km.fit_predict(X)

df_view = df_view.reset_index(drop=True)
df_view["cluster_label"] = labels

# Top terms per cluster dari centroid
feature_names = vec.get_feature_names_out()
def top_terms_of_cluster(cidx, n=top_terms):
    center = km.cluster_centers_[cidx]
    idx = np.argsort(center)[-n:][::-1]
    return ", ".join(feature_names[idx])

cluster_topics = {c: top_terms_of_cluster(c, n=top_terms) for c in range(k_clusters)}
df_view["cluster_topic"] = df_view["cluster_label"].map(cluster_topics)

# ---------- Summary table (kiri) ----------
summary = (
    df_view.groupby("cluster_label")
    .size()
    .rename("cluster_size")
    .reset_index()
    .sort_values("cluster_label")
)
summary["cluster_topic"] = summary["cluster_label"].map(cluster_topics)

left, right = st.columns([0.38, 0.62])
with left:
    st.subheader("Ringkasan Cluster")
    st.dataframe(summary.rename(columns={
        "cluster_label": "Cluster Label",
        "cluster_size": "Cluster Size",
        "cluster_topic": "Cluster Topic",
    }), use_container_width=True, hide_index=True)

# ---------- Bubble timeline (kanan) ----------
with right:
    st.subheader("Timeline Cluster (Bulanan)")
    if date_col is None or df_view[date_col].isna().all():
        st.info("Kolom tanggal tidak tersedia, timeline disembunyikan.")
    else:
        tmp = df_view[[date_col, "cluster_label"]].copy()
        tmp["month"] = tmp[date_col].dt.to_period("M").dt.to_timestamp()
        g = tmp.groupby(["month", "cluster_label"]).size().reset_index(name="count")
        g["cluster_topic"] = g["cluster_label"].map(cluster_topics)

        chart = alt.Chart(g).mark_circle().encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("cluster_topic:N", title="Cluster Topic"),
            size=alt.Size("count:Q", legend=None),
            tooltip=["month:T", "cluster_label:N", "cluster_topic:N", "count:Q"],
            color=alt.Color("cluster_label:N", legend=None),
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)

# ---------- Detail tiket ----------
st.subheader("Daftar Tiket")
cols = {
    "Ticket ID": df_view[id_col] if id_col else pd.Series(["–"] * len(df_view)),
    "Ticket Creator": df_view[creator_col] if creator_col else pd.Series(["–"] * len(df_view)),
    "Resolution Notes": df_view["isi"] if "isi" in df_view.columns else df_view[text_col],
    "Cluster Label": df_view["cluster_label"],
    "Cluster Topic": df_view["cluster_topic"],
    "Service Line": df_view[svc_col] if svc_col else pd.Series(["–"] * len(df_view)),
    "Priority": df_view[prio_col] if prio_col else pd.Series(["–"] * len(df_view)),
}
table = pd.DataFrame(cols)
st.dataframe(table, use_container_width=True, hide_index=True)

# Unduh hasil
out = df_view.copy()
st.download_button(
    "📥 Download Hasil Clustering (CSV)",
    data=out.to_csv(index=False).encode("utf-8"),
    file_name="incident_clusters.csv",
    mime="text/csv",
)
