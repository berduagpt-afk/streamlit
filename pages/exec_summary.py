# pages/exec_summary.py
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.title("Executive Summary of Clustering")
st.caption("Ringkasan tiket insiden: filter → clustering → KPI → tabel eksekutif.")

# ------------ Ambil data dari session ------------
df = st.session_state.get("df_clean")
if df is None:
    df = st.session_state.get("df_raw")
if df is None:
    df = st.session_state.get("dataset")

if df is None or not isinstance(df, pd.DataFrame) or df.empty:
    st.warning("Dataset belum tersedia. Upload & jalankan preprocessing terlebih dahulu.")
    st.stop()

# ------------ Deteksi kolom penting ------------
text_col = "tokens_str" if "tokens_str" in df.columns else ("Deskripsi_Bersih" if "Deskripsi_Bersih" in df.columns else "isi")
if text_col not in df.columns:
    st.error("Tidak menemukan kolom teks. Harus ada salah satu dari: tokens_str / Deskripsi_Bersih / isi.")
    st.stop()

# tanggal
date_col = None
if "timestamp" in df.columns:
    date_col = "timestamp"
elif "id_bugtrack" in df.columns:
    dt = pd.to_datetime(df["id_bugtrack"].astype(str).str.slice(0, 14), format="%Y%m%d%H%M%S", errors="coerce")
    if dt.notna().any():
        df = df.copy()
        df["timestamp"] = dt
        date_col = "timestamp"

# kandidat kolom kategori/aplikasi/service line/team/owner
def first_present(*cols):
    for c in cols:
        if c and c in df.columns:
            return c
    return None

application_col = first_present("kategori", "aplikasi", "application", "app", "seksi_kategori")
service_line_col = first_present("seksi_kategori", "service_line", "layanan")
solutions_team_col = first_present("solutions_team", "unit_penanganan", "tim_solusi", "unit", "nama_kpp_pengirim")
owner_col = first_present("application_owner", "pemilik_aplikasi", "owner", "penanggung_jawab", "nama_kpp_pengirim")

if application_col is None:
    st.error("Tidak ada kolom untuk 'Application' (coba tambahkan 'kategori' atau 'aplikasi').")
    st.stop()

# ------------ Sidebar (filter & parameter) ------------
with st.sidebar:
    st.header("⚙️ Filter & Parameter")

    # pilih kolom yang dipakai
    application_col = st.selectbox("Kolom Application", [application_col] + [c for c in df.columns if c not in [application_col]], index=0)
    service_line_col = st.selectbox("Kolom Service Line", ["(None)"] + list(df.columns), index=(1 if service_line_col else 0))
    solutions_team_col = st.selectbox("Kolom Solutions Team", ["(None)"] + list(df.columns), index=(1 if solutions_team_col else 0))
    owner_col = st.selectbox("Kolom Application Owner", ["(None)"] + list(df.columns), index=(1 if owner_col else 0))

    # nilai filter
    def pick_values(colname, label):
        if not colname or colname == "(None)":
            return None, None
        vals = ["All"] + sorted([str(x) for x in df[colname].dropna().astype(str).unique()])
        return colname, st.selectbox(label, vals, index=0)

    sl_col, sl_val = pick_values(service_line_col, "Service Line")
    team_col, team_val = pick_values(solutions_team_col, "Solutions Team")

    # parameter clustering
    st.markdown("---")
    st.markdown("**Clustering**")
    k_clusters = st.slider("Jumlah cluster (KMeans)", 2, 50, 12, 1)
    ngram = st.selectbox("N-gram TF-IDF", ["1", "1–2"], index=1)
    ngram_range = (1, 1) if ngram == "1" else (1, 2)
    min_df = st.number_input("min_df (≥ dokumen)", min_value=1, value=2, step=1)
    max_df = st.slider("max_df (≤ proporsi dokumen)", 0.5, 1.0, 0.95, 0.01)
    top_terms = st.number_input("Top terms/cluster", 3, 15, 3, 1)
    top_rows = st.number_input("Top baris per tabel", 3, 50, 10, 1)

    # hitung technical incidents (opsional)
    st.markdown("---")
    st.markdown("**Technical Incidents (opsional)**")
    tech_col = st.selectbox("Kolom tipe insiden", ["(None)"] + list(df.columns), index=0)
    tech_vals = []
    if tech_col != "(None)":
        tech_vals = st.multiselect("Nilai dianggap 'technical'", sorted([str(x) for x in df[tech_col].dropna().astype(str).unique()]))

    run = st.button("🚀 Jalankan", use_container_width=True)

# ------------ Terapkan filter ------------
df_view = df.copy()
if sl_col and sl_val and sl_val != "All":
    df_view = df_view[df_view[sl_col].astype(str) == sl_val]
if team_col and team_val and team_val != "All":
    df_view = df_view[df_view[team_col].astype(str) == team_val]

st.write(f"**Dataset aktif:** {len(df_view):,} tiket | Application: `{application_col}` | Text: `{text_col}`")

if not run:
    st.info("Atur parameter di sidebar lalu klik **Jalankan**.")
    st.stop()

if len(df_view) < k_clusters:
    st.warning(f"Baris data ({len(df_view)}) < jumlah cluster (k={k_clusters}). Kurangi K atau perbesar data.")
    st.stop()

# ------------ TF-IDF + KMeans (ringkas & cepat) ------------
texts = df_view[text_col].fillna("").astype(str).tolist()
vec = TfidfVectorizer(
    ngram_range=ngram_range,
    min_df=min_df,
    max_df=(None if max_df >= 0.9999 else max_df),
    sublinear_tf=True,
    use_idf=True,
    norm="l2",
    lowercase=False,
    token_pattern=r"(?u)\b\w+\b",
)
X = vec.fit_transform(texts)

km = KMeans(n_clusters=k_clusters, n_init="auto", random_state=42)
labels = km.fit_predict(X)

df_view = df_view.reset_index(drop=True)
df_view["cluster_label"] = labels

# top terms untuk nama topik cluster
feature_names = vec.get_feature_names_out()
def topic_of_cluster(cid: int, n: int) -> str:
    center = km.cluster_centers_[cid]
    idx = np.argsort(center)[-n:][::-1]
    return ", ".join(feature_names[idx])

cluster_topic = {c: topic_of_cluster(c, top_terms) for c in range(k_clusters)}
df_view["cluster_topic"] = df_view["cluster_label"].map(cluster_topic)

# ------------ KPI cards ------------
total_incident = len(df_view)
if tech_col != "(None)" and tech_vals:
    tech_incident = int(df_view[df_view[tech_col].astype(str).isin(tech_vals)].shape[0])
else:
    tech_incident = total_incident  # fallback
clusters_identified = int(df_view["cluster_label"].nunique())

c1, c2, c3 = st.columns(3)
c1.metric("Incident Tickets", f"{total_incident:,}")
c2.metric("Technical Incidents", f"{tech_incident:,}")
c3.metric("Clusters Identified", f"{clusters_identified:,}")

# ------------ Tabel: Clusters with the most Incidents ------------
app = application_col
by_app_cluster = (
    df_view.groupby([app, "cluster_label"])
    .size()
    .rename("Cluster Size")
    .reset_index()
    .sort_values("Cluster Size", ascending=False)
)
by_app_cluster["Cluster Topic"] = by_app_cluster["cluster_label"].map(cluster_topic)
table_left = by_app_cluster.rename(columns={app: "Application"})[
    ["Application", "Cluster Size", "Cluster Topic"]
].head(top_rows)

# ------------ Tabel: Applications with the most Clusters ------------
# hitung berapa banyak cluster unik yang muncul per application
app_cluster_count = (
    df_view.groupby(app)["cluster_label"].nunique().rename("Number of Clusters").reset_index()
)
# owner (opsional)
if owner_col and owner_col != "(None)" and owner_col in df_view.columns:
    owner_map = (
        df_view.groupby(app)[owner_col].agg(lambda x: x.dropna().astype(str).mode().iloc[0] if not x.dropna().empty else "")
    )
    app_cluster_count["Application Owner"] = app_cluster_count[app].map(owner_map)
else:
    app_cluster_count["Application Owner"] = ""

table_right = app_cluster_count.rename(columns={app: "Application"})[
    ["Application", "Number of Clusters", "Application Owner"]
].sort_values("Number of Clusters", ascending=False).head(top_rows)

# ------------ Layout tabel ------------
st.markdown("### ")
lcol, rcol = st.columns(2)

with lcol:
    st.markdown("**Clusters with the most Incidents**")
    st.dataframe(table_left, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download (left table)",
        data=table_left.to_csv(index=False).encode("utf-8"),
        file_name="clusters_with_most_incidents.csv",
        mime="text/csv",
    )

with rcol:
    st.markdown("**Applications with the most Clusters**")
    st.dataframe(table_right, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download (right table)",
        data=table_right.to_csv(index=False).encode("utf-8"),
        file_name="applications_with_most_clusters.csv",
        mime="text/csv",
    )

# ------------ Simpan ke session (opsional) ------------
st.session_state["exec_summary_tables"] = {
    "clusters_with_most_incidents": table_left,
    "applications_with_most_clusters": table_right,
}
