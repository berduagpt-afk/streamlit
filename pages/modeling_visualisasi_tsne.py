# pages/visualisasi_tsne.py
# Visualisasi t-SNE untuk hasil clustering (TF-IDF -> (SVD) -> t-SNE)
# Sesuai skema:
# - lasis_djp.modeling_runs (run_id, run_time, approach, params_json, data_range, notes, threshold, window_days)
# - lasis_djp.cluster_summary (run_id, cluster_id, modul, ..., n_tickets, window_start/end, representative_text, top_terms, metrics_json)
# - lasis_djp.cluster_members (diasumsikan ada: run_id, modul, cluster_id, incident_number, tgl_submit)
# - lasis_djp.incident_clean (kolom teks: text_clean/isi_permasalahan/dll)

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE


# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# ======================================================
# ⚙️ Konstanta
# ======================================================
SCHEMA = "lasis_djp"
T_RUNS = "modeling_runs"
T_SUMMARY = "cluster_summary"
T_MEMBERS = "cluster_members"
T_CLEAN = "incident_clean"

DEFAULT_RUN_ID = "9"  # run_id Anda bertipe text


# ======================================================
# 🔌 DB Connection (secrets.toml)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


engine = get_engine()


# ======================================================
# 🧱 Helper
# ======================================================
def _guess_text_col(df: pd.DataFrame) -> str:
    candidates = [
        "text_clean", "clean_text", "isi_permasalahan", "detailed_description",
        "description", "deskripsi", "uraian"
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: ambil kolom object pertama
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        raise ValueError("Tidak menemukan kolom teks pada incident_clean.")
    return obj_cols[0]


@st.cache_data(show_spinner=False)
def load_runs() -> pd.DataFrame:
    q = f"""
    SELECT
        run_id,
        run_time,
        approach,
        threshold,
        window_days,
        notes,
        params_json,
        data_range
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC, run_id DESC
    """
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn)


@st.cache_data(show_spinner=False)
def load_run_stats(run_id: str) -> pd.DataFrame:
    # agregasi statistik per run dari cluster_summary
    q = f"""
    SELECT
        run_id,
        COUNT(*)::int AS total_clusters,
        SUM(n_tickets)::int AS total_tickets,
        AVG(n_tickets)::float AS avg_cluster_size,
        MIN(n_tickets)::int AS min_cluster_size,
        MAX(n_tickets)::int AS max_cluster_size
    FROM {SCHEMA}.{T_SUMMARY}
    WHERE run_id = :run_id
    GROUP BY run_id
    """
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn, params={"run_id": run_id})


@st.cache_data(show_spinner=False)
def load_cluster_sizes(run_id: str) -> pd.DataFrame:
    # ukuran cluster per (modul, cluster_id) dari cluster_summary (lebih “official”)
    q = f"""
    SELECT
        run_id,
        cluster_id,
        modul,
        n_tickets
    FROM {SCHEMA}.{T_SUMMARY}
    WHERE run_id = :run_id
    """
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn, params={"run_id": run_id})


@st.cache_data(show_spinner=False)
def load_members(run_id: str) -> pd.DataFrame:
    q = f"""
    SELECT
        run_id,
        modul,
        cluster_id,
        incident_number,
        tgl_submit,
        text_sintaksis
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE run_id = :run_id
    """
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn, params={"run_id": run_id})


@st.cache_data(show_spinner=False)
def load_incident_clean(incident_numbers: list[str]) -> pd.DataFrame:
    q = f"""
    SELECT *
    FROM {SCHEMA}.{T_CLEAN}
    WHERE incident_number = ANY(:ids)
    """
    with engine.connect() as conn:
        return pd.read_sql(text(q), conn, params={"ids": incident_numbers})


def compute_tsne(
    texts: list[str],
    random_state: int,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    use_svd: bool,
    svd_dim: int,
    perplexity: int,
    learning_rate: float,
    n_iter: int,
) -> np.ndarray:
    vec = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(ngram_min, ngram_max),
        min_df=2,
        max_df=0.95
    )
    X = vec.fit_transform(texts)

    if use_svd:
        svd = TruncatedSVD(n_components=svd_dim, random_state=random_state)
        Xr = svd.fit_transform(X)
    else:
        Xr = X.toarray()

    tsne = TSNE(
        n_components=2,
        random_state=random_state,
        init="pca",
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        metric="euclidean",
        verbose=0,
    )
    return tsne.fit_transform(Xr)


# ======================================================
# 🖥️ UI
# ======================================================
st.title("Visualisasi t-SNE — Klaster Tiket Insiden")
st.caption("TF-IDF → (opsional TruncatedSVD) → t-SNE 2D. Menggunakan statistik cluster dari cluster_summary (n_tickets).")

runs = load_runs()
if runs.empty:
    st.warning("Tabel modeling_runs kosong.")
    st.stop()

run_ids = runs["run_id"].astype(str).tolist()
default_idx = run_ids.index(DEFAULT_RUN_ID) if DEFAULT_RUN_ID in run_ids else 0

colA, colB = st.columns([2, 1])
with colA:
    run_id = st.selectbox("Pilih run_id", run_ids, index=default_idx)
with colB:
    st.write("Metadata run")
    st.dataframe(
        runs.loc[runs["run_id"].astype(str) == str(run_id), ["run_id", "approach", "threshold", "window_days", "run_time", "notes"]]
            .reset_index(drop=True),
        use_container_width=True,
        hide_index=True
    )

stats = load_run_stats(str(run_id))
if not stats.empty:
    st.write("Ringkasan statistik (dari cluster_summary)")
    st.dataframe(stats, use_container_width=True, hide_index=True)
else:
    st.info("Tidak ada statistik di cluster_summary untuk run ini (mungkin belum terisi).")

# ======================================================
# Sidebar Parameter
# ======================================================
st.sidebar.header("Filter & Parameter")

min_cluster_size = st.sidebar.number_input("min_cluster_size (n_tickets)", min_value=1, value=3, step=1)
max_cluster_size = st.sidebar.number_input("max_cluster_size (n_tickets)", min_value=1, value=999999, step=1)

sample_per_cluster = st.sidebar.number_input(
    "Sampling per cluster (0 = tidak sampling)", min_value=0, value=200, step=50
)

random_state = st.sidebar.number_input("random_state", min_value=0, value=42, step=1)

st.sidebar.subheader("TF-IDF")
max_features = st.sidebar.number_input("max_features", min_value=1000, value=20000, step=1000)
ngram_min = st.sidebar.selectbox("ngram_min", [1, 2], index=0)
ngram_max = st.sidebar.selectbox("ngram_max", [1, 2, 3], index=1)
if int(ngram_max) < int(ngram_min):
    st.sidebar.warning("ngram_max harus ≥ ngram_min")

st.sidebar.subheader("Percepatan")
use_svd = st.sidebar.checkbox("Gunakan TruncatedSVD sebelum t-SNE (disarankan)", value=True)
svd_dim = st.sidebar.number_input("svd_dim", min_value=10, value=50, step=10)

st.sidebar.subheader("t-SNE")
perplexity = st.sidebar.number_input("perplexity", min_value=5, value=30, step=5)
learning_rate = st.sidebar.number_input("learning_rate", min_value=10.0, value=200.0, step=10.0)
n_iter = st.sidebar.number_input("n_iter", min_value=250, value=1000, step=250)

# ======================================================
# Load cluster sizes from cluster_summary
# ======================================================
sizes = load_cluster_sizes(str(run_id))
if sizes.empty:
    st.warning("cluster_summary kosong untuk run ini. Tidak bisa filter berdasarkan n_tickets.")
    st.stop()

eligible = sizes[
    (sizes["n_tickets"] >= int(min_cluster_size)) &
    (sizes["n_tickets"] <= int(max_cluster_size))
].copy()

if eligible.empty:
    st.warning("Tidak ada cluster yang memenuhi filter min/max cluster size.")
    st.stop()

# Load members for run
members = load_members(str(run_id))
if members.empty:
    st.warning("cluster_members kosong untuk run ini.")
    st.stop()

# Filter members hanya cluster eligible
members2 = members.merge(
    eligible[["run_id", "cluster_id", "modul", "n_tickets"]],
    on=["run_id", "cluster_id", "modul"],
    how="inner",
).rename(columns={"n_tickets": "cluster_size"})

# Sampling per cluster (opsional)
if int(sample_per_cluster) > 0:
    members2 = (
        members2.groupby(["modul", "cluster_id"], group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), int(sample_per_cluster)), random_state=int(random_state)))
        .reset_index(drop=True)
    )

st.write(
    f"Data yang divisualisasikan: **{len(members2):,}** tiket dari "
    f"**{eligible.shape[0]:,}** cluster (setelah filter)."
)

# ======================================================
# 🧾 Ambil teks langsung dari cluster_members (text_sintaksis)
# ======================================================
TEXT_COL = "text_sintaksis"

df = members2.copy()
df[TEXT_COL] = df[TEXT_COL].fillna("").astype(str)

# buang teks kosong
df = df[df[TEXT_COL].str.len() > 0].reset_index(drop=True)
df["text_snippet"] = df[TEXT_COL].str.slice(0, 140)

if df.empty:
    st.warning("Semua text_sintaksis kosong. Pastikan kolom text_sintaksis terisi di cluster_members.")
    st.stop()


# ======================================================
# Run t-SNE
# ======================================================
btn = st.button("🔍 Jalankan t-SNE", type="primary")
if btn:
    with st.spinner("Menghitung TF-IDF, reduksi dimensi, dan t-SNE…"):
        Z = compute_tsne(
            texts=df[TEXT_COL].tolist(),
            random_state=int(random_state),
            max_features=int(max_features),
            ngram_min=int(ngram_min),
            ngram_max=int(ngram_max),
            use_svd=bool(use_svd),
            svd_dim=int(svd_dim),
            perplexity=int(perplexity),
            learning_rate=float(learning_rate),
            n_iter=int(n_iter),
        )

    df_plot = df.copy()
    df_plot["tsne_x"] = Z[:, 0]
    df_plot["tsne_y"] = Z[:, 1]

    st.subheader("Plot t-SNE 2D")

    chart = (
        alt.Chart(df_plot)
        .mark_circle(size=60, opacity=0.75)
        .encode(
            x=alt.X("tsne_x:Q", title="t-SNE X"),
            y=alt.Y("tsne_y:Q", title="t-SNE Y"),
            color=alt.Color("cluster_id:N", title="cluster_id"),
            tooltip=[
                alt.Tooltip("incident_number:N", title="incident_number"),
                alt.Tooltip("modul:N", title="modul"),
                alt.Tooltip("cluster_id:N", title="cluster_id"),
                alt.Tooltip("cluster_size:Q", title="cluster_size"),
                alt.Tooltip("tgl_submit:T", title="tgl_submit"),
                alt.Tooltip("text_snippet:N", title="snippet"),
            ],
        )
        .properties(height=520)
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Top 20 Cluster (berdasarkan n_tickets)")
    top = (
        eligible.sort_values("n_tickets", ascending=False)
        .head(20)
        .rename(columns={"n_tickets": "cluster_size"})
    )
    st.dataframe(top, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download hasil t-SNE (CSV)",
        data=df_plot[[
            "run_id", "modul", "cluster_id", "cluster_size",
            "incident_number", "tgl_submit", "tsne_x", "tsne_y", "text_snippet"
        ]].to_csv(index=False).encode("utf-8"),
        file_name=f"tsne_run_{run_id}.csv",
        mime="text/csv",
    )
else:
    st.info("Klik **Jalankan t-SNE** untuk menghasilkan plot.")
