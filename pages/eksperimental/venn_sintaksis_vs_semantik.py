from __future__ import annotations

import io
import math
import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


# =========================
# 🔐 Guard login
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# =========================
# ⚙️ Konstanta
# =========================
SCHEMA = "lasis_djp"

# Semantik (HDBSCAN)
T_SEM_RUNS = "modeling_semantic_hdbscan_runs"
T_SEM_MEM  = "modeling_semantic_hdbscan_members"

# Sintaksis
T_SYN_RUNS = "modeling_sintaksis_runs"
T_SYN_MEM  = "modeling_sintaksis_members"


# =========================
# 🔌 DB
# =========================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    user = cfg.get("user") or cfg.get("username")
    url = (
        f"postgresql+psycopg2://{user}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(show_spinner=False, ttl=120)
def load_sem_runs(_engine: Engine, limit: int = 200) -> pd.DataFrame:
    q = f"""
    SELECT
      modeling_id::text AS modeling_id,
      model_name,
      run_time,
      n_rows,
      n_clusters,
      n_noise
    FROM {SCHEMA}.{T_SEM_RUNS}
    ORDER BY run_time DESC
    LIMIT :lim
    """
    return pd.read_sql_query(text(q), _engine, params={"lim": int(limit)})


@st.cache_data(show_spinner=False, ttl=120)
def load_syn_runs(_engine: Engine, limit: int = 200) -> pd.DataFrame:
    q = f"""
    SELECT
      modeling_id::text AS modeling_id,
      run_time,
      threshold,
      knn_k,
      window_days,
      approach
    FROM {SCHEMA}.{T_SYN_RUNS}
    ORDER BY run_time DESC
    LIMIT :lim
    """
    return pd.read_sql_query(text(q), _engine, params={"lim": int(limit)})


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# =========================
# Query overlap (SERVER-SIDE, efisien)
# =========================
@st.cache_data(show_spinner=False, ttl=120)
def load_overlap_matrix(
    _engine: Engine,
    sem_modeling_id: str,
    syn_modeling_id: str,
    exclude_noise_sem: bool,
    exclude_singleton_sem: bool,
    exclude_singleton_syn: bool,
    top_sem: int,
    top_syn: int,
) -> pd.DataFrame:
    sem_noise_cond = "AND cluster_id <> -1" if exclude_noise_sem else ""

    sem_singleton_cond = "AND ss.sz >= 2" if exclude_singleton_sem else ""
    syn_singleton_cond = "AND ts.sz >= 2" if exclude_singleton_syn else ""

    q = f"""
    WITH
    sem_base AS (
      SELECT incident_number::text AS incident_number, cluster_id AS sem_cluster_id
      FROM {SCHEMA}.{T_SEM_MEM}
      WHERE modeling_id = :sem_id
      {sem_noise_cond}
    ),
    sem_sizes AS (
      SELECT sem_cluster_id, COUNT(*) AS sz
      FROM sem_base
      GROUP BY sem_cluster_id
    ),
    sem_keep AS (
      SELECT sem_cluster_id
      FROM sem_sizes ss
      WHERE 1=1
      {sem_singleton_cond}
      ORDER BY ss.sz DESC
      LIMIT :top_sem
    ),
    sem_filtered AS (
      SELECT b.incident_number, b.sem_cluster_id
      FROM sem_base b
      JOIN sem_keep k USING (sem_cluster_id)
    ),

    syn_base AS (
      SELECT incident_number::text AS incident_number, cluster_id AS syn_cluster_id
      FROM {SCHEMA}.{T_SYN_MEM}
      WHERE modeling_id = :syn_id
    ),
    syn_sizes AS (
      SELECT syn_cluster_id, COUNT(*) AS sz
      FROM syn_base
      GROUP BY syn_cluster_id
    ),
    syn_keep AS (
      SELECT syn_cluster_id
      FROM syn_sizes ts
      WHERE 1=1
      {syn_singleton_cond}
      ORDER BY ts.sz DESC
      LIMIT :top_syn
    ),
    syn_filtered AS (
      SELECT b.incident_number, b.syn_cluster_id
      FROM syn_base b
      JOIN syn_keep k USING (syn_cluster_id)
    )

    SELECT
      sf.sem_cluster_id,
      yf.syn_cluster_id,
      COUNT(*) AS n_overlap
    FROM sem_filtered sf
    JOIN syn_filtered yf
      ON yf.incident_number = sf.incident_number
    GROUP BY sf.sem_cluster_id, yf.syn_cluster_id
    ORDER BY n_overlap DESC
    """
    return pd.read_sql_query(
        text(q),
        _engine,
        params={
            "sem_id": sem_modeling_id,
            "syn_id": syn_modeling_id,
            "top_sem": int(top_sem),
            "top_syn": int(top_syn),
        },
    )


@st.cache_data(show_spinner=False, ttl=120)
def load_overlap_members(
    _engine: Engine,
    sem_modeling_id: str,
    syn_modeling_id: str,
    sem_cluster_id: int,
    syn_cluster_id: int,
    limit: int = 500,
    exclude_noise_sem: bool = True,
) -> pd.DataFrame:
    sem_noise_cond = "AND cluster_id <> -1" if exclude_noise_sem else ""
    q = f"""
    WITH sem AS (
      SELECT incident_number::text AS incident_number
      FROM {SCHEMA}.{T_SEM_MEM}
      WHERE modeling_id = :sem_id
        AND cluster_id = :sem_cid
      {sem_noise_cond}
    ),
    syn AS (
      SELECT incident_number::text AS incident_number
      FROM {SCHEMA}.{T_SYN_MEM}
      WHERE modeling_id = :syn_id
        AND cluster_id = :syn_cid
    )
    SELECT s.incident_number
    FROM sem s
    JOIN syn y USING (incident_number)
    ORDER BY s.incident_number
    LIMIT :lim
    """
    return pd.read_sql_query(
        text(q),
        _engine,
        params={
            "sem_id": sem_modeling_id,
            "syn_id": syn_modeling_id,
            "sem_cid": int(sem_cluster_id),
            "syn_cid": int(syn_cluster_id),
            "lim": int(limit),
        },
    )


# =========================
# Metrics: Purity & Entropy
# =========================
def compute_purity_entropy(df_ov: pd.DataFrame) -> pd.DataFrame:
    """
    Input df_ov columns: sem_cluster_id, syn_cluster_id, n_overlap
    Output per sem_cluster_id:
      - total_overlap_row
      - dominant_syn_cluster
      - dominant_overlap
      - purity
      - entropy (base2)
      - norm_entropy (0..1)
      - k_nonzero (jumlah syn cluster terlibat)
    """
    if df_ov.empty:
        return pd.DataFrame()

    d = df_ov.copy()
    d["sem_cluster_id"] = d["sem_cluster_id"].astype(int)
    d["syn_cluster_id"] = d["syn_cluster_id"].astype(int)
    d["n_overlap"] = d["n_overlap"].astype(int)

    # total per row
    row_sum = d.groupby("sem_cluster_id", as_index=False)["n_overlap"].sum().rename(columns={"n_overlap": "total_overlap_row"})

    # dominan (max cell per row)
    idx = d.groupby("sem_cluster_id")["n_overlap"].idxmax()
    dom = d.loc[idx, ["sem_cluster_id", "syn_cluster_id", "n_overlap"]].rename(
        columns={"syn_cluster_id": "dominant_syn_cluster", "n_overlap": "dominant_overlap"}
    )

    # entropy
    # p_ij = n_ij / row_sum
    d2 = d.merge(row_sum, on="sem_cluster_id", how="left")
    d2["p"] = d2["n_overlap"] / d2["total_overlap_row"]
    # entropy = -Σ p log2 p
    d2["plogp"] = d2["p"] * np.log2(d2["p"])
    ent = (-d2.groupby("sem_cluster_id", as_index=False)["plogp"].sum()).rename(columns={"plogp": "entropy"})

    # k_nonzero
    k = d2.groupby("sem_cluster_id", as_index=False)["syn_cluster_id"].nunique().rename(columns={"syn_cluster_id": "k_nonzero"})

    out = row_sum.merge(dom, on="sem_cluster_id").merge(ent, on="sem_cluster_id").merge(k, on="sem_cluster_id")

    out["purity"] = out["dominant_overlap"] / out["total_overlap_row"]

    # normalisasi entropy: / log2(k)
    def norm_ent(row):
        kk = int(row["k_nonzero"])
        if kk <= 1:
            return 0.0
        return float(row["entropy"]) / math.log2(kk)

    out["norm_entropy"] = out.apply(norm_ent, axis=1)

    # rapikan
    out = out.sort_values(["purity", "total_overlap_row"], ascending=[False, False]).reset_index(drop=True)
    return out


def compute_global_alignment(df_metrics: pd.DataFrame) -> dict[str, float]:
    """
    Ringkasan global:
    - weighted_purity: Σ (row_total * purity) / Σ row_total
    - weighted_norm_entropy: Σ (row_total * norm_entropy) / Σ row_total
    """
    if df_metrics.empty:
        return {"weighted_purity": float("nan"), "weighted_norm_entropy": float("nan")}

    w = df_metrics["total_overlap_row"].astype(float)
    ws = w.sum()
    if ws <= 0:
        return {"weighted_purity": float("nan"), "weighted_norm_entropy": float("nan")}

    wp = float((w * df_metrics["purity"]).sum() / ws)
    we = float((w * df_metrics["norm_entropy"]).sum() / ws)
    return {"weighted_purity": wp, "weighted_norm_entropy": we}


# =========================
# 🧭 UI
# =========================
st.title("🔥 Heatmap Overlap + Purity & Entropy — Semantik vs Sintaksis")
st.caption(
    "Heatmap menunjukkan jumlah irisan tiket per pasangan cluster. Purity & Entropy mengukur tingkat keselarasan cluster semantik terhadap cluster sintaksis."
)

engine = get_engine()

df_sem_runs = load_sem_runs(engine, limit=300)
df_syn_runs = load_syn_runs(engine, limit=300)

if df_sem_runs.empty:
    st.error("Tidak ada run HDBSCAN semantik.")
    st.stop()
if df_syn_runs.empty:
    st.error("Tidak ada run modeling sintaksis.")
    st.stop()

with st.sidebar:
    st.header("📌 Pilih Run")

    i_sem = st.selectbox(
        "Run Semantik (HDBSCAN)",
        options=list(range(len(df_sem_runs))),
        format_func=lambda i: f"{df_sem_runs.loc[i,'modeling_id']} | {df_sem_runs.loc[i,'model_name']} | rows={int(df_sem_runs.loc[i,'n_rows'] or 0):,}",
    )
    sem_modeling_id = str(df_sem_runs.loc[i_sem, "modeling_id"])

    i_syn = st.selectbox(
        "Run Sintaksis",
        options=list(range(len(df_syn_runs))),
        format_func=lambda i: f"{df_syn_runs.loc[i,'modeling_id']} | thr={df_syn_runs.loc[i,'threshold']} | knn={df_syn_runs.loc[i,'knn_k']} | win={df_syn_runs.loc[i,'window_days']}",
    )
    syn_modeling_id = str(df_syn_runs.loc[i_syn, "modeling_id"])

    st.divider()
    st.header("⚙️ Filter")
    exclude_noise_sem = st.checkbox("Semantik: exclude noise (-1)", value=True)
    exclude_singleton_sem = st.checkbox("Semantik: exclude singleton (size<2)", value=True)
    exclude_singleton_syn = st.checkbox("Sintaksis: exclude singleton (size<2)", value=True)

    st.divider()
    st.header("🎛️ Ukuran Heatmap")
    top_sem = st.slider("Top-N cluster Semantik (by size)", 5, 120, 30, 5)
    top_syn = st.slider("Top-M cluster Sintaksis (by size)", 5, 200, 60, 5)

    st.divider()
    st.header("🔍 Drilldown cell")
    drill_limit = st.number_input("Limit incident_number overlap", min_value=50, max_value=5000, value=500, step=50)


# Load overlap
df_ov = load_overlap_matrix(
    engine,
    sem_modeling_id=sem_modeling_id,
    syn_modeling_id=syn_modeling_id,
    exclude_noise_sem=exclude_noise_sem,
    exclude_singleton_sem=exclude_singleton_sem,
    exclude_singleton_syn=exclude_singleton_syn,
    top_sem=int(top_sem),
    top_syn=int(top_syn),
)

if df_ov.empty:
    st.warning("Overlap kosong. Coba naikkan Top-N/Top-M atau matikan filter singleton/noise.")
    st.stop()

# KPI overlap
total_overlap = int(df_ov["n_overlap"].sum())
max_cell = int(df_ov["n_overlap"].max())
c1, c2, c3 = st.columns(3)
c1.metric("Total overlap (sum cells)", f"{total_overlap:,}")
c2.metric("Max overlap (cell)", f"{max_cell:,}")
c3.metric("Pasangan cluster (cells)", f"{len(df_ov):,}")

# Heatmap
st.subheader("Heatmap Overlap (Semantik × Sintaksis)")
df_plot = df_ov.copy()
df_plot["sem_cluster_id"] = df_plot["sem_cluster_id"].astype(int).astype(str)
df_plot["syn_cluster_id"] = df_plot["syn_cluster_id"].astype(int).astype(str)

heat = (
    alt.Chart(df_plot)
    .mark_rect()
    .encode(
        x=alt.X("syn_cluster_id:O", title="Cluster Sintaksis", sort=None),
        y=alt.Y("sem_cluster_id:O", title="Cluster Semantik", sort=None),
        color=alt.Color("n_overlap:Q", title="Overlap (count)"),
        tooltip=[
            alt.Tooltip("sem_cluster_id:O", title="Semantik"),
            alt.Tooltip("syn_cluster_id:O", title="Sintaksis"),
            alt.Tooltip("n_overlap:Q", title="n_overlap"),
        ],
    )
    .properties(height=520)
)
st.altair_chart(heat, use_container_width=True)

# Table + download overlap matrix
st.subheader("Tabel Overlap (sorted by n_overlap desc)")
st.dataframe(df_ov.sort_values("n_overlap", ascending=False), use_container_width=True, height=420)

st.download_button(
    "⬇️ Download overlap matrix (CSV)",
    data=df_to_csv_bytes(df_ov),
    file_name=f"overlap_matrix_sem_{sem_modeling_id}_syn_{syn_modeling_id}.csv",
    mime="text/csv",
)

# =========================
# Purity & Entropy section
# =========================
st.subheader("Purity & Entropy (per Cluster Semantik)")

df_metrics = compute_purity_entropy(df_ov)
global_m = compute_global_alignment(df_metrics)

g1, g2, g3 = st.columns(3)
g1.metric("Weighted Purity (global)", f"{global_m['weighted_purity']:.4f}")
g2.metric("Weighted Norm Entropy (global)", f"{global_m['weighted_norm_entropy']:.4f}")
g3.metric("Jumlah sem cluster (dihitung)", f"{len(df_metrics):,}")

st.caption(
    "Interpretasi: Purity tinggi & Entropy rendah → cluster semantik selaras ke satu cluster sintaksis dominan. "
    "Purity rendah & Entropy tinggi → cluster semantik menyebar ke banyak cluster sintaksis (tema lebih luas)."
)

# Tabel metrik
st.dataframe(
    df_metrics[[
        "sem_cluster_id", "total_overlap_row",
        "dominant_syn_cluster", "dominant_overlap",
        "purity", "entropy", "norm_entropy", "k_nonzero"
    ]],
    use_container_width=True,
    height=420,
)

st.download_button(
    "⬇️ Download purity_entropy (CSV)",
    data=df_to_csv_bytes(df_metrics),
    file_name=f"purity_entropy_sem_{sem_modeling_id}_syn_{syn_modeling_id}.csv",
    mime="text/csv",
)

# Visual: top 30 by total_overlap_row
st.subheader("Visual: Purity vs Norm Entropy (Top 30 Cluster Semantik by overlap)")
df_top = df_metrics.sort_values("total_overlap_row", ascending=False).head(30).copy()
df_top["sem_cluster_id"] = df_top["sem_cluster_id"].astype(int).astype(str)

scatter = (
    alt.Chart(df_top)
    .mark_circle(size=120)
    .encode(
        x=alt.X("purity:Q", title="purity (max overlap / total overlap)"),
        y=alt.Y("norm_entropy:Q", title="normalized entropy (0..1)"),
        tooltip=[
            "sem_cluster_id",
            alt.Tooltip("total_overlap_row:Q", format=","),
            alt.Tooltip("dominant_syn_cluster:Q"),
            alt.Tooltip("purity:Q", format=".4f"),
            alt.Tooltip("norm_entropy:Q", format=".4f"),
        ],
    )
    .properties(height=360)
)
st.altair_chart(scatter, use_container_width=True)

# =========================
# Drilldown cell
# =========================
st.subheader("Drilldown: incident_number overlap untuk 1 cell (pasangan cluster)")

top_pairs = df_ov.sort_values("n_overlap", ascending=False).head(min(200, len(df_ov))).copy()
pair_options = list(range(len(top_pairs)))

pick = st.selectbox(
    "Pilih pasangan cluster (top cells)",
    options=pair_options,
    format_func=lambda i: (
        f"Sem={int(top_pairs.iloc[i]['sem_cluster_id'])} "
        f"vs Syn={int(top_pairs.iloc[i]['syn_cluster_id'])} "
        f"| overlap={int(top_pairs.iloc[i]['n_overlap']):,}"
    ),
)

sem_cid = int(top_pairs.iloc[pick]["sem_cluster_id"])
syn_cid = int(top_pairs.iloc[pick]["syn_cluster_id"])

df_ids = load_overlap_members(
    engine,
    sem_modeling_id=sem_modeling_id,
    syn_modeling_id=syn_modeling_id,
    sem_cluster_id=sem_cid,
    syn_cluster_id=syn_cid,
    limit=int(drill_limit),
    exclude_noise_sem=exclude_noise_sem,
)

st.write(f"Menampilkan overlap incident_number (limit {int(drill_limit):,}) untuk Sem={sem_cid} & Syn={syn_cid}")
st.dataframe(df_ids, use_container_width=True, height=360)
