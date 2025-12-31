# pages/modeling_sintaksis_evaluation_unsupervised.py
# ============================================================
# Evaluasi Modeling Sintaksis (Unsupervised) — by job_id & threshold
#
# Metrics:
# - Silhouette Score (cosine)  -> computed on TF-IDF sparse matrix
# - Davies-Bouldin Index (DBI) -> computed on SVD-reduced embedding (euclidean)
#
# Sources:
# - lasis_djp.modeling_sintaksis_runs
# - lasis_djp.modeling_sintaksis_members
# - lasis_djp.modeling_sintaksis_clusters (optional, for singleton filtering)
#
# Optional TF-IDF sources (auto-detect):
# - lasis_djp.incident_tfidf_runs & lasis_djp.incident_tfidf_vectors
#   OR
# - lasis_djp.incident_modeling_tfidf_runs & lasis_djp.incident_modeling_tfidf_vectors
#
# Notes:
# - This page computes metrics per threshold within a selected job_id.
# - For each threshold, it uses the latest run_time (modeling_id) by default.
# - Sampling is supported for performance on large datasets.
# ============================================================

from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD


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
T_RUNS = "modeling_sintaksis_runs"
T_MEMBERS = "modeling_sintaksis_members"
T_CLUSTERS = "modeling_sintaksis_clusters"

TFIDF_CANDIDATES = [
    ("incident_tfidf_runs", "incident_tfidf_vectors"),
    ("incident_modeling_tfidf_runs", "incident_modeling_tfidf_vectors"),
]


# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


# ======================================================
# 🧰 Helpers
# ======================================================
@st.cache_data(show_spinner=False)
def table_exists(_engine, schema: str, table: str) -> bool:
    q = text(
        """
        SELECT EXISTS (
          SELECT 1
          FROM information_schema.tables
          WHERE table_schema = :schema
            AND table_name = :table
        ) AS ok
        """
    )
    with _engine.connect() as conn:
        return bool(conn.execute(q, {"schema": schema, "table": table}).scalar())


def detect_tfidf_tables(_engine) -> Optional[Tuple[str, str]]:
    for tr, tv in TFIDF_CANDIDATES:
        if table_exists(_engine, SCHEMA, tr) and table_exists(_engine, SCHEMA, tv):
            return tr, tv
    return None


@st.cache_data(show_spinner=False)
def load_jobs(_engine) -> pd.DataFrame:
    q = text(f"""
        SELECT
            job_id,
            MIN(run_time) AS first_run_time,
            MAX(run_time) AS last_run_time,
            COUNT(*) AS n_runs
        FROM {SCHEMA}.{T_RUNS}
        WHERE job_id IS NOT NULL
        GROUP BY job_id
        ORDER BY last_run_time DESC
    """)
    with _engine.connect() as conn:
        return pd.read_sql(q, conn)


@st.cache_data(show_spinner=False)
def load_runs_by_job(_engine, job_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
            job_id,
            modeling_id,
            run_time,
            approach,
            threshold,
            window_days,
            knn_k,
            min_cluster_size,
            n_rows,
            n_clusters_all,
            n_singletons,
            vocab_size,
            nnz,
            elapsed_sec,
            tfidf_run_id,
            params_json,
            notes
        FROM {SCHEMA}.{T_RUNS}
        WHERE job_id = :job_id
        ORDER BY threshold ASC, run_time DESC
    """)
    with _engine.connect() as conn:
        return pd.read_sql(q, conn, params={"job_id": job_id})


@st.cache_data(show_spinner=False)
def load_members_for_modeling(_engine, modeling_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
            modeling_id, cluster_id, incident_number
        FROM {SCHEMA}.{T_MEMBERS}
        WHERE modeling_id = :modeling_id
        ORDER BY cluster_id, incident_number
    """)
    with _engine.connect() as conn:
        return pd.read_sql(q, conn, params={"modeling_id": modeling_id})


@st.cache_data(show_spinner=False)
def load_cluster_sizes_for_modeling(_engine, modeling_id: str) -> pd.DataFrame:
    # For singleton filtering
    q = text(f"""
        SELECT cluster_id, cluster_size
        FROM {SCHEMA}.{T_CLUSTERS}
        WHERE modeling_id = :modeling_id
    """)
    with _engine.connect() as conn:
        return pd.read_sql(q, conn, params={"modeling_id": modeling_id})


@st.cache_data(show_spinner=False)
def load_tfidf_run_meta(_engine, tfidf_runs_table: str, tfidf_run_id: str) -> List[str]:
    """
    Return feature_names list from tfidf runs table.
    Expected column: feature_names_json (jsonb)
    """
    q = text(f"""
        SELECT feature_names_json
        FROM {SCHEMA}.{tfidf_runs_table}
        WHERE run_id = :run_id
        LIMIT 1
    """)
    with _engine.connect() as conn:
        row = conn.execute(q, {"run_id": tfidf_run_id}).fetchone()

    if not row:
        raise RuntimeError(f"TF-IDF run_id {tfidf_run_id} tidak ditemukan di {SCHEMA}.{tfidf_runs_table}.")

    feature_names = row[0]
    if isinstance(feature_names, str):
        feature_names = json.loads(feature_names)

    if not isinstance(feature_names, list) or not feature_names:
        raise RuntimeError("feature_names_json kosong/tidak valid pada TF-IDF run.")

    return feature_names


@st.cache_data(show_spinner=False)
def load_tfidf_vectors_for_incidents(
    _engine,
    tfidf_vectors_table: str,
    tfidf_run_id: str,
    incident_numbers: Tuple[str, ...],
) -> pd.DataFrame:
    """
    Load tfidf_json for selected incidents.
    Expected columns: incident_number, tfidf_json, run_id
    """
    q = text(f"""
        SELECT incident_number, tfidf_json
        FROM {SCHEMA}.{tfidf_vectors_table}
        WHERE run_id = :run_id
          AND incident_number = ANY(:inc_arr)
    """)
    with _engine.connect() as conn:
        return pd.read_sql(q, conn, params={"run_id": tfidf_run_id, "inc_arr": list(incident_numbers)})


def build_sparse_matrix_from_json(
    df_vec: pd.DataFrame,
    feature_names: List[str],
    incident_order: List[str],
) -> csr_matrix:
    term_to_idx = {t: i for i, t in enumerate(feature_names)}

    # incident_number -> dict(term->value)
    m: Dict[str, Dict[str, float]] = {}
    for _, r in df_vec.iterrows():
        v = r["tfidf_json"]
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict):
            m[str(r["incident_number"])] = v

    rows, cols, data = [], [], []
    for i, inc in enumerate(incident_order):
        d = m.get(str(inc), {})
        if not d:
            continue
        for term, val in d.items():
            j = term_to_idx.get(term)
            if j is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            if fv != 0.0 and not math.isnan(fv):
                rows.append(i)
                cols.append(j)
                data.append(fv)

    return csr_matrix((data, (rows, cols)), shape=(len(incident_order), len(feature_names)), dtype=np.float32)


def compute_unsupervised_metrics(
    X: csr_matrix,
    labels: np.ndarray,
    svd_dim: int = 64,
    max_rows: int = 50000,
    random_seed: int = 42,
) -> Dict[str, float]:
    """
    Silhouette (cosine) on X (optionally sampled),
    DBI on SVD embedding (euclidean).
    """
    # Need at least 2 clusters
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"silhouette": np.nan, "dbi": np.nan, "n_eval": int(X.shape[0])}

    n = X.shape[0]
    if n > max_rows:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(n, size=max_rows, replace=False)
        X_use = X[idx]
        labels_use = labels[idx]
        n_eval = int(max_rows)
    else:
        X_use = X
        labels_use = labels
        n_eval = int(n)

    # Silhouette with cosine
    try:
        sil = float(silhouette_score(X_use, labels_use, metric="cosine"))
    except Exception:
        sil = np.nan

    # DBI on SVD embedding (euclidean)
    try:
        dim = min(int(svd_dim), max(2, X_use.shape[1] - 1))
        svd = TruncatedSVD(n_components=dim, random_state=random_seed)
        Z = svd.fit_transform(X_use)
        dbi = float(davies_bouldin_score(Z, labels_use))
    except Exception:
        dbi = np.nan

    return {"silhouette": sil, "dbi": dbi, "n_eval": n_eval}


@st.cache_data(show_spinner=False)
def eval_one_run_cached(
    _engine,
    modeling_id: str,
    tfidf_runs_table: str,
    tfidf_vectors_table: str,
    tfidf_run_id: str,
    exclude_singletons: bool,
    svd_dim: int,
    max_rows: int,
) -> Dict[str, float]:
    """
    Cached evaluation for one modeling_id.
    """
    dfm = load_members_for_modeling(_engine, modeling_id)
    if dfm.empty:
        return {"silhouette": np.nan, "dbi": np.nan, "n_eval": 0, "n_used": 0, "n_labels": 0}

    labels = dfm["cluster_id"].to_numpy()
    incs = dfm["incident_number"].astype(str).tolist()

    if exclude_singletons:
        dfsz = load_cluster_sizes_for_modeling(_engine, modeling_id)
        size_map = dfsz.set_index("cluster_id")["cluster_size"].to_dict()
        mask = np.array([size_map.get(int(cid), 1) > 1 for cid in labels], dtype=bool)
        dfm = dfm.loc[mask].copy()
        if dfm.empty:
            return {"silhouette": np.nan, "dbi": np.nan, "n_eval": 0, "n_used": 0, "n_labels": 0}
        labels = dfm["cluster_id"].to_numpy()
        incs = dfm["incident_number"].astype(str).tolist()

    feature_names = load_tfidf_run_meta(_engine, tfidf_runs_table, tfidf_run_id)
    df_vec = load_tfidf_vectors_for_incidents(_engine, tfidf_vectors_table, tfidf_run_id, tuple(incs))
    X = build_sparse_matrix_from_json(df_vec, feature_names, incs)

    # Some incidents might be missing tfidf_json rows; keep but metrics may degrade.
    out = compute_unsupervised_metrics(X, labels, svd_dim=svd_dim, max_rows=max_rows)
    out["n_used"] = int(X.shape[0])
    out["n_labels"] = int(len(np.unique(labels)))
    return out


# ======================================================
# 🧾 UI
# ======================================================

st.title("Evaluasi Unsupervised Modeling Sintaksis")
st.caption("Menampilkan Silhouette Score & Davies–Bouldin Index (DBI) untuk setiap cosine similarity threshold dalam satu job_id.")

engine = get_engine()

tfidf_pair = detect_tfidf_tables(engine)
if not tfidf_pair:
    st.error(
        "Tidak menemukan tabel TF-IDF yang dibutuhkan. "
        "Pastikan ada salah satu pasangan tabel:\n"
        f"- {SCHEMA}.incident_tfidf_runs + {SCHEMA}.incident_tfidf_vectors\n"
        f"- {SCHEMA}.incident_modeling_tfidf_runs + {SCHEMA}.incident_modeling_tfidf_vectors"
    )
    st.stop()

tfidf_runs_table, tfidf_vectors_table = tfidf_pair
st.info(f"TF-IDF source terdeteksi: {SCHEMA}.{tfidf_runs_table} + {SCHEMA}.{tfidf_vectors_table}")

jobs = load_jobs(engine)
if jobs.empty:
    st.warning("Belum ada data pada modeling_sintaksis_runs.")
    st.stop()

job_id = st.sidebar.selectbox(
    "Pilih job_id",
    options=jobs["job_id"].astype(str).tolist(),
    format_func=lambda x: x,
)

runs = load_runs_by_job(engine, job_id)
if runs.empty:
    st.warning("Tidak ada run untuk job_id tersebut.")
    st.stop()

# Config
st.sidebar.markdown("---")
st.sidebar.subheader("Konfigurasi evaluasi")
exclude_singletons = st.sidebar.checkbox("Abaikan singleton saat evaluasi", value=True)
svd_dim = st.sidebar.slider("SVD dim (untuk DBI)", min_value=16, max_value=256, value=64, step=16)
max_rows = st.sidebar.number_input("Max rows (sampling)", min_value=2000, max_value=200000, value=50000, step=1000)

# Choose per-threshold run strategy
st.sidebar.markdown("---")
st.sidebar.subheader("Strategi pemilihan run per threshold")
strategy = st.sidebar.radio(
    "Gunakan run mana untuk tiap threshold?",
    options=["Run terbaru (run_time max)", "Run pertama (run_time min)"],
    index=0,
)

# Prepare per-threshold selected run
runs2 = runs.dropna(subset=["threshold"]).copy()
if runs2.empty:
    st.warning("Kolom threshold kosong pada job ini.")
    st.stop()

runs2["run_time"] = pd.to_datetime(runs2["run_time"])
ascending = True if strategy == "Run pertama (run_time min)" else False
# sort to pick top row per threshold
runs2 = runs2.sort_values(["threshold", "run_time"], ascending=[True, ascending])
picked = runs2.groupby("threshold", as_index=False).head(1).copy()
picked = picked.sort_values("threshold")

# Header summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("job_id", str(job_id))
c2.metric("Jumlah threshold", f"{picked['threshold'].nunique():,}")
c3.metric("Total runs (job)", f"{len(runs):,}")
c4.metric("Run selection", "latest" if strategy.startswith("Run terbaru") else "first")

with st.expander("Daftar run dalam job_id (raw)", expanded=False):
    tmp = runs.copy()
    tmp["params_json"] = tmp["params_json"].astype(str)
    st.dataframe(tmp.sort_values(["threshold", "run_time"], ascending=[True, False]), use_container_width=True)

# ======================================================
# Compute metrics per threshold
# ======================================================
st.subheader("Hasil evaluasi per threshold")

rows = []
progress = st.progress(0)
status = st.empty()

total = len(picked)
for i, r in enumerate(picked.itertuples(index=False), start=1):
    thr = float(r.threshold)
    mid = str(r.modeling_id)
    tfidf_run_id = r.tfidf_run_id

    status.write(f"Menghitung threshold={thr:.4f} | modeling_id={mid} ...")

    if not tfidf_run_id:
        rows.append({
            "threshold": thr,
            "modeling_id": mid,
            "run_time": r.run_time,
            "n_rows": r.n_rows,
            "n_clusters_all": r.n_clusters_all,
            "n_singletons": r.n_singletons,
            "silhouette_cosine": np.nan,
            "dbi": np.nan,
            "n_eval": 0,
            "n_labels": 0,
            "note": "tfidf_run_id kosong"
        })
    else:
        try:
            out = eval_one_run_cached(
                engine,
                modeling_id=mid,
                tfidf_runs_table=tfidf_runs_table,
                tfidf_vectors_table=tfidf_vectors_table,
                tfidf_run_id=str(tfidf_run_id),
                exclude_singletons=exclude_singletons,
                svd_dim=int(svd_dim),
                max_rows=int(max_rows),
            )
            rows.append({
                "threshold": thr,
                "modeling_id": mid,
                "run_time": r.run_time,
                "n_rows": r.n_rows,
                "n_clusters_all": r.n_clusters_all,
                "n_singletons": r.n_singletons,
                "silhouette_cosine": out.get("silhouette", np.nan),
                "dbi": out.get("dbi", np.nan),
                "n_eval": out.get("n_eval", 0),
                "n_labels": out.get("n_labels", 0),
                "note": ""
            })
        except Exception as e:
            rows.append({
                "threshold": thr,
                "modeling_id": mid,
                "run_time": r.run_time,
                "n_rows": r.n_rows,
                "n_clusters_all": r.n_clusters_all,
                "n_singletons": r.n_singletons,
                "silhouette_cosine": np.nan,
                "dbi": np.nan,
                "n_eval": 0,
                "n_labels": 0,
                "note": f"error: {e}"
            })

    progress.progress(i / total)

status.empty()
progress.empty()

df_eval = pd.DataFrame(rows).sort_values("threshold")

# KPI: best threshold candidates (heuristic)
# - silhouette higher is better
# - dbi lower is better
best_sil = df_eval.loc[df_eval["silhouette_cosine"].idxmax()] if df_eval["silhouette_cosine"].notna().any() else None
best_dbi = df_eval.loc[df_eval["dbi"].idxmin()] if df_eval["dbi"].notna().any() else None

k1, k2, k3, k4 = st.columns(4)
if best_sil is not None and pd.notna(best_sil["silhouette_cosine"]):
    k1.metric("Silhouette terbaik", f"{best_sil['silhouette_cosine']:.4f}", f"thr={best_sil['threshold']:.4f}")
else:
    k1.metric("Silhouette terbaik", "N/A")
if best_dbi is not None and pd.notna(best_dbi["dbi"]):
    k2.metric("DBI terbaik", f"{best_dbi['dbi']:.4f}", f"thr={best_dbi['threshold']:.4f}")
else:
    k2.metric("DBI terbaik", "N/A")
k3.metric("Sampling n_eval (median)", f"{int(df_eval['n_eval'].median()):,}" if not df_eval.empty else "0")
k4.metric("Exclude singleton", "Ya" if exclude_singletons else "Tidak")

st.dataframe(df_eval, use_container_width=True)

# ======================================================
# Charts
# ======================================================
st.subheader("Visualisasi metrik vs threshold")

df_plot = df_eval.copy()
df_plot["threshold"] = df_plot["threshold"].astype(float)

ch_sil = (
    alt.Chart(df_plot.dropna(subset=["silhouette_cosine"]))
    .mark_line(point=True)
    .encode(
        x=alt.X("threshold:Q", title="Cosine similarity threshold"),
        y=alt.Y("silhouette_cosine:Q", title="Silhouette Score (cosine)"),
        tooltip=["threshold:Q", "silhouette_cosine:Q", "n_eval:Q", "n_labels:Q", "n_clusters_all:Q", "n_singletons:Q"],
    )
    .properties(height=280)
)
st.altair_chart(ch_sil, use_container_width=True)

ch_dbi = (
    alt.Chart(df_plot.dropna(subset=["dbi"]))
    .mark_line(point=True)
    .encode(
        x=alt.X("threshold:Q", title="Cosine similarity threshold"),
        y=alt.Y("dbi:Q", title="Davies–Bouldin Index (lebih kecil lebih baik)"),
        tooltip=["threshold:Q", "dbi:Q", "n_eval:Q", "n_labels:Q", "n_clusters_all:Q", "n_singletons:Q"],
    )
    .properties(height=280)
)
st.altair_chart(ch_dbi, use_container_width=True)

# ======================================================
# Export
# ======================================================
st.download_button(
    "Download CSV (evaluasi per threshold)",
    data=df_eval.to_csv(index=False).encode("utf-8"),
    file_name=f"eval_unsupervised_job_{job_id}.csv",
    mime="text/csv",
)
