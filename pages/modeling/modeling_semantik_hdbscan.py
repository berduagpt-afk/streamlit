# pages/modeling_semantic_hdbscan.py
# ============================================================
# Semantic Clustering — HDBSCAN (berbasis embedding)
#
# Input:
# - lasis_djp.incident_semantic_embeddings
#
# Output:
# - lasis_djp.modeling_semantic_hdbscan_runs
# - lasis_djp.modeling_semantic_hdbscan_clusters
# - lasis_djp.modeling_semantic_hdbscan_members
#
# Fitur:
# - Pilih embedding run_id + model_name
# - Load embedding dari DB (limit / random sample)
# - HDBSCAN clustering
# - Evaluasi internal: Silhouette & Davies-Bouldin (exclude noise)
# - Simpan hasil clustering ke DB
#
# PATCH FINAL:
# ✅ Fix error "Unrecognized metric 'cosine'" pada HDBSCAN (BallTree):
#    - Jika user pilih cosine -> L2-normalize X lalu pakai metric euclidean untuk HDBSCAN
#    - Silhouette tetap boleh pakai cosine
# ✅ Hindari UnhashableParamError: arg Engine di cache pakai _engine
# ✅ Insert params_json via psycopg2.extras.Json (aman jsonb)
# ✅ Upsert cepat via psycopg2 execute_values
# ============================================================

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from psycopg2.extras import Json, execute_values

# deps
try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    hdbscan = None
    _HAS_HDBSCAN = False

try:
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    _HAS_SK = True
except Exception:
    silhouette_score = None
    davies_bouldin_score = None
    _HAS_SK = False


# =========================
# 🔐 Guard login
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()


# =========================
# ⚙️ Konstanta
# =========================
SCHEMA = "lasis_djp"

T_EMB = "incident_semantic_embeddings"  # dari embedding builder

T_RUNS = "modeling_semantic_hdbscan_runs"
T_CLUST = "modeling_semantic_hdbscan_clusters"
T_MEM = "modeling_semantic_hdbscan_members"


# =========================
# 🔌 DB
# =========================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


def ensure_tables(engine: Engine) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_RUNS} (
      modeling_id uuid PRIMARY KEY,
      embedding_run_id uuid NOT NULL,
      model_name text NOT NULL,
      run_time timestamptz NOT NULL DEFAULT now(),
      params_json jsonb NOT NULL,
      notes text,
      n_rows integer,
      n_clusters integer,
      n_noise integer,
      silhouette double precision,
      dbi double precision
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_CLUST} (
      modeling_id uuid NOT NULL,
      cluster_id integer NOT NULL,
      cluster_size integer NOT NULL,
      pct double precision,
      PRIMARY KEY (modeling_id, cluster_id)
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_MEM} (
      modeling_id uuid NOT NULL,
      incident_number text NOT NULL,
      cluster_id integer NOT NULL,
      score double precision,
      PRIMARY KEY (modeling_id, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{T_MEM}_cluster ON {SCHEMA}.{T_MEM} (modeling_id, cluster_id);
    CREATE INDEX IF NOT EXISTS idx_{T_RUNS}_emb_run ON {SCHEMA}.{T_RUNS} (embedding_run_id);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


@st.cache_data(show_spinner=False, ttl=300)
def list_embedding_runs(_engine: Engine) -> pd.DataFrame:
    q = f"""
    SELECT
      run_id::text AS run_id,
      model_name,
      COUNT(*) AS n_rows,
      MAX(created_at) AS last_created_at
    FROM {SCHEMA}.{T_EMB}
    GROUP BY run_id, model_name
    ORDER BY last_created_at DESC
    LIMIT 200
    """
    return pd.read_sql_query(text(q), _engine)


def load_embeddings(
    engine: Engine,
    embedding_run_id: str,
    model_name: str,
    limit_rows: int,
    sampling: str,
    random_seed: int,
) -> pd.DataFrame:
    # Ambil embedding float8[] -> python list -> numpy
    if sampling == "random":
        q = f"""
        SELECT incident_number, tgl_submit, modul, sub_modul, embedding
        FROM {SCHEMA}.{T_EMB}
        WHERE run_id = :run_id AND model_name = :model_name
        ORDER BY md5(incident_number || :seed)
        LIMIT :lim
        """
        params = {"run_id": embedding_run_id, "model_name": model_name, "lim": int(limit_rows), "seed": str(random_seed)}
    else:
        q = f"""
        SELECT incident_number, tgl_submit, modul, sub_modul, embedding
        FROM {SCHEMA}.{T_EMB}
        WHERE run_id = :run_id AND model_name = :model_name
        ORDER BY created_at ASC
        LIMIT :lim
        """
        params = {"run_id": embedding_run_id, "model_name": model_name, "lim": int(limit_rows)}

    return pd.read_sql_query(text(q), engine, params=params)


def upsert_clusters(engine: Engine, modeling_id: str, df_stats: pd.DataFrame) -> None:
    values = []
    for _, r in df_stats.iterrows():
        values.append((modeling_id, int(r["cluster_id"]), int(r["cluster_size"]), float(r["pct"])))

    sql = f"""
    INSERT INTO {SCHEMA}.{T_CLUST} (modeling_id, cluster_id, cluster_size, pct)
    VALUES %s
    ON CONFLICT (modeling_id, cluster_id) DO UPDATE SET
      cluster_size = EXCLUDED.cluster_size,
      pct = EXCLUDED.pct
    """

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            execute_values(cur, sql, values, page_size=min(5000, max(1000, len(values))))
        raw.commit()
    finally:
        raw.close()


def upsert_members(engine: Engine, modeling_id: str, df_members: pd.DataFrame) -> None:
    values = []
    for _, r in df_members.iterrows():
        values.append((
            modeling_id,
            str(r["incident_number"]),
            int(r["cluster_id"]),
            (float(r["score"]) if pd.notna(r["score"]) else None),
        ))

    sql = f"""
    INSERT INTO {SCHEMA}.{T_MEM} (modeling_id, incident_number, cluster_id, score)
    VALUES %s
    ON CONFLICT (modeling_id, incident_number) DO UPDATE SET
      cluster_id = EXCLUDED.cluster_id,
      score = EXCLUDED.score
    """

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            execute_values(cur, sql, values, page_size=min(5000, max(1000, len(values))))
        raw.commit()
    finally:
        raw.close()


def insert_run(engine: Engine, payload: Dict[str, Any]) -> None:
    # IMPORTANT: NO CAST :param::uuid di SQLAlchemy
    q = f"""
    INSERT INTO {SCHEMA}.{T_RUNS}
    (modeling_id, embedding_run_id, model_name, run_time, params_json, notes, n_rows, n_clusters, n_noise, silhouette, dbi)
    VALUES
    (:modeling_id, :embedding_run_id, :model_name, now(), :params_json, :notes, :n_rows, :n_clusters, :n_noise, :silhouette, :dbi)
    """
    with engine.begin() as conn:
        conn.execute(
            text(q),
            {
                "modeling_id": payload["modeling_id"],
                "embedding_run_id": payload["embedding_run_id"],
                "model_name": payload["model_name"],
                "params_json": Json(payload["params_json"]),
                "notes": payload.get("notes", "") or "",
                "n_rows": payload.get("n_rows"),
                "n_clusters": payload.get("n_clusters"),
                "n_noise": payload.get("n_noise"),
                "silhouette": payload.get("silhouette"),
                "dbi": payload.get("dbi"),
            },
        )


# =========================
# Utils
# =========================
def to_matrix(df: pd.DataFrame) -> np.ndarray:
    # df["embedding"] adalah list[float] (hasil parsing psycopg2)
    X = np.vstack(df["embedding"].apply(lambda v: np.asarray(v, dtype=np.float32)).to_numpy())
    return X


def l2_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def summarize_clusters(labels: np.ndarray) -> pd.DataFrame:
    s = pd.Series(labels, name="cluster_id")
    g = s.value_counts(dropna=False).sort_index().reset_index()
    g.columns = ["cluster_id", "cluster_size"]
    g["pct"] = g["cluster_size"] / g["cluster_size"].sum() * 100.0
    return g


def compute_metrics(X: np.ndarray, labels: np.ndarray, silhouette_metric: str) -> Tuple[Optional[float], Optional[float], int, int]:
    mask = labels != -1
    n_noise = int((labels == -1).sum())
    unique_clusters = sorted(set(labels[mask].tolist()))
    n_clusters = len(unique_clusters)

    sil = None
    dbi = None
    if _HAS_SK and mask.sum() >= 3 and n_clusters >= 2:
        try:
            sil = float(silhouette_score(X[mask], labels[mask], metric=silhouette_metric))
        except Exception:
            sil = None
        try:
            # DBI tidak menerima metric; gunakan X yang sudah sesuai (dinormalisasi jika perlu)
            dbi = float(davies_bouldin_score(X[mask], labels[mask]))
        except Exception:
            dbi = None

    return sil, dbi, n_clusters, n_noise


# =========================
# 🧭 UI
# =========================
st.title("🌌 Semantic Clustering — HDBSCAN")
st.caption(
    "HDBSCAN melakukan clustering langsung pada embedding (tanpa threshold cosine) dan mendeteksi noise (cluster_id = -1)."
)

engine = get_engine()
ensure_tables(engine)

if not _HAS_HDBSCAN:
    st.warning("Package `hdbscan` belum tersedia. Install di environment Streamlit: `pip install hdbscan`")
if not _HAS_SK:
    st.warning("Package `scikit-learn` belum tersedia. Install: `pip install scikit-learn`")

df_runs = list_embedding_runs(engine)
if df_runs.empty:
    st.error(f"Tidak ada data di `{SCHEMA}.{T_EMB}`. Jalankan embedding dulu.")
    st.stop()

with st.sidebar:
    st.header("🔎 Pilih Embedding Run")

    pick = st.selectbox(
        "Embedding run (run_id | model | n_rows)",
        options=list(range(len(df_runs))),
        format_func=lambda i: f"{df_runs.loc[i,'run_id']} | {df_runs.loc[i,'model_name']} | n={int(df_runs.loc[i,'n_rows']):,}",
    )
    embedding_run_id = str(df_runs.loc[pick, "run_id"])
    model_name = str(df_runs.loc[pick, "model_name"])
    total_rows = int(df_runs.loc[pick, "n_rows"])

    st.divider()
    st.header("⚙️ Pengambilan Data")

    limit_rows = st.number_input(
        "Limit rows untuk clustering",
        min_value=500,
        max_value=min(300000, max(500, total_rows)),
        value=min(50000, total_rows),
        step=500,
        help="Mulai dari 20k–80k dulu, lalu naik bertahap sesuai RAM.",
    )

    sampling = st.selectbox("Sampling", options=["random", "first"], index=0)
    random_seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)

    st.divider()
    st.header("⚙️ Parameter HDBSCAN")

    metric_ui = st.selectbox(
        "Metric (UI)",
        options=["cosine", "euclidean"],
        index=0,
        help=(
            "Catatan: sebagian versi HDBSCAN tidak mendukung metric 'cosine' pada BallTree. "
            "Jika memilih cosine, sistem akan L2-normalize embedding dan menjalankan HDBSCAN dengan euclidean (stabil)."
        ),
    )

    min_cluster_size = st.slider("min_cluster_size", 5, 500, 25, 5)
    min_samples = st.slider("min_samples", 1, 100, 10, 1)
    cluster_selection_method = st.selectbox("cluster_selection_method", options=["eom", "leaf"], index=0)
    allow_single_cluster = st.checkbox("allow_single_cluster", value=False)
    gen_probs = st.checkbox("hitung membership probability (score)", value=True)

    st.divider()
    notes = st.text_input("Catatan (opsional)", value="")

    colA, colB = st.columns(2)
    do_preview = colA.button("👀 Preview data", use_container_width=True)
    do_run = colB.button("🚀 Run HDBSCAN", type="primary", use_container_width=True)


# =========================
# Preview
# =========================
if do_preview:
    try:
        df = load_embeddings(engine, embedding_run_id, model_name, int(limit_rows), sampling, int(random_seed))
        if df.empty:
            st.warning("Tidak ada data terambil.")
        else:
            st.subheader("Preview Data (top 50)")
            st.dataframe(df.head(50), use_container_width=True, height=420)
            st.caption(f"Terambil: {len(df):,} baris dari total {total_rows:,} (run_id={embedding_run_id})")
    except Exception as e:
        st.exception(e)


# =========================
# Run HDBSCAN
# =========================
if do_run:
    try:
        if not _HAS_HDBSCAN:
            st.error("Tidak bisa menjalankan HDBSCAN karena package `hdbscan` belum ter-install.")
            st.stop()

        df = load_embeddings(engine, embedding_run_id, model_name, int(limit_rows), sampling, int(random_seed))
        if df.empty:
            st.warning("Tidak ada data untuk clustering.")
            st.stop()

        st.info(f"Menyiapkan matriks embedding untuk {len(df):,} baris…")
        X = to_matrix(df)

        # PATCH: cosine -> normalize + euclidean untuk HDBSCAN (BallTree safe)
        chosen_metric = str(metric_ui)
        if chosen_metric == "cosine":
            X = l2_normalize_rows(X)
            hdb_metric = "euclidean"
            silhouette_metric = "cosine"
        else:
            hdb_metric = "euclidean"  # euclidean native
            silhouette_metric = "euclidean"

        st.info(f"Menjalankan HDBSCAN… (internal metric={hdb_metric}, silhouette metric={silhouette_metric})")
        t0 = time.time()

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=int(min_cluster_size),
            min_samples=int(min_samples),
            metric=hdb_metric,
            cluster_selection_method=str(cluster_selection_method),
            allow_single_cluster=bool(allow_single_cluster),
            prediction_data=bool(gen_probs),
        )

        labels = clusterer.fit_predict(X)
        probs = getattr(clusterer, "probabilities_", None) if gen_probs else None

        elapsed = time.time() - t0

        # summary
        df_stats = summarize_clusters(labels)

        # KPIs
        sil, dbi, n_clusters, n_noise = compute_metrics(X, labels, silhouette_metric=silhouette_metric)
        n_rows = len(df)

        st.success(f"Selesai clustering dalam {elapsed:,.1f}s")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{n_rows:,}")
        c2.metric("Clusters (excl. noise)", f"{n_clusters:,}")
        c3.metric("Noise (-1)", f"{n_noise:,}")
        c4.metric("Noise %", f"{(n_noise / max(1, n_rows) * 100):.2f}%")

        c5, c6 = st.columns(2)
        c5.metric("Silhouette", "-" if sil is None else f"{sil:.4f}")
        c6.metric("DBI", "-" if dbi is None else f"{dbi:.4f}")

        st.subheader("Distribusi Ukuran Cluster")
        show_noise = st.checkbox("Tampilkan noise (-1)", value=False)
        df_plot = df_stats.copy()
        if not show_noise:
            df_plot = df_plot[df_plot["cluster_id"] != -1]

        chart = (
            alt.Chart(df_plot)
            .mark_bar()
            .encode(
                x=alt.X("cluster_id:O", title="cluster_id"),
                y=alt.Y("cluster_size:Q", title="cluster_size"),
                tooltip=["cluster_id", "cluster_size", alt.Tooltip("pct:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart, use_container_width=True)

        with st.expander("Tabel ringkasan cluster", expanded=False):
            st.dataframe(df_stats, use_container_width=True, height=420)

        # members
        df_members = pd.DataFrame(
            {
                "incident_number": df["incident_number"].astype(str).values,
                "cluster_id": labels.astype(int),
                "score": (probs.astype(float) if probs is not None else np.full(len(df), np.nan)),
            }
        )

        # save to DB
        modeling_id = str(uuid.uuid4())
        params_json = {
            "embedding_run_id": embedding_run_id,
            "model_name": model_name,
            "limit_rows": int(limit_rows),
            "sampling": sampling,
            "random_seed": int(random_seed),
            "metric_ui": chosen_metric,
            "internal": {
                "hdbscan_metric": hdb_metric,
                "silhouette_metric": silhouette_metric,
                "l2_normalize_when_cosine": (chosen_metric == "cosine"),
            },
            "hdbscan": {
                "min_cluster_size": int(min_cluster_size),
                "min_samples": int(min_samples),
                "cluster_selection_method": cluster_selection_method,
                "allow_single_cluster": bool(allow_single_cluster),
                "probabilities": bool(gen_probs),
            },
            "runtime_sec": float(elapsed),
        }

        insert_run(
            engine,
            {
                "modeling_id": modeling_id,
                "embedding_run_id": embedding_run_id,
                "model_name": model_name,
                "params_json": params_json,
                "notes": notes or "",
                "n_rows": int(n_rows),
                "n_clusters": int(n_clusters),
                "n_noise": int(n_noise),
                "silhouette": sil,
                "dbi": dbi,
            },
        )

        upsert_clusters(engine, modeling_id, df_stats)
        upsert_members(engine, modeling_id, df_members)

        st.success(f"Hasil tersimpan. modeling_id = {modeling_id}")
        st.caption(f"Tables: `{SCHEMA}.{T_RUNS}`, `{SCHEMA}.{T_CLUST}`, `{SCHEMA}.{T_MEM}`")

        with st.expander("Sampel member (top 50)", expanded=False):
            q = f"""
            SELECT incident_number, cluster_id, score
            FROM {SCHEMA}.{T_MEM}
            WHERE modeling_id = :mid
            ORDER BY cluster_id, incident_number
            LIMIT 50
            """
            df_s = pd.read_sql_query(text(q), engine, params={"mid": modeling_id})
            st.dataframe(df_s, use_container_width=True, height=420)

    except Exception as e:
        st.exception(e)
