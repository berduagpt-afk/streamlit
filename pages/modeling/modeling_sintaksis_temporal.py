# pages/modeling_sintaksis_temporal_clustering.py
# ============================================================
# Temporal Clustering Builder (7, 14, 30 hari)
# Sumber data: lasis_djp.modeling_sintaksis_runs / _members
#
# Konsep:
# - Ambil hasil clustering sintaksis (cluster_id) dari modeling_sintaksis_members untuk 1 modeling_id.
# - EXCLUDE base cluster_id dengan ukuran < 2 (singleton base clusters TIDAK diproses).
# - Untuk setiap cluster_id (yang lolos), urutkan tiket berdasarkan tgl_submit.
# - Pecah menjadi "temporal sub-cluster" bila jarak antar tiket berturut > window_days
#   (gap dihitung presisi pakai Timedelta, bukan dt.days floor).
#
# Output (tabel baru):
# - lasis_djp.modeling_sintaksis_temporal_runs
# - lasis_djp.modeling_sintaksis_temporal_clusters
# - lasis_djp.modeling_sintaksis_temporal_members
#
# Patch utama:
# ✅ Auto CREATE TABLE IF NOT EXISTS + auto ADD COLUMN (ALTER) + indexes
# ✅ Fix SQLAlchemy ":params_json::jsonb" -> CAST(:params_json AS jsonb)
# ✅ params_json disimpan sebagai JSON string (aman untuk psycopg2)
# ✅ Idempotent delete-then-insert per (base_modeling_id, window_days)
# ✅ LAZY LOAD: members baru di-load saat tombol diklik (lebih ringan)
# ✅ Tambah metrik rows_without_time dan n_base_clusters_ge2 (cluster_id size>=2)
# ✅ Exclude cluster_id ukuran < 2 sebelum proses temporal
# ============================================================

from __future__ import annotations

import json
import uuid
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


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

# Output tables (temporal)
T_T_RUNS = "modeling_sintaksis_temporal_runs"
T_T_CLUSTERS = "modeling_sintaksis_temporal_clusters"
T_T_MEMBERS = "modeling_sintaksis_temporal_members"


# ======================================================
# DB
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


engine = get_engine()


# ======================================================
# Bootstrap (CREATE + ALTER + Index)
# ======================================================
def _existing_columns(_engine: Engine, schema: str, table: str) -> set[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table
    """
    df = pd.read_sql(text(q), _engine, params={"schema": schema, "table": table})
    return set(df["column_name"].astype(str).tolist()) if not df.empty else set()


def _ensure_columns(_engine: Engine, schema: str, table: str, desired_cols: dict[str, str]) -> None:
    existing = _existing_columns(_engine, schema, table)
    if not existing:
        return
    alters = []
    for col, coldef in desired_cols.items():
        if col not in existing:
            alters.append(f"ALTER TABLE {schema}.{table} ADD COLUMN {col} {coldef};")
    if alters:
        with _engine.begin() as conn:
            for stmt in alters:
                conn.execute(text(stmt))


def ensure_temporal_tables(_engine: Engine) -> None:
    ddl_create = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_T_RUNS}
    (
        temporal_job_id uuid NOT NULL,
        run_time timestamp with time zone NOT NULL,
        base_job_id uuid,
        base_modeling_id uuid NOT NULL,
        threshold double precision,
        knn_k integer,
        window_days integer NOT NULL,

        -- counts
        n_rows bigint,
        n_base_clusters_ge2 bigint,
        rows_without_time bigint,
        n_temporal_clusters bigint,
        n_singletons_temporal bigint,

        -- averages
        avg_temporal_cluster_size double precision,

        notes text,
        params_json jsonb,

        CONSTRAINT {T_T_RUNS}_pkey PRIMARY KEY (temporal_job_id, base_modeling_id, window_days)
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_T_MEMBERS}
    (
        temporal_job_id uuid NOT NULL,
        base_modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        temporal_cluster_id bigint NOT NULL,
        cluster_id bigint,
        incident_number text NOT NULL,
        tgl_submit timestamp without time zone,
        site text,
        assignee text,
        modul text,
        sub_modul text,
        CONSTRAINT {T_T_MEMBERS}_pkey PRIMARY KEY (base_modeling_id, window_days, incident_number)
    );

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_T_CLUSTERS}
    (
        temporal_job_id uuid NOT NULL,
        base_modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        temporal_cluster_id bigint NOT NULL,
        temporal_cluster_size integer NOT NULL,
        n_semantic_cluster_ids integer,
        min_time timestamp without time zone,
        max_time timestamp without time zone,
        span_days integer,
        CONSTRAINT {T_T_CLUSTERS}_pkey PRIMARY KEY (base_modeling_id, window_days, temporal_cluster_id)
    );
    """

    ddl_indexes = f"""
    CREATE INDEX IF NOT EXISTS idx_{T_T_RUNS}_base_model_win_time
      ON {SCHEMA}.{T_T_RUNS} (base_modeling_id, window_days, run_time DESC);

    CREATE INDEX IF NOT EXISTS idx_{T_T_MEMBERS}_base_model_win_cluster
      ON {SCHEMA}.{T_T_MEMBERS} (base_modeling_id, window_days, temporal_cluster_id);

    CREATE INDEX IF NOT EXISTS idx_{T_T_CLUSTERS}_base_model_win_size
      ON {SCHEMA}.{T_T_CLUSTERS} (base_modeling_id, window_days, temporal_cluster_size DESC);

    CREATE INDEX IF NOT EXISTS idx_{T_T_MEMBERS}_base_model_win
      ON {SCHEMA}.{T_T_MEMBERS} (base_modeling_id, window_days);

    CREATE INDEX IF NOT EXISTS idx_{T_T_CLUSTERS}_base_model_win
      ON {SCHEMA}.{T_T_CLUSTERS} (base_modeling_id, window_days);
    """

    with _engine.begin() as conn:
        conn.execute(text(ddl_create))
        conn.execute(text(ddl_indexes))

    desired_runs = {
        "temporal_job_id": "uuid NOT NULL",
        "run_time": "timestamp with time zone NOT NULL",
        "base_job_id": "uuid",
        "base_modeling_id": "uuid NOT NULL",
        "threshold": "double precision",
        "knn_k": "integer",
        "window_days": "integer NOT NULL",
        "n_rows": "bigint",
        "n_base_clusters_ge2": "bigint",
        "rows_without_time": "bigint",
        "n_temporal_clusters": "bigint",
        "n_singletons_temporal": "bigint",
        "avg_temporal_cluster_size": "double precision",
        "notes": "text",
        "params_json": "jsonb",
    }
    desired_members = {
        "temporal_job_id": "uuid NOT NULL",
        "base_modeling_id": "uuid NOT NULL",
        "window_days": "integer NOT NULL",
        "temporal_cluster_id": "bigint NOT NULL",
        "cluster_id": "bigint",
        "incident_number": "text NOT NULL",
        "tgl_submit": "timestamp without time zone",
        "site": "text",
        "assignee": "text",
        "modul": "text",
        "sub_modul": "text",
    }
    desired_clusters = {
        "temporal_job_id": "uuid NOT NULL",
        "base_modeling_id": "uuid NOT NULL",
        "window_days": "integer NOT NULL",
        "temporal_cluster_id": "bigint NOT NULL",
        "temporal_cluster_size": "integer NOT NULL",
        "n_semantic_cluster_ids": "integer",
        "min_time": "timestamp without time zone",
        "max_time": "timestamp without time zone",
        "span_days": "integer",
    }

    _ensure_columns(_engine, SCHEMA, T_T_RUNS, desired_runs)
    _ensure_columns(_engine, SCHEMA, T_T_MEMBERS, desired_members)
    _ensure_columns(_engine, SCHEMA, T_T_CLUSTERS, desired_clusters)


# Bootstrap sekali; kalau gagal, tampilkan error UI (tidak crash silent)
try:
    ensure_temporal_tables(engine)
except Exception as e:
    st.error(f"Gagal memastikan tabel temporal_* di DB: {e}")
    st.stop()


# ======================================================
# Loaders
# ======================================================
@st.cache_data(show_spinner=False, ttl=120)
def load_runs(_engine: Engine) -> pd.DataFrame:
    cols = _existing_columns(_engine, SCHEMA, T_RUNS)
    wanted = [
        "job_id",
        "modeling_id",
        "run_time",
        "threshold",
        "knn_k",
        "n_rows",
        "n_clusters_all",
        "n_singletons",
        "n_clusters_multi",
    ]
    selected = [c for c in wanted if c in cols]
    if not selected:
        return pd.DataFrame()

    q = f"""
    SELECT {", ".join(selected)}
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    """
    df = pd.read_sql(text(q), _engine)
    if df.empty:
        return df

    df["run_time"] = pd.to_datetime(df.get("run_time"), errors="coerce")
    for c in ["job_id", "modeling_id"]:
        if c in df.columns:
            df[c] = df[c].astype("string")
    if "threshold" in df.columns:
        df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    if "knn_k" in df.columns:
        df["knn_k"] = pd.to_numeric(df["knn_k"], errors="coerce").astype("Int64")

    for c in ["n_rows", "n_clusters_all", "n_singletons", "n_clusters_multi"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    for c in wanted:
        if c not in df.columns:
            df[c] = pd.NA

    return df


@st.cache_data(show_spinner=False, ttl=120)
def load_members(_engine: Engine, modeling_id: str, limit_rows: int | None = None) -> pd.DataFrame:
    lim = f"LIMIT {int(limit_rows)}" if limit_rows else ""
    q = f"""
    SELECT job_id, modeling_id, cluster_id, incident_number, tgl_submit, site, assignee, modul, sub_modul
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE modeling_id = CAST(:mid AS uuid)
    ORDER BY cluster_id ASC, tgl_submit ASC NULLS LAST
    {lim}
    """
    df = pd.read_sql(text(q), _engine, params={"mid": modeling_id})
    if df.empty:
        return df

    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["cluster_id"] = pd.to_numeric(df["cluster_id"], errors="coerce").astype("Int64")
    df["job_id"] = df["job_id"].astype("string")
    df["modeling_id"] = df["modeling_id"].astype("string")
    return df


# ======================================================
# Filtering: exclude base cluster_id size < 2
# ======================================================
def exclude_small_base_clusters(members: pd.DataFrame, min_size: int = 2) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Buang cluster_id dengan ukuran < min_size.
    Return: (members_filtered, stats)
    """
    if members.empty:
        return members.copy(), {"n_rows_in": 0, "n_rows_out": 0, "n_clusters_in": 0, "n_clusters_out": 0}

    df = members.copy()
    # cluster_id NA dianggap tidak valid (dibuat terbuang)
    df_valid = df[df["cluster_id"].notna()].copy()

    size_by = df_valid.groupby("cluster_id", as_index=False).size().rename(columns={"size": "cluster_size"})
    keep_ids = set(size_by[size_by["cluster_size"] >= int(min_size)]["cluster_id"].astype("Int64").tolist())

    out = df_valid[df_valid["cluster_id"].isin(list(keep_ids))].copy()

    stats = {
        "n_rows_in": int(len(df)),
        "n_rows_out": int(len(out)),
        "n_clusters_in": int(df_valid["cluster_id"].nunique()),
        "n_clusters_out": int(out["cluster_id"].nunique()),
    }
    return out, stats


# ======================================================
# Temporal splitting logic (presisi Timedelta)
# ======================================================
def build_temporal_clusters(members: pd.DataFrame, window_days: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pecah tiap base cluster_id menjadi subcluster berdasarkan gap waktu.
    Rule: jika (tgl_submit - prev_time) > Timedelta(days=window_days) => mulai temporal cluster baru.

    NOTE: Base cluster_id sudah difilter size>=2 di upstream.
    """
    if members.empty:
        return members.copy(), pd.DataFrame()

    df = members.copy()
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")

    has_time = df["tgl_submit"].notna()
    df_time = df[has_time].copy()
    df_na = df[~has_time].copy()

    df_time = df_time.sort_values(["cluster_id", "tgl_submit", "incident_number"], ascending=[True, True, True])

    df_time["prev_time"] = df_time.groupby("cluster_id")["tgl_submit"].shift(1)
    delta = (df_time["tgl_submit"] - df_time["prev_time"])

    df_time["new_seg"] = df_time["prev_time"].isna() | (delta > pd.Timedelta(days=int(window_days)))
    df_time["seg_in_cluster"] = df_time.groupby("cluster_id")["new_seg"].cumsum().astype("int64")

    # surrogate ID (non-deterministic if input order changes)
    keys = list(zip(df_time["cluster_id"].astype("Int64").astype(str), df_time["seg_in_cluster"].astype(str)))
    df_time["temporal_cluster_id"] = pd.factorize(keys)[0].astype("int64")

    # rows tanpa waktu: tetap ikut (opsional), dibuat ID unik per row
    if not df_na.empty:
        df_na = df_na.sort_values(["cluster_id", "incident_number"])
        max_id = int(df_time["temporal_cluster_id"].max()) if not df_time.empty else -1
        df_na["temporal_cluster_id"] = (max_id + 1 + np.arange(len(df_na))).astype("int64")

    out = pd.concat(
        [df_time.drop(columns=["prev_time", "new_seg", "seg_in_cluster"], errors="ignore"), df_na],
        ignore_index=True,
    )
    out["temporal_cluster_id"] = pd.to_numeric(out["temporal_cluster_id"], errors="coerce").astype("Int64")

    clusters = (
        out.groupby("temporal_cluster_id", as_index=False)
        .agg(
            temporal_cluster_size=("incident_number", "count"),
            n_semantic_cluster_ids=("cluster_id", lambda s: int(pd.Series(s).dropna().nunique())),
            min_time=("tgl_submit", "min"),
            max_time=("tgl_submit", "max"),
        )
    )
    if not clusters.empty:
        clusters["span_days"] = (clusters["max_time"] - clusters["min_time"]).dt.days
        clusters["temporal_cluster_id"] = pd.to_numeric(clusters["temporal_cluster_id"], errors="coerce").astype("Int64")
        clusters["temporal_cluster_size"] = pd.to_numeric(clusters["temporal_cluster_size"], errors="coerce").astype("Int64")

    return out, clusters


def compute_temporal_metrics(
    members_out: pd.DataFrame,
    clusters_out: pd.DataFrame,
    n_base_clusters_ge2: int,
    rows_without_time: int,
) -> Dict[str, float | int]:
    n_rows = int(len(members_out))
    n_temporal_clusters = int(clusters_out["temporal_cluster_id"].nunique()) if not clusters_out.empty else 0
    n_singletons_temporal = int(
        (pd.to_numeric(clusters_out["temporal_cluster_size"], errors="coerce") == 1).sum()
    ) if not clusters_out.empty else 0

    avg_size = float(n_rows / n_temporal_clusters) if n_temporal_clusters > 0 else 0.0
    return {
        "n_rows": n_rows,
        "n_base_clusters_ge2": int(n_base_clusters_ge2),
        "rows_without_time": int(rows_without_time),
        "n_temporal_clusters": n_temporal_clusters,
        "n_singletons_temporal": n_singletons_temporal,
        "avg_temporal_cluster_size": avg_size,
    }


def save_temporal_results(
    _engine: Engine,
    base_run_row: pd.Series,
    window_days: int,
    temporal_job_id: str,
    members_out: pd.DataFrame,
    clusters_out: pd.DataFrame,
    base_filter_stats: Dict[str, int],
) -> None:
    ensure_temporal_tables(_engine)

    base_modeling_id = str(base_run_row["modeling_id"])
    base_job_id = str(base_run_row["job_id"]) if pd.notna(base_run_row.get("job_id")) else None
    threshold = float(base_run_row["threshold"]) if pd.notna(base_run_row.get("threshold")) else None
    knn_k = int(base_run_row["knn_k"]) if pd.notna(base_run_row.get("knn_k")) else None

    rows_without_time = int(members_out["tgl_submit"].isna().sum()) if "tgl_submit" in members_out.columns else 0
    metrics = compute_temporal_metrics(
        members_out=members_out,
        clusters_out=clusters_out,
        n_base_clusters_ge2=base_filter_stats.get("n_clusters_out", 0),
        rows_without_time=rows_without_time,
    )

    params_obj = {
        "rule": "new temporal segment if (tgl_submit - prev_time) > Timedelta(days=window_days)",
        "window_days": int(window_days),
        "base_modeling_id": base_modeling_id,
        "excluded_base_clusters_lt_2": True,
        "base_filter_stats": base_filter_stats,
        "note_temporal_cluster_id": "surrogate via pandas.factorize(keys), not deterministic if input order changes",
    }
    params_json_str = json.dumps(params_obj, ensure_ascii=False)

    mem = members_out.copy()
    mem.insert(0, "temporal_job_id", temporal_job_id)
    mem.insert(1, "base_modeling_id", base_modeling_id)
    mem.insert(2, "window_days", int(window_days))
    mem = mem[[
        "temporal_job_id",
        "base_modeling_id",
        "window_days",
        "temporal_cluster_id",
        "cluster_id",
        "incident_number",
        "tgl_submit",
        "site",
        "assignee",
        "modul",
        "sub_modul",
    ]].copy()

    clu = clusters_out.copy()
    clu.insert(0, "temporal_job_id", temporal_job_id)
    clu.insert(1, "base_modeling_id", base_modeling_id)
    clu.insert(2, "window_days", int(window_days))
    clu = clu[[
        "temporal_job_id",
        "base_modeling_id",
        "window_days",
        "temporal_cluster_id",
        "temporal_cluster_size",
        "n_semantic_cluster_ids",
        "min_time",
        "max_time",
        "span_days",
    ]].copy()

    with _engine.begin() as conn:
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.{T_T_MEMBERS} WHERE base_modeling_id = CAST(:mid AS uuid) AND window_days = :w"),
            {"mid": base_modeling_id, "w": int(window_days)},
        )
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.{T_T_CLUSTERS} WHERE base_modeling_id = CAST(:mid AS uuid) AND window_days = :w"),
            {"mid": base_modeling_id, "w": int(window_days)},
        )
        conn.execute(
            text(f"DELETE FROM {SCHEMA}.{T_T_RUNS} WHERE base_modeling_id = CAST(:mid AS uuid) AND window_days = :w"),
            {"mid": base_modeling_id, "w": int(window_days)},
        )

        conn.execute(
            text(f"""
                INSERT INTO {SCHEMA}.{T_T_RUNS}
                (
                    temporal_job_id, run_time, base_job_id, base_modeling_id,
                    threshold, knn_k, window_days,
                    n_rows, n_base_clusters_ge2, rows_without_time,
                    n_temporal_clusters, n_singletons_temporal,
                    avg_temporal_cluster_size, notes, params_json
                )
                VALUES
                (
                    CAST(:temporal_job_id AS uuid),
                    NOW(),
                    {("CAST(:base_job_id AS uuid)" if base_job_id else "NULL")},
                    CAST(:base_modeling_id AS uuid),
                    :threshold, :knn_k, :window_days,
                    :n_rows, :n_base_clusters_ge2, :rows_without_time,
                    :n_temporal_clusters, :n_singletons_temporal,
                    :avg_temporal_cluster_size,
                    :notes,
                    CAST(:params_json AS jsonb)
                )
            """),
            {
                "temporal_job_id": temporal_job_id,
                "base_job_id": base_job_id,
                "base_modeling_id": base_modeling_id,
                "threshold": threshold,
                "knn_k": knn_k,
                "window_days": int(window_days),
                "n_rows": int(metrics["n_rows"]),
                "n_base_clusters_ge2": int(metrics["n_base_clusters_ge2"]),
                "rows_without_time": int(metrics["rows_without_time"]),
                "n_temporal_clusters": int(metrics["n_temporal_clusters"]),
                "n_singletons_temporal": int(metrics["n_singletons_temporal"]),
                "avg_temporal_cluster_size": float(metrics["avg_temporal_cluster_size"]),
                "notes": "Temporal sub-clustering by splitting base clusters (size>=2) using time gaps > window_days.",
                "params_json": params_json_str,
            },
        )

        mem.to_sql(
            name=T_T_MEMBERS,
            con=conn,
            schema=SCHEMA,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=5000,
        )
        if not clu.empty:
            clu.to_sql(
                name=T_T_CLUSTERS,
                con=conn,
                schema=SCHEMA,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5000,
            )


# ======================================================
# UI
# ======================================================
st.title("🕒 Temporal Clustering Builder (7/14/30 hari)")
st.caption(
    "Membentuk **temporal sub-cluster** dari hasil cluster sintaksis (cluster_id) "
    "dengan memecahnya berdasarkan **gap waktu** (tgl_submit). "
    "**Catatan**: base cluster_id berukuran < 2 (singleton) **di-exclude** dari proses."
)

runs = load_runs(engine)
if runs.empty:
    st.warning(f"Belum ada data runs di {SCHEMA}.{T_RUNS}. Jalankan offline modeling dulu.")
    st.stop()

st.sidebar.header("⚙️ Parameter")

job_keys = runs["job_id"].fillna("NO_JOB").astype(str).unique().tolist()
job_keys = sorted(job_keys, reverse=True)

sel_job = st.sidebar.selectbox("Pilih job_id", options=job_keys, index=0)
runs_job = runs[runs["job_id"].fillna("NO_JOB").astype(str) == sel_job].copy().sort_values("run_time", ascending=False)

run_labels = runs_job.apply(
    lambda r: f"{(r['run_time'].strftime('%Y-%m-%d %H:%M') if pd.notna(r['run_time']) else 'NA')} "
              f"| thr={r.get('threshold')} k={r.get('knn_k')} "
              f"| clusters={r.get('n_clusters_all')} singletons={r.get('n_singletons')} "
              f"| {r.get('modeling_id')}",
    axis=1
).tolist()
run_map = dict(zip(run_labels, runs_job["modeling_id"].astype(str).tolist()))

sel_run_label = st.sidebar.selectbox("Pilih base modeling_id", options=run_labels, index=0)
base_modeling_id = run_map[sel_run_label]
base_row = runs_job.loc[runs_job["modeling_id"].astype(str) == base_modeling_id].iloc[0]

windows = st.sidebar.multiselect("Window days", options=[7, 14, 30], default=[7, 14, 30])

limit_rows = st.sidebar.number_input("Limit rows (opsional, untuk test cepat)", min_value=0, value=0, step=1000)
limit_rows = int(limit_rows) if int(limit_rows) > 0 else None

st.sidebar.divider()
do_save = st.sidebar.checkbox("Simpan hasil ke DB", value=True)
btn = st.sidebar.button("🚀 Generate Temporal Clusters", type="primary")

# KPI base run
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Base modeling_id", base_modeling_id)
c2.metric("Threshold", f"{base_row.get('threshold')}")
c3.metric("kNN-k", f"{base_row.get('knn_k')}")
c4.metric("n_clusters_all", f"{int(base_row.get('n_clusters_all') or 0):,}")
c5.metric("n_singletons", f"{int(base_row.get('n_singletons') or 0):,}")

st.divider()

if btn:
    if not windows:
        st.warning("Pilih minimal 1 window_days.")
        st.stop()

    with st.spinner("Memuat members dari DB..."):
        members_raw = load_members(engine, base_modeling_id, limit_rows=limit_rows)

    if members_raw.empty:
        st.error("Members kosong untuk modeling_id ini. Cek tabel modeling_sintaksis_members.")
        st.stop()

    # EXCLUDE base cluster size < 2
    members_f, fstats = exclude_small_base_clusters(members_raw, min_size=2)

    st.write(f"Total members loaded: **{len(members_raw):,}** (limit={limit_rows or 'ALL'})")
    st.info(
        f"Exclude base cluster_id size<2 → rows: {fstats['n_rows_in']:,} → {fstats['n_rows_out']:,} | "
        f"clusters: {fstats['n_clusters_in']:,} → {fstats['n_clusters_out']:,}"
    )

    if members_f.empty:
        st.warning("Setelah exclude singleton base clusters, data kosong. Tidak ada yang diproses.")
        st.stop()

    temporal_job_id = str(uuid.uuid4())
    st.success(f"Mulai generate temporal clusters. temporal_job_id = {temporal_job_id}")

    results: Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Dict]] = {}

    for w in windows:
        with st.spinner(f"Memproses window_days={w}..."):
            mem_out, clu_out = build_temporal_clusters(members_f, window_days=int(w))

            rows_without_time = int(mem_out["tgl_submit"].isna().sum()) if "tgl_submit" in mem_out.columns else 0
            met = compute_temporal_metrics(
                members_out=mem_out,
                clusters_out=clu_out,
                n_base_clusters_ge2=fstats.get("n_clusters_out", 0),
                rows_without_time=rows_without_time,
            )
            results[int(w)] = (mem_out, clu_out, met)

            if do_save:
                save_temporal_results(
                    engine,
                    base_run_row=base_row,
                    window_days=int(w),
                    temporal_job_id=temporal_job_id,
                    members_out=mem_out,
                    clusters_out=clu_out,
                    base_filter_stats=fstats,
                )

    st.success("Selesai. Hasil siap dilihat di bawah.")

    rows = []
    for w, (_, _, met) in results.items():
        rows.append({
            "window_days": w,
            "n_rows_processed": met["n_rows"],
            "n_base_clusters_ge2": met["n_base_clusters_ge2"],
            "rows_without_time": met["rows_without_time"],
            "n_temporal_clusters": met["n_temporal_clusters"],
            "n_singletons_temporal": met["n_singletons_temporal"],
            "avg_temporal_cluster_size": round(float(met["avg_temporal_cluster_size"]), 3),
        })
    summ = pd.DataFrame(rows).sort_values("window_days")

    st.subheader("Ringkasan Hasil Temporal Clustering")
    st.dataframe(summ, use_container_width=True)

    st.markdown("### Grafik Perbandingan (berdasarkan window_days)")
    plot = summ.copy()
    plot["window_days"] = pd.to_numeric(plot["window_days"], errors="coerce")

    ttip = [
        alt.Tooltip("window_days:Q"),
        alt.Tooltip("n_temporal_clusters:Q"),
        alt.Tooltip("n_singletons_temporal:Q"),
        alt.Tooltip("avg_temporal_cluster_size:Q"),
        alt.Tooltip("rows_without_time:Q"),
    ]

    g1, g2 = st.columns(2)
    with g1:
        st.altair_chart(
            alt.Chart(plot, title="Window vs #Temporal Clusters")
            .mark_line(point=True)
            .encode(
                x=alt.X("window_days:Q", title="window_days"),
                y=alt.Y("n_temporal_clusters:Q", title="n_temporal_clusters"),
                tooltip=ttip,
            )
            .properties(height=260),
            use_container_width=True,
        )
    with g2:
        st.altair_chart(
            alt.Chart(plot, title="Window vs Avg Temporal Cluster Size")
            .mark_line(point=True)
            .encode(
                x=alt.X("window_days:Q", title="window_days"),
                y=alt.Y("avg_temporal_cluster_size:Q", title="avg_temporal_cluster_size"),
                tooltip=ttip,
            )
            .properties(height=260),
            use_container_width=True,
        )

    st.divider()
    st.subheader("Detail per Window")
    tabs = st.tabs([f"{w} hari" for w in sorted(results.keys())])
    for tab, w in zip(tabs, sorted(results.keys())):
        mem_out, clu_out, met = results[w]
        with tab:
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("n_temporal_clusters", f"{int(met['n_temporal_clusters']):,}")
            c2.metric("n_singletons_temporal", f"{int(met['n_singletons_temporal']):,}")
            c3.metric("avg_temporal_cluster_size", f"{float(met['avg_temporal_cluster_size']):.3f}")
            c4.metric("rows_without_time", f"{int(met['rows_without_time']):,}")
            c5.metric("n_base_clusters_ge2", f"{int(met['n_base_clusters_ge2']):,}")

            if clu_out.empty:
                st.info("Cluster temporal kosong.")
                continue

            st.markdown("**Top temporal clusters (by size)**")
            st.dataframe(
                clu_out.sort_values("temporal_cluster_size", ascending=False).head(30),
                use_container_width=True,
            )

            st.markdown("**Distribusi ukuran temporal cluster**")
            st.altair_chart(
                alt.Chart(clu_out)
                .mark_bar()
                .encode(
                    x=alt.X("temporal_cluster_size:Q", bin=alt.Bin(maxbins=40), title="temporal_cluster_size"),
                    y=alt.Y("count():Q", title="jumlah temporal cluster"),
                    tooltip=[alt.Tooltip("count():Q", title="n_temporal_clusters")],
                )
                .properties(height=240),
                use_container_width=True,
            )

            pick = st.selectbox(
                "Lihat members untuk temporal_cluster_id",
                options=clu_out.sort_values("temporal_cluster_size", ascending=False)["temporal_cluster_id"].dropna().astype(int).tolist()[:50],
                index=0,
                key=f"pick_{w}",
            )
            view_mem = mem_out[mem_out["temporal_cluster_id"].astype("Int64") == int(pick)].copy()
            st.write(f"Members: **{len(view_mem):,}**")
            st.dataframe(view_mem.sort_values("tgl_submit", na_position="last").head(300), use_container_width=True)

else:
    st.info("Pilih job_id & base modeling_id, lalu klik **Generate Temporal Clusters**.")

with st.expander("ℹ️ Catatan Metodologis"):
    st.write(
        "- Temporal clustering di sini adalah **pemecahan** cluster sintaksis menjadi subcluster berdasar jarak waktu.\n"
        "- Aturan: jika (tgl_submit - prev_time) > window_days (presisi Timedelta), maka dibuat subcluster baru.\n"
        "- **Base cluster_id berukuran < 2 (singleton)** di-exclude dari proses agar fokus pada pola berulang.\n"
        "- Output tersimpan di tabel temporal_* agar bisa dianalisis di viewer temporal."
    )
