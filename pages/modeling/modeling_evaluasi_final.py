# pages/evaluation/compare_sintaksis_vs_semantik_v2.py
from __future__ import annotations

import json
from typing import Any, Dict, Optional, List

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"
TABLE = "modeling_evaluation_results"
T_SELECTED = "modeling_selected_representatives"

UUID_COLS = ["eval_id", "job_id", "modeling_id", "embedding_run_id"]
TEXT_COLS = ["jenis_pendekatan", "temporal_id", "notes"]
NUM_COLS = ["silhouette_score", "dbi", "threshold", "dbcv"]


# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


# ======================================================
# Helpers
# ======================================================
def _safe_json_loads(x) -> Optional[dict]:
    if x is None:
        return None
    # pandas NA/NaN
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s or s.lower() in {"na", "nan", "none", "null"}:
            return None
        try:
            return json.loads(s)
        except Exception:
            return None
    return None


def _pick(d: Optional[dict], keys: List[str]) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    out: Dict[str, Any] = {}
    for k in keys:
        if k in d and d[k] is not None:
            out[k] = d[k]
    return out


def _clean_scalar(v: Any) -> Any:
    """
    Convert pandas NA/NaN, numpy scalars, and placeholder strings ('NA', 'nan', '') into DB-safe python values.
    """
    # pd.NA / NaN
    if v is pd.NA:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # Strings -> normalize placeholders
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in {"na", "nan", "none", "null"}:
            return None
        return s

    # numpy scalar -> python scalar
    try:
        import numpy as np
        if isinstance(v, np.generic):
            v = v.item()
    except Exception:
        pass

    # Anything else is ok
    return v


def _clean_uuid_str(v: Any) -> Optional[str]:
    """
    Ensure UUID-like values are stored as str or None (psycopg2 can handle str for uuid columns via CAST).
    """
    v = _clean_scalar(v)
    if v is None:
        return None
    # already string
    if isinstance(v, str):
        return v.strip() or None
    return str(v)


def sanitize_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep-clean record values; ensures no pd.NA / numpy types reach psycopg2.
    """
    out: Dict[str, Any] = {}
    for k, v in rec.items():
        if k in {"eval_id", "job_id", "modeling_id", "embedding_run_id"}:
            out[k] = _clean_uuid_str(v)
        elif k in {"threshold"}:
            vv = _clean_scalar(v)
            out[k] = float(vv) if vv is not None else None
        else:
            out[k] = _clean_scalar(v)
    return out


def load_from_db() -> pd.DataFrame:
    eng = get_engine()
    q = text(
        f"""
        SELECT
            eval_id,
            run_time,
            jenis_pendekatan,
            job_id,
            embedding_run_id,
            modeling_id,
            temporal_id,            -- TEXT
            silhouette_score,
            dbi,
            threshold,
            --notes,
            meta_json,              -- JSONB
            dbcv
        FROM {SCHEMA}.{TABLE}
        ORDER BY run_time DESC
        """
    )
    with eng.connect() as conn:
        return pd.read_sql(q, conn)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "run_time" in df.columns:
        df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")

    # Cast UUID-like to string for UI/filtering (keep NA as <NA> for now; we will sanitize on save)
    for c in UUID_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")

    for c in TEXT_COLS:
        if c in df.columns:
            df[c] = df[c].astype("string")

    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # normalize jenis_pendekatan (robust)
    if "jenis_pendekatan" in df.columns:
        jp = df["jenis_pendekatan"].astype("string").str.lower().str.strip()
        mapping = {
            "syntactic": "sintaksis",
            "syntax": "sintaksis",
            "sintaks": "sintaksis",
            "sintaksis": "sintaksis",
            "semantic": "semantik",
            "semantics": "semantik",
            "semmantik": "semantik",
            "semantik": "semantik",
        }
        jp = jp.replace(mapping)
        jp = jp.mask(jp.str.contains("semantik|semantic", na=False), "semantik")
        jp = jp.mask(jp.str.contains("sintaksis|syntactic|syntax|sintaks", na=False), "sintaksis")
        df["jenis_pendekatan"] = jp

    # json
    if "meta_json" in df.columns:
        df["meta_obj"] = df["meta_json"].apply(_safe_json_loads)
    else:
        df["meta_obj"] = None

    return df


def recommend_params(row: pd.Series, pendekatan: str) -> Dict[str, Any]:
    meta = row.get("meta_obj")
    if not isinstance(meta, dict):
        return {}

    if pendekatan == "sintaksis":
        keys = [
            "k", "k_neighbors", "knn_k", "top_k",
            "ngram_range", "max_features", "min_df", "max_df",
            "vectorizer", "tfidf",
            "cosine_threshold", "threshold",
            "window_days", "include_noise", "eligible_rule", "time_col",
        ]
    else:  # semantik
        keys = [
            "embedding_model", "model_name",
            "min_cluster_size", "min_samples", "metric",
            "cluster_selection_method", "cluster_selection_epsilon",
            "window_days", "include_noise", "eligible_rule", "time_col",
        ]

    out = _pick(meta, keys)
    for nested_key in ["params_json", "params", "hdbscan_params", "tfidf_params"]:
        nested = meta.get(nested_key)
        if isinstance(nested, dict):
            out.update(_pick(nested, keys))
    return out


def best_by_keys(data: pd.DataFrame, group_keys: List[str], metric_mode: str) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()

    tmp = data.copy()
    if metric_mode == "dbcv_dbi":
        tmp = tmp.sort_values(["dbcv", "dbi", "run_time"], ascending=[False, True, False], na_position="last")
    else:
        tmp = tmp.sort_values(["silhouette_score", "dbi", "run_time"], ascending=[False, True, False], na_position="last")

    return tmp.groupby(group_keys, dropna=False, as_index=False).head(1).copy()


def best_of_best(df_in: pd.DataFrame, metric_mode: str) -> pd.DataFrame:
    if df_in.empty:
        return pd.DataFrame()
    tmp = df_in.copy()
    if metric_mode == "dbcv_dbi":
        tmp = tmp.sort_values(["dbcv", "dbi", "run_time"], ascending=[False, True, False], na_position="last")
    else:
        tmp = tmp.sort_values(["silhouette_score", "dbi", "run_time"], ascending=[False, True, False], na_position="last")
    return tmp.head(1).copy()


def _opts(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    return sorted([x for x in df[col].dropna().unique().tolist()])


def save_selected_representatives(engine, records: List[Dict[str, Any]]) -> int:
    """
    Upsert ke lasis_djp.modeling_selected_representatives
    Unique: (jenis_pendekatan, eval_id)
    """
    if not records:
        return 0

    sql = text(
        f"""
        INSERT INTO {SCHEMA}.{T_SELECTED}
        (jenis_pendekatan, eval_id, job_id, modeling_id, embedding_run_id, temporal_id, threshold, source_page, notes)
        VALUES
        (:jenis_pendekatan, CAST(:eval_id AS uuid), CAST(:job_id AS uuid), CAST(:modeling_id AS uuid), CAST(:embedding_run_id AS uuid),
         :temporal_id, :threshold, :source_page, :notes)
        ON CONFLICT (jenis_pendekatan, eval_id)
        DO UPDATE SET
            selected_at = now(),
            job_id = EXCLUDED.job_id,
            modeling_id = EXCLUDED.modeling_id,
            embedding_run_id = EXCLUDED.embedding_run_id,
            temporal_id = EXCLUDED.temporal_id,
            threshold = EXCLUDED.threshold,
            source_page = EXCLUDED.source_page,
            notes = EXCLUDED.notes
        """
    )

    with engine.begin() as conn:
        conn.execute(sql, records)

    return len(records)


def _row_to_record(row: pd.Series) -> Dict[str, Any]:
    jp = (row.get("jenis_pendekatan") or "").strip()

    # semantik tidak pakai threshold (hindari -1.0 placeholder)
    thr = row.get("threshold")
    if jp == "semantik":
        thr = None

    rec = {
        "jenis_pendekatan": jp,
        "eval_id": row.get("eval_id"),
        "job_id": row.get("job_id"),
        "modeling_id": row.get("modeling_id"),
        "embedding_run_id": row.get("embedding_run_id"),
        "temporal_id": row.get("temporal_id"),
        "threshold": thr,
        "source_page": "compare_sintaksis_vs_semantik_v2",
        "notes": row.get("notes"),
    }
    return sanitize_record(rec)


# ======================================================
# UI
# ======================================================
st.title("🔎 Perbandingan Hasil Evaluasi Pendekatan Sintaksis dan Semantik")
st.caption("Sumber: lasis_djp.modeling_evaluation_results. Semantik filter utama: embedding_run_id.")

df = prep(load_from_db())

df_syn = df[df["jenis_pendekatan"] == "sintaksis"].copy()
df_sem = df[df["jenis_pendekatan"] == "semantik"].copy()

with st.sidebar:
    st.subheader("Filter Sintaksis")
    syn_job = st.multiselect("job_id", _opts(df_syn, "job_id"), default=_opts(df_syn, "job_id")[:1] if _opts(df_syn, "job_id") else [])
    syn_model = st.multiselect("modeling_id", _opts(df_syn, "modeling_id"), default=[])
    syn_temp = st.multiselect("temporal_id (text)", _opts(df_syn, "temporal_id"), default=[])

    syn_thr_min, syn_thr_max = None, None
    if "threshold" in df_syn.columns and df_syn["threshold"].notna().any():
        tmin = float(df_syn["threshold"].min())
        tmax = float(df_syn["threshold"].max())
        syn_thr_min, syn_thr_max = st.slider("threshold (sintaksis)", min_value=tmin, max_value=tmax, value=(tmin, tmax))

    st.divider()
    st.subheader("Filter Semantik (utama: embedding_run_id)")
    sem_emb = st.multiselect(
        "embedding_run_id (semantik)",
        _opts(df_sem, "embedding_run_id"),
        default=_opts(df_sem, "embedding_run_id")[:1] if _opts(df_sem, "embedding_run_id") else []
    )
    sem_temp = st.multiselect("temporal_id (text)", _opts(df_sem, "temporal_id"), default=[])
    sem_model = st.multiselect("modeling_id (opsional)", _opts(df_sem, "modeling_id"), default=[])

    st.divider()
    st.subheader("Mode Ranking")
    syn_scope = st.radio("Ranking Sintaksis per:", ["job_id+modeling_id+temporal_id+threshold", "job_id+modeling_id", "global"], index=0)
    sem_scope = st.radio("Ranking Semantik per:", ["embedding_run_id+temporal_id", "embedding_run_id", "global"], index=0)
    sem_metric_mode = st.radio("Metric Semantik:", ["Silhouette+DBI", "DBCV+DBI (fallback)"], index=0)

# Apply filters: Sintaksis
dff_syn = df_syn.copy()
if syn_job:
    dff_syn = dff_syn[dff_syn["job_id"].isin(syn_job)]
if syn_model:
    dff_syn = dff_syn[dff_syn["modeling_id"].isin(syn_model)]
if syn_temp:
    dff_syn = dff_syn[dff_syn["temporal_id"].isin(syn_temp)]
if syn_thr_min is not None:
    dff_syn = dff_syn[(dff_syn["threshold"] >= syn_thr_min) & (dff_syn["threshold"] <= syn_thr_max)]

# Apply filters: Semantik (awal)
dff_sem_raw = df_sem.copy()
if sem_emb:
    dff_sem_raw = dff_sem_raw[dff_sem_raw["embedding_run_id"].isin(sem_emb)]
if sem_model:
    dff_sem_raw = dff_sem_raw[dff_sem_raw["modeling_id"].isin(sem_model)]
if sem_temp:
    dff_sem_raw = dff_sem_raw[dff_sem_raw["temporal_id"].isin(sem_temp)]

st.subheader("🧪 Diagnostik Semantik (sebelum STRICT metrik)")
if dff_sem_raw.empty:
    st.info("Tidak ada data semantik setelah filter awal.")
# else:
#     st.write(
#         {
#             "total_rows_semantik_filtered_awal": int(len(dff_sem_raw)),
#             "rows_lengkap_3_metrik": int(len(dff_sem_raw.dropna(subset=["silhouette_score", "dbi", "dbcv"]))),
#             "missing_silhouette_score": int(dff_sem_raw["silhouette_score"].isna().sum()),
#             "missing_dbi": int(dff_sem_raw["dbi"].isna().sum()),
#             "missing_dbcv": int(dff_sem_raw["dbcv"].isna().sum()),
#         }
#     )

# STRICT semantik
dff_sem = dff_sem_raw.dropna(subset=["silhouette_score", "dbi", "dbcv"]).copy()
st.caption(f"Semantik (valid lengkap metrik): {len(dff_sem)} baris")

# Summary
c1, c2 = st.columns(2)
c1.metric("Rows Sintaksis", int(len(dff_syn)))
c2.metric("Rows Semantik", int(len(dff_sem)))
# c3.metric("Silhouette mean (syn)", f"{float(dff_syn['silhouette_score'].mean()):.4f}" if dff_syn["silhouette_score"].notna().any() else "-")
# c4.metric("DBI mean (sem)", f"{float(dff_sem['dbi'].mean()):.4f}" if (not dff_sem.empty and dff_sem["dbi"].notna().any()) else "-")

st.divider()

# Ranking
left, right = st.columns(2, gap="large")

with left:
    st.subheader("🏆 Best Run — Sintaksis")
    if syn_scope == "global":
        syn_keys = ["jenis_pendekatan"]
    elif syn_scope == "job_id+modeling_id":
        syn_keys = ["jenis_pendekatan", "job_id", "modeling_id"]
    else:
        syn_keys = ["jenis_pendekatan", "job_id", "modeling_id", "temporal_id", "threshold"]

    best_syn = best_by_keys(dff_syn, syn_keys, "sil_dbi")
    st.dataframe(
        best_syn[[c for c in [
            "eval_id","job_id","modeling_id","temporal_id","threshold",
            "silhouette_score","dbi"
        ] if c in best_syn.columns]],
        use_container_width=True,
        height=220
    )

    st.markdown("**Rekomendasi Parameter (Sintaksis)**")
    if best_syn.empty:
        st.info("Tidak ada best run sintaksis (cek filter / metrik NULL).")
    # else:
    #     st.json(recommend_params(best_syn.iloc[0], "sintaksis") or {"info": "meta_json kosong / key tidak dikenali"})

with right:
    st.subheader("🏆 Best Run — Semantik")
    if sem_scope == "global":
        sem_keys = ["jenis_pendekatan"]
    elif sem_scope == "embedding_run_id":
        sem_keys = ["jenis_pendekatan", "embedding_run_id"]
    else:
        sem_keys = ["jenis_pendekatan", "embedding_run_id", "temporal_id"]

    metric_mode = "sil_dbi" if sem_metric_mode == "Silhouette+DBI" else "dbcv_dbi"
    best_sem = best_by_keys(dff_sem, sem_keys, metric_mode)

    st.dataframe(
        best_sem[[c for c in [
            "eval_id","embedding_run_id","modeling_id",
            "silhouette_score","dbi","dbcv"
        ] if c in best_sem.columns]],
        use_container_width=True,
        height=220
    )

    st.markdown("**Rekomendasi Parameter (Semantik)**")
    if best_sem.empty:
        st.info("Tidak ada best run semantik (tidak ada baris dengan silhouette_score+dbi+dbcv).")
    # else:
    #     st.json(recommend_params(best_sem.iloc[0], "semantik") or {"info": "meta_json kosong / key tidak dikenali"})

st.divider()

# Best-of-best
st.subheader("⚖️ Compare Ringkas (Best-of-best)")
bob_syn = best_of_best(dff_syn, "sil_dbi")
bob_sem = best_of_best(dff_sem, metric_mode)

compare = pd.concat([bob_syn, bob_sem], ignore_index=True)
st.dataframe(
    compare[[c for c in [
        "jenis_pendekatan","eval_id","job_id","modeling_id","embedding_run_id",
        "threshold","silhouette_score","dbi","dbcv"
    ] if c in compare.columns]],
    use_container_width=True,
    height=140
)

# Save representatives
st.subheader("💾 Simpan Perwakilan (Sintaksis & Semantik)")

rep_records: List[Dict[str, Any]] = []
if not bob_syn.empty:
    rep_records.append(_row_to_record(bob_syn.iloc[0]))
if not bob_sem.empty:
    rep_records.append(_row_to_record(bob_sem.iloc[0]))

if rep_records:
    st.dataframe(pd.DataFrame(rep_records), use_container_width=True, height=140)
else:
    st.info("Belum ada perwakilan yang bisa disimpan (best-of-best kosong).")

invalid = [r for r in rep_records if not r.get("eval_id")]
if invalid:
    st.warning("Ada perwakilan tanpa eval_id. Tidak bisa disimpan.")

col_save1, col_save2 = st.columns([0.35, 0.65])
with col_save1:
    do_save = st.button("💾 Simpan Perwakilan", use_container_width=True)
with col_save2:
    st.caption(f"Upsert ke {SCHEMA}.{T_SELECTED} (ON CONFLICT jenis_pendekatan, eval_id).")

if do_save:
    if not rep_records or invalid:
        st.error("Tidak ada record valid untuk disimpan.")
    else:
        try:
            eng = get_engine()
            # extra-hard safety: sanitize again (double guard)
            safe_records = [sanitize_record(r) for r in rep_records]
            n = save_selected_representatives(eng, safe_records)
            st.success(f"Berhasil menyimpan {n} perwakilan.")
        except Exception as e:
            st.error("Gagal menyimpan perwakilan ke database.")
            st.code(str(e))
            st.write("Debug safe_records:")
            st.json(safe_records)

st.divider()

