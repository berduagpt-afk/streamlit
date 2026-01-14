# pages/evaluation_sintaksis_job_silhouette_dbi.py
# ============================================================
# Evaluasi Sintaksis (Per JOB_ID) — Silhouette Score & DBI
# Menampilkan hasil untuk seluruh threshold/run dalam satu job_id.
#
# Sumber:
# - lasis_djp.modeling_sintaksis_runs
# - lasis_djp.modeling_sintaksis_members
# - lasis_djp.modeling_sintaksis_clusters
# - (butuh) lasis_djp.incident_tfidf_vectors (tfidf_json)
#
# ✅ FINAL PATCH:
# - Penyimpanan ke lasis_djp.modeling_evaluation_results
# - Mencegah INSERT berulang saat embedding_run_id NULL:
#   → unique index khusus sintaksis (tanpa embedding_run_id)
#   → ON CONFLICT target sesuai index tersebut
# ============================================================

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from scipy.sparse import csr_matrix
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize, StandardScaler
from sqlalchemy import create_engine, text


# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ======================================================
# ⚙️ KONSTANTA DB
# ======================================================
SCHEMA = "lasis_djp"
T_RUNS = "modeling_sintaksis_runs"
T_MEMBERS = "modeling_sintaksis_members"
T_CLUSTERS = "modeling_sintaksis_clusters"
T_EVAL = "modeling_evaluation_results"  # tabel gabungan (sintaksis + semantik)

# ⚠️ Sesuaikan jika beda
T_TFIDF_VECTORS = "incident_tfidf_vectors"  # kolom: run_id, incident_number, tfidf_json


# ======================================================
# 🔌 DB CONNECTION
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
        future=True,
    )


engine = get_engine()


# ======================================================
# ✅ Ensure eval table + UNIQUE khusus sintaksis
# ======================================================
def ensure_eval_table(_engine) -> None:
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_EVAL}
    (
        eval_id uuid NOT NULL DEFAULT gen_random_uuid(),
        run_time timestamp with time zone NOT NULL DEFAULT now(),

        jenis_pendekatan text NOT NULL,
        job_id uuid,
        modeling_id uuid,
        embedding_run_id uuid,
        temporal_id text,

        silhouette_score double precision,
        dbi double precision,
        dbcv double precision,

        threshold double precision,
        notes text,
        meta_json jsonb,

        CONSTRAINT {T_EVAL}_pkey PRIMARY KEY (eval_id)
    );

    -- Pastikan kolom ada bila tabel sudah terbuat duluan
    ALTER TABLE {SCHEMA}.{T_EVAL} ADD COLUMN IF NOT EXISTS dbcv double precision;
    ALTER TABLE {SCHEMA}.{T_EVAL} ADD COLUMN IF NOT EXISTS silhouette_score double precision;
    ALTER TABLE {SCHEMA}.{T_EVAL} ADD COLUMN IF NOT EXISTS dbi double precision;
    ALTER TABLE {SCHEMA}.{T_EVAL} ADD COLUMN IF NOT EXISTS threshold double precision;
    ALTER TABLE {SCHEMA}.{T_EVAL} ADD COLUMN IF NOT EXISTS notes text;
    ALTER TABLE {SCHEMA}.{T_EVAL} ADD COLUMN IF NOT EXISTS meta_json jsonb;

    -- Index umum (boleh ada untuk semantik/sintaksis campur)
    CREATE UNIQUE INDEX IF NOT EXISTS uq_{T_EVAL}_core
    ON {SCHEMA}.{T_EVAL} (
        jenis_pendekatan,
        modeling_id,
        embedding_run_id,
        temporal_id,
        threshold
    );

    -- ✅ KRUSIAL: UNIQUE khusus sintaksis TANPA embedding_run_id (karena NULL tidak konflik di PG)
    CREATE UNIQUE INDEX IF NOT EXISTS uq_{T_EVAL}_sintaksis
    ON {SCHEMA}.{T_EVAL} (
        jenis_pendekatan,
        job_id,
        modeling_id,
        temporal_id,
        threshold
    );

    CREATE INDEX IF NOT EXISTS idx_{T_EVAL}_main
    ON {SCHEMA}.{T_EVAL} (jenis_pendekatan, job_id, modeling_id, run_time DESC);
    """

    stmts = [s.strip() for s in ddl.split(";") if s.strip()]
    with _engine.begin() as conn:
        for s in stmts:
            conn.exec_driver_sql(s)


ensure_eval_table(engine)


# ======================================================
# 🧰 Helpers
# ======================================================
def _safe_json_obj(x: Any) -> Dict[str, float]:
    """Terima dict atau string JSON. Return dict term->float (filter non-finite/0)."""
    if x is None:
        return {}
    if isinstance(x, dict):
        d = x
    elif isinstance(x, str):
        try:
            d = json.loads(x)
        except Exception:
            return {}
        if not isinstance(d, dict):
            return {}
    else:
        return {}

    out: Dict[str, float] = {}
    for k, v in d.items():
        try:
            fv = float(v)
        except Exception:
            continue
        if fv == 0.0 or (not np.isfinite(fv)):
            continue
        out[str(k)] = fv
    return out


def build_csr(tfidf_list: List[Any]) -> Tuple[csr_matrix, int]:
    """Build CSR matrix dari list tfidf_json per dokumen."""
    vocab: Dict[str, int] = {}
    indptr: List[int] = [0]
    indices: List[int] = []
    data: List[float] = []

    for doc in tfidf_list:
        d = _safe_json_obj(doc)
        for term, fv in d.items():
            if term not in vocab:
                vocab[term] = len(vocab)
            indices.append(vocab[term])
            data.append(float(fv))
        indptr.append(len(indices))

    X = csr_matrix((data, indices, indptr), dtype=np.float64)
    X.sum_duplicates()
    return X, len(vocab)


def _fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "-"
    return f"{x:.{nd}f}"


def _min_cluster_ok(labels: np.ndarray) -> bool:
    """Silhouette & DBI butuh >=2 cluster dan tidak semua label unik."""
    uniq = np.unique(labels)
    return len(uniq) >= 2 and len(uniq) < len(labels)


@st.cache_data(show_spinner=False, ttl=300)
def get_table_columns(_engine, schema: str, table: str) -> set[str]:
    q = text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
    """)
    with _engine.begin() as conn:
        df = pd.read_sql(q, conn, params={"schema": schema, "table": table})
    return set(df["column_name"].astype(str).tolist()) if not df.empty else set()


def resolve_temporal_id_for_sintaksis(_engine, modeling_id_text: str) -> str:
    """
    Fallback builder untuk temporal_id.
    Jika tabel temporal sintaksis ada, ambil kombinasi window_days|time_col|include_noise terbaru.
    Jika tidak ada, kembalikan 'NA'.
    """
    t_temporal = "modeling_sintaksis_temporal_summary"
    cols = get_table_columns(_engine, SCHEMA, t_temporal)
    needed = {"modeling_id", "window_days", "time_col", "include_noise"}
    if not needed.issubset(cols):
        return "NA"

    q = text(f"""
        SELECT window_days, time_col, include_noise
        FROM {SCHEMA}.{t_temporal}
        WHERE modeling_id::text = :mid
        ORDER BY run_time DESC
        LIMIT 1
    """)
    with _engine.begin() as conn:
        row = conn.execute(q, {"mid": modeling_id_text}).fetchone()

    if not row:
        return "NA"

    wd, tc, inoise = row[0], row[1], row[2]
    return f"wd={wd}|time={tc}|noise={inoise}"


def save_eval_results_sintaksis(
    _engine,
    job_id_text: str,
    df_res: pd.DataFrame,
    notes: str = "",
) -> int:
    """
    Simpan hasil evaluasi sintaksis (per threshold/run) ke tabel gabungan.
    df_res minimal punya: modeling_id, threshold, silhouette, dbi, n_used, n_clusters_used, note(optional)
    """
    if df_res.empty:
        return 0

    n_saved = 0
    with _engine.begin() as conn:
        for _, r in df_res.iterrows():
            mid = str(r.get("modeling_id", "")).strip()
            if not mid or mid.lower() == "nan":
                continue

            temporal_id = resolve_temporal_id_for_sintaksis(_engine, mid)

            meta_obj = {
                "n_used": int(r.get("n_used", 0) or 0),
                "n_clusters_used": int(r.get("n_clusters_used", 0) or 0),
                "note_row": str(r.get("note", "") or ""),
            }

            payload = {
                "jenis_pendekatan": "sintaksis",
                "job_id": str(job_id_text),
                "modeling_id": mid,
                "embedding_run_id": None,  # sintaksis tidak pakai embedding
                "temporal_id": temporal_id,
                "silhouette_score": r.get("silhouette", None),
                "dbi": r.get("dbi", None),
                "dbcv": None,  # tidak dipakai untuk sintaksis
                "threshold": r.get("threshold", None),
                "notes": notes or "",
                "meta_json": json.dumps(meta_obj, ensure_ascii=False),
            }

            # ✅ ON CONFLICT target = unique index sintaksis (tanpa embedding_run_id)
            conn.execute(
                text(f"""
                    INSERT INTO {SCHEMA}.{T_EVAL}
                    (
                        jenis_pendekatan, job_id, modeling_id, embedding_run_id, temporal_id,
                        silhouette_score, dbi, dbcv, threshold, notes, meta_json
                    )
                    VALUES
                    (
                        :jenis_pendekatan,
                        CAST(:job_id AS uuid),
                        CAST(:modeling_id AS uuid),
                        NULL,
                        :temporal_id,
                        :silhouette_score,
                        :dbi,
                        NULL,
                        :threshold,
                        :notes,
                        CAST(:meta_json AS jsonb)
                    )
                    ON CONFLICT (jenis_pendekatan, job_id, modeling_id, temporal_id, threshold)
                    DO UPDATE SET
                        run_time = now(),
                        silhouette_score = EXCLUDED.silhouette_score,
                        dbi = EXCLUDED.dbi,
                        notes = EXCLUDED.notes,
                        meta_json = EXCLUDED.meta_json
                """),
                payload,
            )
            n_saved += 1

    return n_saved


# ======================================================
# 📥 LOADERS
# ======================================================
@st.cache_data(show_spinner=False, ttl=120)
def load_jobs() -> pd.DataFrame:
    q = text(f"""
        SELECT
            job_id::text AS job_id,
            COUNT(*)::int AS n_runs,
            MIN(run_time) AS min_run_time,
            MAX(run_time) AS max_run_time,
            MIN(tfidf_run_id::text) AS tfidf_run_id_any
        FROM {SCHEMA}.{T_RUNS}
        WHERE job_id IS NOT NULL
        GROUP BY job_id
        ORDER BY MAX(run_time) DESC
        LIMIT 300
    """)
    with engine.begin() as conn:
        return pd.read_sql(q, conn)


@st.cache_data(show_spinner=False, ttl=120)
def load_runs_for_job(job_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
            modeling_id::text AS modeling_id,
            job_id::text AS job_id,
            run_time,
            tfidf_run_id::text AS tfidf_run_id,
            threshold,
            knn_k,
            n_rows,
            n_clusters_all,
            n_singletons
        FROM {SCHEMA}.{T_RUNS}
        WHERE job_id::text = :jid
        ORDER BY threshold ASC NULLS LAST, run_time DESC
    """)
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"jid": job_id})


@st.cache_data(show_spinner=True, ttl=120)
def load_members_for_job(job_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
            modeling_id::text AS modeling_id,
            incident_number::text AS incident_number,
            cluster_id::bigint AS cluster_id
        FROM {SCHEMA}.{T_MEMBERS}
        WHERE job_id::text = :jid
    """)
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"jid": job_id})


@st.cache_data(show_spinner=True, ttl=120)
def load_clusters_for_job(job_id: str) -> pd.DataFrame:
    q = text(f"""
        SELECT
            modeling_id::text AS modeling_id,
            cluster_id::bigint AS cluster_id,
            cluster_size::int AS cluster_size
        FROM {SCHEMA}.{T_CLUSTERS}
        WHERE job_id::text = :jid
    """)
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"jid": job_id})


@st.cache_data(show_spinner=True, ttl=120)
def load_vectors_for_incidents(tfidf_run_id: str, incidents: List[str]) -> pd.DataFrame:
    if not incidents:
        return pd.DataFrame()

    q = text(f"""
        SELECT
            incident_number::text AS incident_number,
            tfidf_json
        FROM {SCHEMA}.{T_TFIDF_VECTORS}
        WHERE run_id::text = :rid
          AND incident_number::text = ANY(CAST(:inc_arr AS text[]))
    """)
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"rid": str(tfidf_run_id), "inc_arr": incidents})


# ======================================================
# 🧭 UI
# ======================================================
st.title("📏 Evaluasi Sintaksis — Silhouette & DBI (Per Job)")
st.caption("Menghitung metrik untuk seluruh run/threshold yang berada dalam satu job_id.")

jobs = load_jobs()
if jobs.empty:
    st.warning("Belum ada job_id pada modeling_sintaksis_runs.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Pengaturan")

    job_labels = jobs.apply(
        lambda r: (
            f"{str(r['job_id'])[:8]}... | runs={int(r['n_runs'])} | "
            f"{pd.to_datetime(r['min_run_time']).date()}..{pd.to_datetime(r['max_run_time']).date()}"
        ),
        axis=1,
    ).tolist()
    job_map = dict(zip(job_labels, jobs["job_id"].astype(str).tolist()))
    sel = st.selectbox("Pilih job_id", options=job_labels, index=0)
    job_id = job_map[sel]

    exclude_singletons = st.checkbox(
        "Keluarkan klaster singleton (ukuran=1)",
        value=True,
        help="Disarankan agar metrik lebih stabil.",
    )

    max_points = st.number_input(
        "Maksimum titik untuk evaluasi (sampling acak)",
        min_value=500,
        max_value=200_000,
        value=20_000,
        step=500,
        help="Jika data > batas, evaluasi dilakukan pada sampel acak agar cepat.",
    )

    seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)

    st.divider()
    st.markdown("**Pengaturan DBI**")
    svd_dim = st.slider(
        "Dimensi SVD untuk DBI",
        min_value=10,
        max_value=200,
        value=50,
        step=5,
        help="DBI dihitung pada ruang TruncatedSVD (lebih efisien dibanding TF-IDF asli).",
    )

rng = np.random.default_rng(int(seed))

# ======================================================
# Load runs + members + clusters untuk job
# ======================================================
df_runs = load_runs_for_job(job_id)
if df_runs.empty:
    st.warning("Tidak ada run untuk job_id ini.")
    st.stop()

df_mem_all = load_members_for_job(job_id)
if df_mem_all.empty:
    st.error("Tidak ada members untuk job_id ini (tabel modeling_sintaksis_members).")
    st.stop()

df_clu_all = load_clusters_for_job(job_id)

st.subheader("Ringkasan Job")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Job ID", job_id)
c2.metric("Jumlah Run", f"{len(df_runs):,}")
c3.metric("Threshold tersedia", f"{df_runs['threshold'].dropna().nunique():,}")
c4.metric("Total member (job)", f"{df_mem_all.shape[0]:,}")

# ======================================================
# Bangun matriks X untuk JOB (sekali)
# ======================================================
tfidf_ids = df_runs["tfidf_run_id"].dropna().astype(str).unique().tolist()
tfidf_run_id = tfidf_ids[0] if tfidf_ids else ""

if not tfidf_run_id or tfidf_run_id.lower() == "nan":
    st.error("tfidf_run_id tidak ditemukan pada runs job ini. Tidak bisa membangun TF-IDF matrix.")
    st.stop()

inc_unique = sorted(set(df_mem_all["incident_number"].astype(str).str.strip().tolist()))
inc_unique = [x for x in inc_unique if x]

with st.spinner("Mengambil TF-IDF vectors untuk job dan membangun matriks (sekali) ..."):
    df_vec = load_vectors_for_incidents(tfidf_run_id, inc_unique)
    if df_vec.empty:
        st.error(
            f"TF-IDF vectors kosong. Cek {SCHEMA}.{T_TFIDF_VECTORS} "
            f"untuk run_id={tfidf_run_id} dan key incident_number."
        )
        st.stop()

    df_vec["incident_number"] = df_vec["incident_number"].astype(str).str.strip()
    df_vec = df_vec.drop_duplicates(subset=["incident_number"], keep="first")
    df_vec = df_vec[df_vec["incident_number"].isin(inc_unique)].copy()

    inc_to_idx = {inc: i for i, inc in enumerate(df_vec["incident_number"].tolist())}

    X, vocab_size = build_csr(df_vec["tfidf_json"].tolist())
    normalize(X, norm="l2", axis=1, copy=False)

st.caption(
    f"TF-IDF matrix job: **{X.shape[0]:,} dokumen**, vocab **{vocab_size:,}** (tfidf_run_id={tfidf_run_id})."
)

with st.spinner("Menghitung TruncatedSVD untuk DBI (sekali per job) ..."):
    n_comp = int(min(int(svd_dim), max(2, X.shape[1] - 1)))
    svd = TruncatedSVD(n_components=n_comp, random_state=int(seed))
    Z = svd.fit_transform(X)
    Z = StandardScaler(with_mean=True, with_std=True).fit_transform(Z)

N = X.shape[0]
if N > int(max_points):
    sample_idx_global = rng.choice(np.arange(N), size=int(max_points), replace=False)
    sample_idx_global = np.sort(sample_idx_global)
    sample_note = f"sample {len(sample_idx_global):,} of {N:,}"
else:
    sample_idx_global = np.arange(N)
    sample_note = "all points"

st.info(f"Mode evaluasi: **{sample_note}**")

# ======================================================
# Evaluasi per run/threshold
# ======================================================
st.subheader("Hasil Evaluasi per Threshold (dalam Job)")

sizes_by_mid: Dict[str, Dict[int, int]] = {}
if not df_clu_all.empty:
    for mid, g in df_clu_all.groupby("modeling_id"):
        sizes_by_mid[str(mid)] = dict(zip(g["cluster_id"].astype(int), g["cluster_size"].astype(int)))

results: List[Dict[str, Any]] = []

df_runs2 = df_runs.copy()
df_runs2["threshold"] = pd.to_numeric(df_runs2["threshold"], errors="coerce")
df_runs2 = df_runs2.sort_values(["threshold", "run_time"], ascending=[True, False])

for _, run in df_runs2.iterrows():
    mid = str(run["modeling_id"])
    thr = run.get("threshold")

    g = df_mem_all[df_mem_all["modeling_id"].astype(str) == mid]
    if g.empty:
        results.append(
            dict(
                modeling_id=mid,
                threshold=thr,
                n_used=0,
                n_clusters_used=0,
                silhouette=None,
                dbi=None,
                note="members kosong",
            )
        )
        continue

    incs = g["incident_number"].astype(str).str.strip().tolist()
    cids = g["cluster_id"].astype(int).tolist()

    labels_full = np.full(shape=(N,), fill_value=-999, dtype=np.int32)
    for inc, cid in zip(incs, cids):
        idx = inc_to_idx.get(inc)
        if idx is not None:
            labels_full[idx] = int(cid)

    mask_present = labels_full != -999

    if exclude_singletons:
        sz = sizes_by_mid.get(mid)
        if sz:
            keep_ids = {cid for cid, s in sz.items() if int(s) >= 2}
            mask_keep = np.isin(labels_full, list(keep_ids))
            mask = mask_present & mask_keep
        else:
            mask = mask_present
    else:
        mask = mask_present

    mask_sample = np.zeros(N, dtype=bool)
    mask_sample[sample_idx_global] = True
    final_mask = mask & mask_sample

    y = labels_full[final_mask]
    if y.size < 10 or (not _min_cluster_ok(y)):
        results.append(
            dict(
                modeling_id=mid,
                threshold=thr,
                n_used=int(y.size),
                n_clusters_used=int(len(np.unique(y))) if y.size else 0,
                silhouette=None,
                dbi=None,
                note="data/cluster tidak cukup (>=2 klaster diperlukan)",
            )
        )
        continue

    X_sub = X[final_mask]
    Z_sub = Z[final_mask]

    sil_val = None
    dbi_val = None
    note = ""

    try:
        sil_val = float(silhouette_score(X_sub, y, metric="cosine"))
    except Exception as e:
        note += f"sil_err:{e} "

    try:
        dbi_val = float(davies_bouldin_score(Z_sub, y))
    except Exception as e:
        note += f"dbi_err:{e} "

    results.append(
        dict(
            modeling_id=mid,
            threshold=thr,
            n_used=int(y.size),
            n_clusters_used=int(len(np.unique(y))),
            silhouette=sil_val,
            dbi=dbi_val,
            note=note.strip() if note.strip() else "",
        )
    )

df_res = pd.DataFrame(results)
df_res["threshold"] = pd.to_numeric(df_res["threshold"], errors="coerce")
df_res = df_res.sort_values("threshold", ascending=True)

table = df_res.copy()
table["silhouette"] = table["silhouette"].apply(lambda x: float(x) if x is not None else np.nan)
table["dbi"] = table["dbi"].apply(lambda x: float(x) if x is not None else np.nan)
table = table.rename(
    columns={
        "threshold": "Threshold",
        "n_used": "Jumlah Tiket Dievaluasi",
        "n_clusters_used": "Jumlah Klaster (dievaluasi)",
        "silhouette": "Silhouette Score (cosine)",
        "dbi": "Davies–Bouldin Index (DBI)",
        "modeling_id": "Modeling ID",
        "note": "Catatan",
    }
)

st.dataframe(table, use_container_width=True)

st.divider()
st.subheader("Simpan Hasil Evaluasi")

save_note = st.text_input(
    "Catatan penyimpanan (opsional)",
    value="Evaluasi sintaksis per job_id: Silhouette(cosine) & DBI(SVD-space).",
)

if st.button("💾 Simpan hasil evaluasi job ini ke tabel gabungan", type="primary"):
    with st.spinner("Menyimpan ke tabel evaluasi gabungan..."):
        n_saved = save_eval_results_sintaksis(
            _engine=engine,
            job_id_text=str(job_id),
            df_res=df_res,
            notes=save_note,
        )
    st.success(f"Selesai. Tersimpan/ter-update: **{n_saved:,}** baris ke `{SCHEMA}.{T_EVAL}`.")

st.markdown("### Grafik Tren Metrik vs Threshold")

plot_df = df_res.dropna(subset=["threshold"]).copy()
plot_df["silhouette"] = pd.to_numeric(plot_df["silhouette"], errors="coerce")
plot_df["dbi"] = pd.to_numeric(plot_df["dbi"], errors="coerce")

cL, cR = st.columns(2)

with cL:
    if plot_df["silhouette"].notna().any():
        ch1 = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title="Threshold"),
                y=alt.Y("silhouette:Q", title="Silhouette Score (cosine)"),
                tooltip=["threshold:Q", "silhouette:Q", "n_used:Q", "n_clusters_used:Q"],
            )
            .properties(height=280, title="Silhouette vs Threshold")
        )
        st.altair_chart(ch1, use_container_width=True)
    else:
        st.info("Silhouette tidak tersedia (banyak run tidak memenuhi syarat klaster minimal).")

with cR:
    if plot_df["dbi"].notna().any():
        ch2 = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title="Threshold"),
                y=alt.Y("dbi:Q", title="Davies–Bouldin Index (DBI)"),
                tooltip=["threshold:Q", "dbi:Q", "n_used:Q", "n_clusters_used:Q"],
            )
            .properties(height=280, title="DBI vs Threshold")
        )
        st.altair_chart(ch2, use_container_width=True)
    else:
        st.info("DBI tidak tersedia (banyak run tidak memenuhi syarat klaster minimal).")

with st.expander("ℹ️ Interpretasi singkat (untuk Bab IV)", expanded=True):
    st.markdown(
        f"""
- Evaluasi ini dilakukan **per job_id** untuk membandingkan performa pembentukan klaster pada berbagai **threshold** dalam kondisi data & TF-IDF yang konsisten.
- **Silhouette Score (cosine)**: semakin tinggi → klaster cenderung semakin terpisah dan anggotanya lebih kompak.
- **DBI**: semakin rendah → dispersi intra-klaster relatif kecil dibanding jarak antar klaster.
- DBI dihitung pada ruang fitur hasil **TruncatedSVD (dim={int(n_comp)})** untuk efisiensi.
- Mode sampling: **{sample_note}** (konsisten untuk semua threshold agar adil).
        """
    )
