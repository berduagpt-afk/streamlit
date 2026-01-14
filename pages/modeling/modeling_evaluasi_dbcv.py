from __future__ import annotations

import json
import math
from typing import List, Optional

import numpy as np
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


# ======================================================
# ⚙️ Konstanta DB
# ======================================================
SCHEMA = "lasis_djp"

T_RUNS = "modeling_semantik_hdbscan_runs"
T_CLUSTERS = "modeling_semantik_hdbscan_clusters"
T_MEMBERS = "modeling_semantik_hdbscan_members"
T_EMB_VECTORS = "semantik_embedding_vectors"  # embedding_json (jsonb)
T_EVALUATIONS = "modeling_evaluation_results"  # simpan DBCV ke sini


# ======================================================
# 🔌 Database Connection
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
# 🧰 Helpers
# ======================================================
def safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return None


def fmt_float(x: Optional[float], nd: int = 4) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "-"
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def compute_dbcv_safe(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> Optional[float]:
    """
    DBCV via hdbscan.validity.validity_index dengan guard agar stabil.
    """
    if X is None or labels is None:
        return None
    if not isinstance(X, np.ndarray) or not isinstance(labels, np.ndarray):
        return None
    if X.size == 0 or labels.size == 0:
        return None
    if X.shape[0] != labels.shape[0]:
        return None

    uniq = np.unique(labels)
    n_clusters = int(np.sum(uniq != -1))
    n_non_noise = int(np.sum(labels != -1))

    # syarat minimal
    if n_non_noise < 5 or n_clusters < 2:
        return None

    # degeneratif: semua vektor sama
    try:
        if np.allclose(X.max(axis=0), X.min(axis=0)):
            return None
    except Exception:
        return None

    try:
        from hdbscan.validity import validity_index
        return float(validity_index(X, labels, metric=metric))
    except Exception:
        return None


# ======================================================
# ✅ Ensure evaluation table has dbcv + unique key
# ======================================================
def ensure_eval_table(_engine) -> None:
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_EVALUATIONS}
    (
        eval_id uuid NOT NULL DEFAULT gen_random_uuid(),
        run_time timestamptz NOT NULL DEFAULT now(),

        jenis_pendekatan text NOT NULL,
        job_id uuid,
        modeling_id uuid,
        embedding_run_id uuid,
        temporal_id text,

        silhouette_score double precision,
        dbi double precision,
        threshold double precision,
        notes text,
        meta_json jsonb,

        CONSTRAINT {T_EVALUATIONS}_pkey PRIMARY KEY (eval_id)
    );

    ALTER TABLE {SCHEMA}.{T_EVALUATIONS}
        ADD COLUMN IF NOT EXISTS dbcv double precision;

    CREATE UNIQUE INDEX IF NOT EXISTS uq_{T_EVALUATIONS}_core
    ON {SCHEMA}.{T_EVALUATIONS} (
        jenis_pendekatan,
        modeling_id,
        embedding_run_id,
        temporal_id,
        threshold
    );
    """
    stmts = [s.strip() for s in ddl.split(";") if s.strip()]
    with _engine.begin() as conn:
        for s in stmts:
            conn.exec_driver_sql(s)


def save_or_update_dbcv_in_evaluations(
    _engine,
    modeling_id_text: str,
    embedding_run_id_text: str,
    dbcv_value: float,
    notes_append: str,
    meta_json_obj: dict,
) -> str:
    """
    UPSERT dengan preferensi:
    - UPDATE dulu baris evaluasi semantik untuk modeling_id (prioritas threshold=-1 & temporal_id IS NULL)
    - jika tidak ada, INSERT baris baru.
    Return: "updated" atau "inserted"
    """
    notes_append = (notes_append or "").strip()
    meta_json_str = json.dumps(meta_json_obj, ensure_ascii=False)

    with _engine.begin() as conn:
        # 1) UPDATE dulu (prioritas baris DBCV: threshold=-1.0 & temporal_id IS NULL)
        q_update = text(
            f"""
            UPDATE {SCHEMA}.{T_EVALUATIONS} t
            SET
                run_time = now(),
                dbcv = :dbcv,
                notes = CASE
                    WHEN :notes = '' THEN t.notes
                    WHEN t.notes IS NULL OR t.notes = '' THEN :notes
                    ELSE t.notes || E'\\n' || :notes
                END,
                meta_json = CAST(:meta_json AS jsonb)
            WHERE t.eval_id = (
                SELECT e.eval_id
                FROM {SCHEMA}.{T_EVALUATIONS} e
                WHERE e.jenis_pendekatan = 'semantik'
                  AND e.modeling_id = CAST(:mid AS uuid)
                  AND e.threshold = -1.0
                  AND e.temporal_id IS NULL
                ORDER BY e.run_time DESC
                LIMIT 1
            )
            """
        )
        res = conn.execute(
            q_update,
            {
                "dbcv": float(dbcv_value),
                "notes": notes_append,
                "meta_json": meta_json_str,
                "mid": modeling_id_text,
            },
        )
        if int(getattr(res, "rowcount", 0) or 0) > 0:
            return "updated"

        # 2) Jika belum ada baris DBCV khusus, coba UPDATE baris semantik terbaru (fallback)
        q_update_fallback = text(
            f"""
            UPDATE {SCHEMA}.{T_EVALUATIONS} t
            SET
                run_time = now(),
                dbcv = :dbcv,
                notes = CASE
                    WHEN :notes = '' THEN t.notes
                    WHEN t.notes IS NULL OR t.notes = '' THEN :notes
                    ELSE t.notes || E'\\n' || :notes
                END,
                meta_json = CAST(:meta_json AS jsonb)
            WHERE t.eval_id = (
                SELECT e.eval_id
                FROM {SCHEMA}.{T_EVALUATIONS} e
                WHERE e.jenis_pendekatan = 'semantik'
                  AND e.modeling_id = CAST(:mid AS uuid)
                ORDER BY e.run_time DESC
                LIMIT 1
            )
            """
        )
        res2 = conn.execute(
            q_update_fallback,
            {
                "dbcv": float(dbcv_value),
                "notes": notes_append,
                "meta_json": meta_json_str,
                "mid": modeling_id_text,
            },
        )
        if int(getattr(res2, "rowcount", 0) or 0) > 0:
            return "updated"

        # 3) Jika benar-benar belum ada record untuk modeling_id tsb → INSERT baris baru
        q_insert = text(
            f"""
            INSERT INTO {SCHEMA}.{T_EVALUATIONS}
            (
                jenis_pendekatan, job_id, modeling_id, embedding_run_id, temporal_id,
                silhouette_score, dbi, dbcv, threshold, notes, meta_json
            )
            VALUES
            (
                'semantik',
                NULL,
                CAST(:mid AS uuid),
                CAST(:emb AS uuid),
                NULL,
                NULL,
                NULL,
                :dbcv,
                -1.0,
                NULLIF(:notes, ''),
                CAST(:meta_json AS jsonb)
            )
            ON CONFLICT (jenis_pendekatan, modeling_id, embedding_run_id, temporal_id, threshold)
            DO UPDATE SET
                run_time = now(),
                dbcv = EXCLUDED.dbcv,
                notes = CASE
                    WHEN EXCLUDED.notes IS NULL OR EXCLUDED.notes = '' THEN {SCHEMA}.{T_EVALUATIONS}.notes
                    WHEN {SCHEMA}.{T_EVALUATIONS}.notes IS NULL OR {SCHEMA}.{T_EVALUATIONS}.notes = '' THEN EXCLUDED.notes
                    ELSE {SCHEMA}.{T_EVALUATIONS}.notes || E'\\n' || EXCLUDED.notes
                END,
                meta_json = EXCLUDED.meta_json
            """
        )
        conn.execute(
            q_insert,
            {
                "mid": modeling_id_text,
                "emb": embedding_run_id_text,
                "dbcv": float(dbcv_value),
                "notes": notes_append,
                "meta_json": meta_json_str,
            },
        )
        return "inserted"



# ======================================================
# 📥 Loaders
# ======================================================
@st.cache_data(show_spinner=False, ttl=120)
def load_runs() -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            modeling_id::text AS modeling_id,
            run_time,
            embedding_run_id::text AS embedding_run_id,
            n_rows,
            n_clusters,
            n_noise,
            dbcv,
            params_json,
            notes
        FROM {SCHEMA}.{T_RUNS}
        ORDER BY run_time DESC
        LIMIT 200
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn)


@st.cache_data(show_spinner=False, ttl=120)
def count_members(modeling_id_text: str) -> int:
    q = text(
        f"""
        SELECT COUNT(*)::bigint
        FROM {SCHEMA}.{T_MEMBERS}
        WHERE modeling_id::text = :mid
        """
    )
    with engine.begin() as conn:
        return int(conn.execute(q, {"mid": modeling_id_text}).scalar() or 0)


@st.cache_data(show_spinner=False, ttl=120)
def load_member_keys(modeling_id_text: str) -> List[str]:
    # ✅ PATCH: ORDER BY agar deterministik (sampling jadi konsisten)
    q = text(
        f"""
        SELECT incident_number::text
        FROM {SCHEMA}.{T_MEMBERS}
        WHERE modeling_id::text = :mid
        ORDER BY incident_number
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(q, {"mid": modeling_id_text}).fetchall()
    return [str(r[0]).strip() for r in rows if r and r[0] is not None]


@st.cache_data(show_spinner=True, ttl=120)
def load_members_by_keys(modeling_id_text: str, keys: List[str]) -> pd.DataFrame:
    if not keys:
        return pd.DataFrame()
    q = text(
        f"""
        SELECT
            modeling_id::text AS modeling_id,
            cluster_id,
            is_noise,
            incident_number,
            tgl_submit,
            site,
            modul,
            sub_modul,
            prob,
            outlier_score
        FROM {SCHEMA}.{T_MEMBERS}
        WHERE modeling_id::text = :mid
          AND incident_number::text = ANY(CAST(:keys AS text[]))
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"mid": modeling_id_text, "keys": keys})


@st.cache_data(show_spinner=True, ttl=120)
def load_embeddings_for_members(emb_run_id_text: str, inc_arr: List[str]) -> pd.DataFrame:
    if not inc_arr:
        return pd.DataFrame()
    q = text(
        f"""
        SELECT
            incident_number::text AS key_ticket,
            embedding_json
        FROM {SCHEMA}.{T_EMB_VECTORS}
        WHERE run_id::text = :rid
          AND incident_number::text = ANY(CAST(:inc_arr AS text[]))
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"rid": emb_run_id_text, "inc_arr": inc_arr})


@st.cache_data(show_spinner=False, ttl=120)
def load_eval_preview(_engine, modeling_id_text: str, embedding_run_id_text: str) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            run_time,
            jenis_pendekatan,
            modeling_id::text AS modeling_id,
            embedding_run_id::text AS embedding_run_id,
            temporal_id,
            threshold,
            dbcv,
            notes
        FROM {SCHEMA}.{T_EVALUATIONS}
        WHERE jenis_pendekatan = 'semantik'
          AND modeling_id::text = :mid
          AND embedding_run_id::text = :erid
          AND threshold = -1.0
          AND temporal_id IS NULL
        ORDER BY run_time DESC
        LIMIT 50
        """
    )
    with _engine.begin() as conn:
        return pd.read_sql(q, conn, params={"mid": modeling_id_text, "erid": embedding_run_id_text})


# ======================================================
# 🧭 UI
# ======================================================
st.title("Evaluasi Clustering Semantik — DBCV (HDBSCAN)")
st.caption("DBCV dihitung dari embedding (semantik_embedding_vectors) + label cluster (members).")

# warn jika hdbscan belum ada
try:
    import hdbscan  # noqa: F401
except Exception:
    st.warning("Library **hdbscan** belum tersedia. Install: `pip install hdbscan` (DBCV akan '-' bila tidak ada).")

ensure_eval_table(engine)

with st.expander("Pengaturan", expanded=True):
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    with c1:
        max_points = st.number_input(
            "Maksimum titik untuk evaluasi (sampling acak)",
            min_value=500,
            max_value=200_000,
            value=20_000,
            step=500,
            help="Jika data > batas ini, dilakukan sampling acak.",
        )
    with c2:
        metric = st.selectbox(
            "Metric DBCV",
            options=["euclidean", "cosine", "manhattan"],
            index=0,
            help="Jika embedding dinormalisasi, cosine sering masuk akal.",
        )
    with c3:
        seed = st.number_input("Random seed", min_value=1, max_value=999999, value=42, step=1)
    with c4:
        min_cluster_size_dbcv = st.number_input(
            "Min ukuran cluster utk DBCV (min-size variant)",
            min_value=2,
            max_value=50,
            value=5,
            step=1,
            help="Mengurangi ketidakstabilan DBCV akibat cluster mikro (1–2).",
        )

rng = np.random.default_rng(int(seed))

df_runs = load_runs()
if df_runs.empty:
    st.warning("Belum ada data di tabel runs HDBSCAN.")
    st.stop()

mid = st.selectbox("Pilih modeling_id", options=df_runs["modeling_id"].tolist(), index=0)
run_row = df_runs[df_runs["modeling_id"] == mid].iloc[0]

embedding_run_id = str(run_row.get("embedding_run_id") or "").strip()
if not embedding_run_id:
    st.error("Kolom embedding_run_id kosong pada runs. Pastikan runs menyimpan embedding_run_id (sesuai DDL).")
    st.stop()

with st.expander("Info Run", expanded=False):
    st.write(f"**modeling_id**: `{mid}`")
    st.write(f"**embedding_run_id**: `{embedding_run_id}`")
    st.write(f"**run_time**: {run_row.get('run_time')}")
    if run_row.get("notes"):
        st.write(f"**notes**: {run_row.get('notes')}")
    pj = safe_json_loads(run_row.get("params_json"))
    if pj:
        st.json(pj)

# ======================================================
# 🧾 Load members + sampling (deterministik)
# ======================================================
n_total = count_members(mid)
st.write(f"Total member: **{n_total:,}**")

all_keys = load_member_keys(mid)
if not all_keys:
    st.warning("Tidak ada member untuk modeling_id ini.")
    st.stop()

need_sample = len(all_keys) > int(max_points)
if need_sample:
    keys_sample = rng.choice(np.array(all_keys, dtype=object), size=int(max_points), replace=False).tolist()
else:
    keys_sample = all_keys

# unik + strip
keys_sample = sorted(set(map(lambda x: str(x).strip(), keys_sample)))

df_mem = load_members_by_keys(mid, keys_sample)
if df_mem.empty:
    st.warning("Query members kosong setelah sampling.")
    st.stop()

df_mem["incident_number"] = df_mem["incident_number"].astype(str).str.strip()

# ======================================================
# ✅ Join embedding vectors (run_id + incident_number)
# ======================================================
inc_list = sorted(set(df_mem["incident_number"].tolist()))
df_vec = load_embeddings_for_members(embedding_run_id, inc_list)

if df_vec.empty:
    st.error(
        "Tidak ada embedding yang cocok.\n"
        f"Pastikan: semantik_embedding_vectors berisi run_id={embedding_run_id} dan incident_number yang sama."
    )
    st.stop()

df_vec["key_ticket"] = df_vec["key_ticket"].astype(str).str.strip()

merged = df_mem.merge(df_vec, left_on="incident_number", right_on="key_ticket", how="inner")
if merged.empty:
    st.error("Join members ↔ embedding_vectors menghasilkan 0 baris (key tidak cocok).")
    st.write("Contoh incident_number (members):", df_mem["incident_number"].head(3).tolist())
    st.write("Contoh key_ticket (vectors):", df_vec["key_ticket"].head(3).tolist())
    st.stop()

# ✅ diagnostics join
st.caption(
    f"Join diagnostics: members={len(df_mem):,} | vectors={len(df_vec):,} | joined={len(merged):,} "
    f"| join_rate={(len(merged)/max(len(df_mem),1)):.2%}"
)

# ======================================================
# 🧠 Parse embedding + label (-1 untuk noise)
# ======================================================
objs = merged["embedding_json"].map(safe_json_loads)
valid_mask = objs.map(lambda x: isinstance(x, list) and len(x) > 0)
merged2 = merged.loc[valid_mask].copy()

if merged2.shape[0] < 10:
    st.error("Embedding terlalu sedikit setelah parsing. Pastikan embedding_json berupa list angka.")
    st.stop()

emb_list: List[List[float]] = objs.loc[valid_mask].tolist()
labels_raw = merged2["cluster_id"].astype(int).to_numpy()
is_noise_arr = merged2["is_noise"].fillna(False).astype(bool).to_numpy()
labels = np.where(is_noise_arr, -1, labels_raw).astype(np.int32)

X = np.asarray(emb_list, dtype=np.float64)
feature_mode = f"Embedding ({X.shape[1]} dim) | run_id={embedding_run_id}"

# ======================================================
# 📊 KPI & DBCV
# ======================================================
n_points = int(X.shape[0])
n_noise = int((labels == -1).sum())
n_non_noise = n_points - n_noise
uniq = np.unique(labels)
n_clusters = int(np.sum(uniq != -1))
coverage = (n_non_noise / n_points) if n_points else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mode fitur", feature_mode)
k2.metric("Jumlah titik", f"{n_points:,}")
k3.metric("Cluster (non-noise)", f"{n_clusters:,}")
k4.metric("Noise (-1)", f"{n_noise:,}")

# ✅ prerequisites
st.caption(f"DBCV prerequisites: n_non_noise={n_non_noise:,} | n_clusters(non-noise)={n_clusters:,}")

# mask utk min cluster size
cluster_sizes_non_noise = pd.Series(labels[labels != -1]).value_counts()
valid_clusters = cluster_sizes_non_noise[cluster_sizes_non_noise >= int(min_cluster_size_dbcv)].index.to_numpy()

if len(valid_clusters) < 2:
    st.warning("Cluster valid (min-size) < 2 → DBCV min-size kemungkinan ‘-’.")

mask_min_size = np.isin(labels, valid_clusters)
X_min = X[mask_min_size]
y_min = labels[mask_min_size]

with st.spinner("Menghitung DBCV (overall, tanpa noise, min-size)..."):
    dbcv_overall = compute_dbcv_safe(X, labels, metric=metric)

    mask_nn = labels != -1
    X_nn = X[mask_nn]
    y_nn = labels[mask_nn]
    dbcv_no_noise = compute_dbcv_safe(X_nn, y_nn, metric=metric)

    dbcv_min_size = compute_dbcv_safe(X_min, y_min, metric=metric)

cA, cB, cC, cD = st.columns(4)
cA.metric("Coverage non-noise", f"{coverage:.2%}")
cB.metric("DBCV (overall)", fmt_float(dbcv_overall, 4))
cC.metric("DBCV (tanpa noise)", fmt_float(dbcv_no_noise, 4))
cD.metric(f"DBCV (min size ≥ {int(min_cluster_size_dbcv)})", fmt_float(dbcv_min_size, 4))

st.caption(f"Cluster valid utk min-size: **{len(valid_clusters)}** dari total **{n_clusters}** (non-noise).")

with st.expander("Interpretasi singkat", expanded=True):
    st.markdown(
        f"""
- **Coverage non-noise**: {coverage:.2%}
- Jika DBCV **“-”**, biasanya karena **cluster efektif < 2** atau terlalu banyak cluster mikro sehingga metrik tidak stabil.
- **ORDER BY incident_number** membuat sampling konsisten untuk seed yang sama (mengurangi DBCV “kadang ada kadang tidak”).
        """
    )

# ======================================================
# 💾 Simpan DBCV ke modeling_evaluation_results
# ======================================================
st.divider()
st.subheader("Simpan hasil DBCV ke tabel evaluasi (modeling_evaluation_results)")

dbcv_choices = {
    "DBCV (tanpa noise) — disarankan": dbcv_no_noise,
    "DBCV (overall)": dbcv_overall,
    f"DBCV (min size ≥ {int(min_cluster_size_dbcv)})": dbcv_min_size,
}

choice = st.selectbox("Pilih nilai DBCV yang disimpan", options=list(dbcv_choices.keys()), index=0)
selected_val = dbcv_choices[choice]

note_append = st.text_input(
    "Tambahkan catatan (opsional) ke notes evaluasi",
    value=f"[DBCV] saved={choice} metric={metric} max_points={int(max_points)} seed={int(seed)}",
)

meta_obj = {
    "page": "evaluation_semantik_dbcv",
    "metric": metric,
    "max_points": int(max_points),
    "seed": int(seed),
    "min_cluster_size_dbcv": int(min_cluster_size_dbcv),
    "modeling_id": mid,
    "embedding_run_id": embedding_run_id,
    "dbcv_choice": choice,
    "join_diagnostics": {
        "members_sample": int(len(df_mem)),
        "vectors_found": int(len(df_vec)),
        "joined": int(len(merged)),
        "join_rate": float(len(merged) / max(len(df_mem), 1)),
    },
    "prerequisites": {
        "n_points": int(n_points),
        "n_non_noise": int(n_non_noise),
        "n_clusters_non_noise": int(n_clusters),
    },
}

colS1, colS2 = st.columns([1, 2])
with colS1:
    if st.button("💾 Simpan DBCV", type="primary"):
        if selected_val is None:
            st.error("Nilai DBCV yang dipilih kosong (‘-’). Tidak disimpan.")
        else:
            try:
                action = save_or_update_dbcv_in_evaluations(
                    engine,
                    modeling_id_text=mid,
                    embedding_run_id_text=embedding_run_id,
                    dbcv_value=float(selected_val),
                    notes_append=note_append.strip(),
                    meta_json_obj=meta_obj,
                )
                if action == "updated":
                    st.success(f"DBCV berhasil di-UPDATE pada {SCHEMA}.{T_EVALUATIONS} untuk modeling_id={mid}")
                else:
                    st.success(f"DBCV berhasil di-INSERT (baru) ke {SCHEMA}.{T_EVALUATIONS} untuk modeling_id={mid}")
            except Exception as e:
                st.error(f"Gagal simpan: {e}")
with colS2:
    st.info(
        f"Yang akan disimpan: **{choice} = {fmt_float(selected_val, 6)}**\n\n"
        f"Target tabel: `{SCHEMA}.{T_EVALUATIONS}` (kolom `dbcv`, threshold=-1.0, temporal_id=NULL)."
    )

with st.expander("Preview data evaluasi yang tersimpan (50 terbaru)", expanded=False):
    try:
        df_prev = load_eval_preview(engine, mid, embedding_run_id)
        st.dataframe(df_prev, use_container_width=True, height=260)
    except Exception as e:
        st.info(f"Gagal load preview: {e}")

# ======================================================
# 📈 Distribusi ukuran cluster
# ======================================================
st.divider()
st.subheader("Distribusi Ukuran Cluster")

sizes = (
    pd.Series(labels)
    .value_counts()
    .rename_axis("cluster_id")
    .reset_index(name="n")
    .sort_values(["cluster_id"])
)

cL, cR = st.columns([1, 1])

with cL:
    st.dataframe(
        sizes.rename(columns={"n": "jumlah_member"}),
        use_container_width=True,
        height=360,
    )

with cR:
    sizes_nn = sizes[sizes["cluster_id"] != -1].copy()
    if not sizes_nn.empty:
        topn = sizes_nn.sort_values("n", ascending=False).head(30)
        ch = (
            alt.Chart(topn)
            .mark_bar()
            .encode(
                x=alt.X("cluster_id:O", title="cluster_id"),
                y=alt.Y("n:Q", title="jumlah member"),
                tooltip=["cluster_id:O", "n:Q"],
            )
            .properties(height=340)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("Tidak ada cluster non-noise (semua noise).")

# ======================================================
# 🧾 Ringkasan tabel clusters (opsional)
# ======================================================
st.divider()
st.subheader("Ringkasan Cluster (tabel clusters, bila tersedia)")

@st.cache_data(show_spinner=False, ttl=120)
def load_clusters(modeling_id_text: str) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            modeling_id::text AS modeling_id,
            cluster_id,
            cluster_size,
            avg_prob,
            avg_outlier_score,
            span_days
        FROM {SCHEMA}.{T_CLUSTERS}
        WHERE modeling_id::text = :mid
        ORDER BY cluster_size DESC, cluster_id ASC
        LIMIT 2000
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"mid": modeling_id_text})

try:
    df_clusters = load_clusters(mid)
    if df_clusters.empty:
        st.info("Tidak ada ringkasan di tabel clusters untuk modeling_id ini.")
    else:
        st.dataframe(df_clusters, use_container_width=True, height=420)
except Exception as e:
    st.info(f"Tabel clusters belum siap / strukturnya berbeda: {e}")
