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

T_EMB_VECTORS = "semantik_embedding_vectors"  # wajib punya embedding_json


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
    )


engine = get_engine()


# ======================================================
# 🧰 Helpers
# ======================================================
@st.cache_data(show_spinner=False, ttl=300)
def get_table_columns(schema: str, table: str) -> List[str]:
    q = text(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(q, {"schema": schema, "table": table}).fetchall()
    return [r[0] for r in rows]


def safe_json_loads(x):
    if x is None:
        return None
    if isinstance(x, (dict, list)):
        return x
    try:
        return json.loads(x)
    except Exception:
        return None


def fmt_float(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "-"
    return f"{x:.{nd}f}"


def find_first(cols: List[str], candidates: List[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        key = cand.lower()
        if key in lower_map:
            return lower_map[key]
    return None


@st.cache_data(show_spinner=False, ttl=120)
def get_embedding_run_id_from_runs(modeling_id_text: str) -> Optional[str]:
    """
    Ambil embedding_run_id dari modeling_semantik_hdbscan_runs:
    - prioritas: kolom langsung embedding_run_id (jika ada)
    - fallback: params_json (kunci: embedding_run_id / embed_run_id / semantik_run_id / run_id)
    """
    runs_cols = get_table_columns(SCHEMA, T_RUNS)
    col_direct = find_first(
        runs_cols,
        ["embedding_run_id", "embed_run_id", "semantik_run_id", "embedding_run", "run_id_embedding"],
    )

    if col_direct:
        q = text(
            f"""
            SELECT {col_direct}::text AS emb_run_id
            FROM {SCHEMA}.{T_RUNS}
            WHERE modeling_id::text = :mid
            ORDER BY run_time DESC
            LIMIT 1
            """
        )
        with engine.begin() as conn:
            v = conn.execute(q, {"mid": modeling_id_text}).scalar()
        return str(v) if v else None

    q = text(
        f"""
        SELECT params_json
        FROM {SCHEMA}.{T_RUNS}
        WHERE modeling_id::text = :mid
        ORDER BY run_time DESC
        LIMIT 1
        """
    )
    with engine.begin() as conn:
        pj = conn.execute(q, {"mid": modeling_id_text}).scalar()

    obj = safe_json_loads(pj)
    if isinstance(obj, dict):
        for k in ["embedding_run_id", "embed_run_id", "semantik_run_id", "embedding_run", "run_id"]:
            if k in obj and obj[k]:
                return str(obj[k])

    return None


def compute_dbcv(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean") -> float:
    """
    DBCV via hdbscan.validity.validity_index
    """
    try:
        from hdbscan.validity import validity_index
    except Exception as e:
        raise RuntimeError("Package 'hdbscan' belum tersedia. Install: pip install hdbscan") from e

    return float(validity_index(X, labels, metric=metric))


# ======================================================
# 🧭 UI
# ======================================================
st.title("Evaluasi Clustering Semantik — DBCV (HDBSCAN)")
st.caption("DBCV dihitung dari embedding (Opsi A: embedding_run_id diambil dari tabel runs/params_json).")

with st.expander("Pengaturan", expanded=True):
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        max_points = st.number_input(
            "Maksimum titik untuk evaluasi (sampling acak)",
            min_value=500,
            max_value=200_000,
            value=20_000,
            step=500,
            help="DBCV bisa berat untuk embedding berdimensi tinggi. Jika data > batas ini, dilakukan sampling acak.",
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

rng = np.random.default_rng(int(seed))


# ======================================================
# 📥 Ambil daftar runs
# ======================================================
@st.cache_data(show_spinner=False, ttl=120)
def load_runs() -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            modeling_id::text AS modeling_id,
            run_time,
            params_json,
            notes
        FROM {SCHEMA}.{T_RUNS}
        ORDER BY run_time DESC
        LIMIT 200
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn)


df_runs = load_runs()
if df_runs.empty:
    st.warning("Belum ada data di tabel runs HDBSCAN.")
    st.stop()

mid = st.selectbox("Pilih modeling_id", options=df_runs["modeling_id"].tolist(), index=0)
run_row = df_runs[df_runs["modeling_id"] == mid].iloc[0]

with st.expander("Info Run", expanded=False):
    st.write(f"**modeling_id**: `{mid}`")
    st.write(f"**run_time**: {run_row.get('run_time')}")
    if run_row.get("notes"):
        st.write(f"**notes**: {run_row.get('notes')}")
    pj = safe_json_loads(run_row.get("params_json"))
    if pj:
        st.json(pj)


# ======================================================
# 🧾 Load members (sampling acak yang benar)
# ======================================================
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
        n = conn.execute(q, {"mid": modeling_id_text}).scalar()
    return int(n or 0)


@st.cache_data(show_spinner=False, ttl=120)
def load_member_keys(modeling_id_text: str) -> List[str]:
    q = text(
        f"""
        SELECT incident_number::text
        FROM {SCHEMA}.{T_MEMBERS}
        WHERE modeling_id::text = :mid
        """
    )
    with engine.begin() as conn:
        rows = conn.execute(q, {"mid": modeling_id_text}).fetchall()
    return [r[0] for r in rows if r and r[0] is not None]


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
          AND incident_number::text = ANY(:keys)
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"mid": modeling_id_text, "keys": keys})


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

# pastikan list str + unik (ringankan query)
keys_sample = sorted(set(map(str, keys_sample)))

df_mem = load_members_by_keys(mid, keys_sample)
if df_mem.empty:
    st.warning("Query members kosong setelah sampling.")
    st.stop()


# ======================================================
# ✅ Opsi A: Ambil embedding_run_id dari runs, join embedding_vectors via incident_number
# ======================================================
emb_run_id = get_embedding_run_id_from_runs(mid)
if not emb_run_id:
    st.error(
        "embedding_run_id tidak ditemukan di tabel runs/params_json. "
        "Pastikan modeling_semantik_hdbscan_runs menyimpan embedding_run_id (kolom) atau params_json memuatnya."
    )
    st.stop()

vec_cols = get_table_columns(SCHEMA, T_EMB_VECTORS)
col_embedding_json = find_first(vec_cols, ["embedding_json", "embedding", "vector_json"])
col_vec_key = find_first(vec_cols, ["incident_number", "incident_id", "ticket_id", "id_bugtrack", "id"])
col_vec_run = find_first(vec_cols, ["run_id", "embedding_run_id", "embed_run_id"])

if col_embedding_json is None or col_vec_key is None or col_vec_run is None:
    st.error(
        "Tabel semantik_embedding_vectors tidak memiliki kolom yang dibutuhkan.\n"
        f"- embedding_json: {bool(col_embedding_json)}\n"
        f"- key tiket: {bool(col_vec_key)}\n"
        f"- run_id: {bool(col_vec_run)}"
    )
    st.stop()

inc_list = sorted(set(df_mem["incident_number"].astype(str).tolist()))
if not inc_list:
    st.error("incident_number kosong pada members.")
    st.stop()


@st.cache_data(show_spinner=True, ttl=120)
def load_embeddings_for_members(emb_run_id_text: str, inc_arr: List[str]) -> pd.DataFrame:
    if not inc_arr:
        return pd.DataFrame()
    q = text(
        f"""
        SELECT
            {col_vec_key}::text AS key_ticket,
            {col_embedding_json} AS embedding_json
        FROM {SCHEMA}.{T_EMB_VECTORS}
        WHERE {col_vec_run}::text = :rid
          AND {col_vec_key}::text = ANY(:inc_arr)
        """
    )
    with engine.begin() as conn:
        return pd.read_sql(q, conn, params={"rid": emb_run_id_text, "inc_arr": inc_arr})


df_vec = load_embeddings_for_members(emb_run_id, inc_list)
if df_vec.empty:
    st.error(
        "Tidak ada embedding yang cocok untuk members yang diambil.\n"
        f"Cek: key vectors ({col_vec_key}) vs incident_number, dan run_id={emb_run_id}."
    )
    st.stop()

merged = df_mem.merge(df_vec, left_on="incident_number", right_on="key_ticket", how="inner")
if merged.empty:
    st.error("Join members ↔ embedding_vectors menghasilkan 0 baris (key tidak cocok).")
    st.write("Kolom vectors yang dipakai sebagai key:", col_vec_key)
    st.write("Contoh incident_number (members):", df_mem["incident_number"].head(3).tolist())
    st.write("Contoh key_ticket (vectors):", df_vec["key_ticket"].head(3).tolist())
    st.stop()

# parse embedding_json + label (pakai is_noise untuk memaksa noise -> -1)
emb_list: List[List[float]] = []
lab_list: List[int] = []

for _, r in merged.iterrows():
    obj = safe_json_loads(r["embedding_json"])
    if isinstance(obj, list) and len(obj) > 0:
        cid = int(r["cluster_id"])
        if bool(r.get("is_noise", False)):
            cid = -1
        emb_list.append(obj)
        lab_list.append(cid)

if len(emb_list) < 10:
    st.error("Embedding terlalu sedikit setelah parsing. Pastikan embedding_json berupa list angka.")
    st.stop()

# penting: float64 agar tidak warning "expected double_t but got float"
X = np.asarray(emb_list, dtype=np.float64)
labels = np.asarray(lab_list, dtype=np.int32)
feature_mode = f"Embedding ({X.shape[1]} dim) | emb_run_id={emb_run_id}"


# ======================================================
# 📊 KPI & DBCV (overall vs tanpa noise)
# ======================================================
n_points = int(X.shape[0])
n_noise = int((labels == -1).sum())
n_non_noise = n_points - n_noise
uniq = set(labels.tolist())
n_clusters = int(len(uniq) - (1 if -1 in uniq else 0))
coverage = (n_non_noise / n_points) if n_points else 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Mode fitur", feature_mode)
k2.metric("Jumlah titik dihitung", f"{n_points:,}")
k3.metric("Cluster (non-noise)", f"{n_clusters:,}")
k4.metric("Noise (-1)", f"{n_noise:,}")

k5, k6, k7 = st.columns(3)
k5.metric("Coverage non-noise", f"{coverage:.2%}")
k6.metric("DBCV (overall)", "-")
k7.metric("DBCV (tanpa noise)", "-")

if metric in ("euclidean", "manhattan") and X.shape[1] >= 256 and n_points > 20000:
    st.warning("DBCV pada embedding dimensi tinggi dengan >20k titik bisa berat. Pertimbangkan turunkan max_points.")

sizes = (
    pd.Series(labels)
    .value_counts()
    .rename_axis("cluster_id")
    .reset_index(name="n")
    .sort_values(["cluster_id"])
)

with st.spinner("Menghitung DBCV (overall & tanpa noise)..."):
    dbcv_overall = None
    dbcv_no_noise = None
    err = None
    try:
        # 1) DBCV OVERALL (termasuk noise)
        if n_non_noise < 5 or n_clusters < 2:
            err = (
                "DBCV membutuhkan struktur klaster memadai. "
                f"Saat ini non-noise={n_non_noise}, n_clusters={n_clusters}. "
                "Coba modeling_id lain atau set parameter HDBSCAN agar menghasilkan >1 klaster non-noise."
            )
        else:
            dbcv_overall = compute_dbcv(X, labels, metric=metric)

        # 2) DBCV TANPA NOISE (hanya anggota cluster valid)
        mask = labels != -1
        X_nn = X[mask]
        y_nn = labels[mask]
        uniq_nn = np.unique(y_nn)
        if len(y_nn) >= 5 and len(uniq_nn) >= 2:
            dbcv_no_noise = compute_dbcv(X_nn, y_nn, metric=metric)

    except Exception as e:
        err = str(e)

# Update KPI DBCV
k6.metric("DBCV (overall)", fmt_float(dbcv_overall, 4) if dbcv_overall is not None else "-")
k7.metric("DBCV (tanpa noise)", fmt_float(dbcv_no_noise, 4) if dbcv_no_noise is not None else "-")

if err:
    st.warning(err)
else:
    if dbcv_overall is not None:
        st.success(f"DBCV (overall) = **{fmt_float(dbcv_overall, 4)}**")
        st.caption("Overall menilai kualitas struktur global termasuk penalti dari noise.")
    if dbcv_no_noise is not None:
        st.info(f"DBCV (tanpa noise) = **{fmt_float(dbcv_no_noise, 4)}**")
        st.caption("Tanpa noise menilai kualitas internal cluster yang dianggap valid oleh HDBSCAN.")


# ======================================================
# 📈 Ringkasan distribusi ukuran cluster
# ======================================================
st.subheader("Distribusi Ukuran Cluster")
cL, cR = st.columns([1, 1])

with cL:
    st.dataframe(
        sizes.rename(columns={"cluster_id": "cluster_id", "n": "jumlah"}),
        use_container_width=True,
        height=340,
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
            .properties(height=320)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        st.info("Tidak ada cluster non-noise (semua noise).")


# ======================================================
# 🧾 Ringkasan tabel clusters (opsional jika tersedia)
# ======================================================
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
            avg_outlier_score
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


# ======================================================
# 📝 Catatan singkat
# ======================================================
with st.expander("Catatan metodologis (untuk tesis)", expanded=False):
    st.markdown(
        """
- DBCV adalah metrik validasi internal untuk **density-based clustering** yang menilai kualitas klaster dari aspek **kepadatan** dan **pemisahan**.
- Pada halaman ini, evaluasi dilakukan dalam dua skenario:
  1) **DBCV (overall)**: seluruh data termasuk noise, sehingga mencerminkan kualitas struktur global + penalti noise.
  2) **DBCV (tanpa noise)**: hanya anggota cluster valid, sehingga menilai kualitas internal cluster yang dipertahankan HDBSCAN.
- Coverage non-noise menunjukkan proporsi data yang berhasil dikelompokkan menjadi cluster (bukan noise).
        """
    )
