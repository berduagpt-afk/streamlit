# pages/visualisasi_temporal_connected_components.py
# ======================================================
# Temporal Connected Components (Streamlit) — FULL (type-safe)
#
# FIX penting:
# ✅ cache_data: parameter _engine (underscore) agar tidak di-hash
# ✅ run_id & modeling_id: aman terhadap tipe kolom (text/uuid)
#    pakai v.run_id::text = :rid dan m.modeling_id::text = :mid
# ======================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


# ======================================================
# 🔐 Guard login (sesuaikan dengan app kamu)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


SCHEMA = "lasis_djp"
T_RUNS = "modeling_sintaksis_runs"
T_MEMBERS = "modeling_sintaksis_members"
T_TFIDF = "incident_tfidf_vectors"
NOISE_ID = -1


# ======================================================
# 🔌 DB Connection (secrets.toml)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


# ======================================================
# ✅ Build CSR matrix from tfidf_json list
# ======================================================
def build_csr_matrix(tfidf_json_list: list[object]) -> tuple[csr_matrix, dict[str, int]]:
    vocab: dict[str, int] = {}
    docs: list[dict[str, float]] = []

    for x in tfidf_json_list:
        d = x if isinstance(x, dict) else {}
        docs.append(d)
        for term in d.keys():
            if term not in vocab:
                vocab[term] = len(vocab)

    indptr = [0]
    indices: list[int] = []
    data: list[float] = []

    for d in docs:
        for term, val in d.items():
            j = vocab.get(term)
            if j is None:
                continue
            try:
                fv = float(val)
            except Exception:
                continue
            if fv == 0.0:
                continue
            indices.append(j)
            data.append(fv)
        indptr.append(len(indices))

    X = csr_matrix(
        (np.array(data, dtype=np.float32),
         np.array(indices, dtype=np.int32),
         np.array(indptr, dtype=np.int32)),
        shape=(len(docs), len(vocab)),
        dtype=np.float32
    )
    return X, vocab


# ======================================================
# 📦 DB Loaders (cached) — NOTE: _engine (underscore)
# ======================================================
@st.cache_data(show_spinner=False)
def load_runs(_engine: Engine) -> pd.DataFrame:
    sql = f"""
    SELECT modeling_id, run_time, approach,
           tfidf_run_id, threshold, window_days, knn_k, min_cluster_size,
           n_rows, n_clusters_recurring, n_noise_tickets
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT 200
    """
    df = pd.read_sql(text(sql), _engine)
    df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_clusters_for_model(_engine: Engine, modeling_id: str) -> pd.DataFrame:
    # type-safe: m.modeling_id::text = :mid
    sql = f"""
    SELECT cluster_id, COUNT(*) AS cluster_size,
           MIN(tgl_submit) AS min_time, MAX(tgl_submit) AS max_time
    FROM {SCHEMA}.{T_MEMBERS}
    WHERE modeling_id::text = :mid
      AND cluster_id <> {NOISE_ID}
    GROUP BY cluster_id
    ORDER BY cluster_size DESC
    """
    df = pd.read_sql(text(sql), _engine, params={"mid": str(modeling_id)})
    df["min_time"] = pd.to_datetime(df["min_time"], errors="coerce")
    df["max_time"] = pd.to_datetime(df["max_time"], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_cluster_members_with_tfidf(
    _engine: Engine,
    modeling_id: str,
    cluster_id: int,
    tfidf_run_id: str,
) -> pd.DataFrame:
    # ✅ type-safe join:
    # - v.run_id::text = :rid  (aman untuk run_id text/uuid)
    # - m.modeling_id::text = :mid (aman untuk modeling_id text/uuid)
    sql = f"""
    SELECT
        m.incident_number,
        m.tgl_submit,
        m.site,
        m.assignee,
        m.modul,
        m.sub_modul,
        v.tfidf_json
    FROM {SCHEMA}.{T_MEMBERS} m
    JOIN {SCHEMA}.{T_TFIDF} v
      ON v.incident_number = m.incident_number
     AND v.run_id::text = :rid
    WHERE m.modeling_id::text = :mid
      AND m.cluster_id = :cid
    ORDER BY m.tgl_submit NULLS LAST
    """
    df = pd.read_sql(
        text(sql),
        _engine,
        params={"mid": str(modeling_id), "cid": int(cluster_id), "rid": str(tfidf_run_id)},
    )
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    return df


# ======================================================
# 🔗 Build edges (kNN + filter threshold + time window)
# ======================================================
def build_edges_knn(
    df: pd.DataFrame,
    X: csr_matrix,
    threshold: float,
    window_days: int,
    knn_k: int,
) -> pd.DataFrame:
    n = X.shape[0]
    if n < 2:
        return pd.DataFrame(columns=["src", "dst", "sim", "day_diff"])

    normalize(X, norm="l2", axis=1, copy=False)

    t_floor = df["tgl_submit"].dt.floor("D")
    t_days = np.full(n, -10**18, dtype=np.int64)
    mask = t_floor.notna().to_numpy()
    if mask.any():
        t_days[mask] = t_floor.to_numpy(dtype="datetime64[D]")[mask].astype(np.int64)

    k = int(max(2, min(int(knn_k), n)))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(X)

    dist, idx = nn.kneighbors(X, return_distance=True)
    sim = 1.0 - dist

    thr = float(threshold)
    win = int(window_days)

    edges = []
    for i in range(n):
        ti = int(t_days[i])
        for pos in range(1, k):
            j = int(idx[i, pos])
            s = float(sim[i, pos])
            if s < thr:
                continue
            tj = int(t_days[j])
            if ti < -10**17 or tj < -10**17:
                continue
            dd = int(abs(ti - tj))
            if dd > win:
                continue
            if i < j:
                edges.append((i, j, s, dd))

    return pd.DataFrame(edges, columns=["src", "dst", "sim", "day_diff"])


# ======================================================
# 📊 Plot Temporal Graph (Altair)
# ======================================================
def plot_temporal_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> alt.Chart:
    if not edges.empty:
        e = (
            edges.merge(nodes[["idx", "tgl_submit", "y"]], left_on="src", right_on="idx", how="left")
                 .rename(columns={"tgl_submit": "x", "y": "y1"})
                 .drop(columns=["idx"])
                 .merge(nodes[["idx", "tgl_submit", "y"]], left_on="dst", right_on="idx", how="left")
                 .rename(columns={"tgl_submit": "x2", "y": "y2"})
                 .drop(columns=["idx"])
        )
        edge_chart = alt.Chart(e).mark_rule().encode(
            x="x:T", y="y1:Q",
            x2="x2:T", y2="y2:Q",
            tooltip=[
                alt.Tooltip("sim:Q", format=".3f", title="cosine_sim"),
                alt.Tooltip("day_diff:Q", title="day_diff"),
                alt.Tooltip("x:T", title="tgl_src"),
                alt.Tooltip("x2:T", title="tgl_dst"),
            ],
        )
    else:
        edge_chart = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_rule()

    node_chart = alt.Chart(nodes).mark_circle().encode(
        x=alt.X("tgl_submit:T", title="Tanggal Submit"),
        y=alt.Y("y:Q", title="Urutan Tiket (dalam cluster)"),
        size=alt.Size("degree:Q", title="degree", scale=alt.Scale(range=[20, 650])),
        tooltip=[
            alt.Tooltip("incident_number:N"),
            alt.Tooltip("tgl_submit:T"),
            alt.Tooltip("degree:Q"),
        ],
    )
    return (edge_chart + node_chart).properties(height=560).interactive()


# ======================================================
# UI
# ======================================================
st.title("📈 Temporal Connected Components (Node–Edge Graph)")
st.caption(
    "Node = tiket insiden. Edge dibuat jika cosine similarity ≥ threshold dan |selisih hari| ≤ window_days. "
    "Grafik ini memperlihatkan efek *rantai koneksi* (connected components)."
)

engine = get_engine()
runs = load_runs(engine)

if runs.empty:
    st.warning("Tidak ada data pada modeling_sintaksis_runs.")
    st.stop()

runs = runs.copy()
runs["label"] = runs.apply(
    lambda r: (
        f"{str(r['run_time'])[:19]} | mid={r['modeling_id']} | "
        f"thr={r.get('threshold', None)} | win={r.get('window_days', None)} | k={r.get('knn_k', None)}"
    ),
    axis=1
)

pick = st.selectbox("Pilih modeling run", runs["label"].tolist(), index=0)
row = runs.loc[runs["label"] == pick].iloc[0]

modeling_id = str(row["modeling_id"])
tfidf_run_id = str(row["tfidf_run_id"]) if pd.notna(row["tfidf_run_id"]) else ""

st.sidebar.header("⚙️ Parameter Edge (Override)")
default_thr = float(row["threshold"]) if pd.notna(row.get("threshold")) else 0.80
default_win = int(row["window_days"]) if pd.notna(row.get("window_days")) else 30
default_k = int(row["knn_k"]) if pd.notna(row.get("knn_k")) else 25

thr = st.sidebar.slider("Cosine similarity threshold", 0.50, 0.99, default_thr, 0.01)
win = st.sidebar.slider("Temporal window (hari)", 1, 180, default_win, 1)
knn_k = st.sidebar.slider("kNN (neighbors)", 2, 120, default_k, 1)

st.sidebar.markdown("---")
max_nodes = st.sidebar.slider("Maks node ditampilkan (sampling)", 50, 3000, 700, 50)
seed = st.sidebar.number_input("Random seed", value=42, step=1)

cl = load_clusters_for_model(engine, modeling_id)
if cl.empty:
    st.warning("Tidak ada cluster recurring (cluster_id <> -1) untuk modeling_id ini.")
    st.stop()

cluster_id = st.selectbox(
    "Pilih cluster_id (urut dari terbesar)",
    cl["cluster_id"].astype(int).tolist(),
    index=0
)

cinfo = cl.loc[cl["cluster_id"] == cluster_id].iloc[0]
colA, colB, colC, colD = st.columns(4)
colA.metric("Cluster size", int(cinfo["cluster_size"]))
colB.metric("Min time", str(pd.to_datetime(cinfo["min_time"]))[:19])
colC.metric("Max time", str(pd.to_datetime(cinfo["max_time"]))[:19])
try:
    span_days = (pd.to_datetime(cinfo["max_time"]) - pd.to_datetime(cinfo["min_time"])).days
except Exception:
    span_days = None
colD.metric("Span days (observed)", span_days if span_days is not None else "-")

if not tfidf_run_id or tfidf_run_id.lower() == "none":
    st.error("Kolom tfidf_run_id pada modeling_sintaksis_runs kosong. Pastikan script offline menyimpan tfidf_run_id.")
    st.stop()

with st.spinner("Memuat node + tfidf_json..."):
    df = load_cluster_members_with_tfidf(engine, modeling_id, int(cluster_id), tfidf_run_id)

if df.empty:
    st.warning(
        "Join ke incident_tfidf_vectors tidak menghasilkan baris. "
        "Pastikan incident_number match dan tfidf_run_id sesuai."
    )
    st.stop()

df = df.sort_values("tgl_submit").dropna(subset=["incident_number"]).copy()

if len(df) > max_nodes:
    rs = np.random.RandomState(int(seed))
    keep = rs.choice(df.index.to_numpy(), size=int(max_nodes), replace=False)
    df = df.loc[np.sort(keep)].copy()
    df = df.sort_values("tgl_submit")
    st.info(f"Cluster terlalu besar, ditampilkan sampling {max_nodes} node.")

with st.spinner("Membangun matriks TF-IDF (CSR) dan edge kNN..."):
    X, vocab = build_csr_matrix(df["tfidf_json"].tolist())
    edges = build_edges_knn(df, X, threshold=thr, window_days=win, knn_k=knn_k)

nodes = df[["incident_number", "tgl_submit"]].copy().reset_index(drop=True)
nodes["idx"] = nodes.index.astype(int)
nodes["y"] = nodes.index.astype(int)

deg = np.zeros(len(nodes), dtype=int)
if not edges.empty:
    for a, b in zip(edges["src"].to_numpy(), edges["dst"].to_numpy()):
        deg[int(a)] += 1
        deg[int(b)] += 1
nodes["degree"] = deg

m1, m2, m3, m4 = st.columns(4)
m1.metric("Nodes", len(nodes))
m2.metric("Edges (filtered)", int(len(edges)))
m3.metric("Avg degree", float(deg.mean()) if len(nodes) else 0.0)
m4.metric("Max degree", int(deg.max()) if len(nodes) else 0)

st.subheader("Graf Temporal Connected Components")
st.altair_chart(plot_temporal_graph(nodes, edges), use_container_width=True)

with st.expander("🔎 Tiket penghubung (degree tertinggi)"):
    st.dataframe(nodes.sort_values("degree", ascending=False).head(20), use_container_width=True)

with st.expander("🔗 Daftar edge (pasangan tiket yang tersambung)"):
    if edges.empty:
        st.write("Tidak ada edge yang lolos threshold+window. Turunkan threshold atau naikkan kNN.")
    else:
        e2 = edges.copy()
        e2["src_incident"] = e2["src"].map(lambda i: nodes.loc[int(i), "incident_number"])
        e2["dst_incident"] = e2["dst"].map(lambda i: nodes.loc[int(i), "incident_number"])
        st.dataframe(
            e2[["src_incident", "dst_incident", "sim", "day_diff"]].sort_values("sim", ascending=False),
            use_container_width=True
        )
