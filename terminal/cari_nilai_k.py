"""
Eksperimen Sensitivitas kNN untuk Clustering TF-IDF
Graph-based clustering (kNN + cosine similarity + connected components)

Input:
- lasis_djp.incident_tfidf_runs
- lasis_djp.incident_tfidf_vectors

Output:
- CSV hasil eksperimen kNN
"""

import json
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score, davies_bouldin_score


# ======================================================
# 🔐 KONFIGURASI DATABASE (HARDCODED)
# ======================================================
DB = {
    "host": "localhost",
    "port": 5432,
    "database": "incident_djp",
    "user": "postgres",
    "password": "admin*123",
}


# ======================================================
# ⚙️ KONFIGURASI EKSPERIMEN
# ======================================================
@dataclass
class Config:
    schema: str = "lasis_djp"
    t_runs: str = "incident_tfidf_runs"
    t_vecs: str = "incident_tfidf_vectors"

    # Grid kNN
    k_grid: Tuple[int, ...] = (10,20, 25, 30)

    # Ambang cosine similarity (edge graph)
    sim_threshold: float = 0.70

    # Batasi jumlah dokumen (online-friendly)
    max_docs: Optional[int] = 60000

    # Sampling untuk evaluasi (Silhouette/DBI dihitung pada sample)
    eval_sample_size: int = 60000

    # Reduksi dimensi untuk evaluasi (hemat runtime)
    svd_dim: int = 50

    seed: int = 42

    # ---------- PATCH BARU (DBI SAFE) ----------
    # Jika jumlah cluster sangat besar, DBI standar akan meledak (O(K^2)).
    # Maka DBI dihitung pada subset cluster "cukup besar".
    dbi_max_clusters: int = 400      # top-K cluster terbesar yang dipakai untuk DBI
    dbi_min_cluster_size: int = 5    # buang cluster kecil (termasuk mayoritas singleton)
    # ------------------------------------------


CFG = Config()


# ======================================================
# 🔌 KONEKSI DATABASE
# ======================================================
def get_engine():
    url = (
        f"postgresql+psycopg2://{DB['user']}:{DB['password']}"
        f"@{DB['host']}:{DB['port']}/{DB['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


# ======================================================
# 📥 LOAD DATA
# ======================================================
def fetch_latest_run_id(engine) -> str:
    q = text(f"""
        SELECT run_id
        FROM {CFG.schema}.{CFG.t_runs}
        ORDER BY run_time DESC
        LIMIT 1
    """)
    with engine.connect() as conn:
        run_id = conn.execute(q).scalar()
    if not run_id:
        raise RuntimeError("run_id tidak ditemukan.")
    return run_id


def fetch_feature_names(engine, run_id: str) -> List[str]:
    q = text(f"""
        SELECT feature_names_json
        FROM {CFG.schema}.{CFG.t_runs}
        WHERE run_id = :run_id
    """)
    with engine.connect() as conn:
        row = conn.execute(q, {"run_id": run_id}).fetchone()
    if not row or row[0] is None:
        raise RuntimeError("feature_names_json kosong / tidak ditemukan.")
    return list(row[0])


def fetch_tfidf(engine, run_id: str, max_docs: Optional[int]) -> pd.DataFrame:
    limit_sql = f"LIMIT {max_docs}" if max_docs else ""
    q = text(f"""
        SELECT incident_number, tfidf_json
        FROM {CFG.schema}.{CFG.t_vecs}
        WHERE run_id = :run_id
          AND tfidf_json IS NOT NULL
        {limit_sql}
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params={"run_id": run_id})
    if df.empty:
        raise RuntimeError("Tidak ada data tfidf_json untuk run_id ini.")
    return df


# ======================================================
# 🧱 BUILD SPARSE TF-IDF MATRIX
# ======================================================
def build_sparse_matrix(df: pd.DataFrame, feature_names: List[str]) -> csr_matrix:
    vocab = {t: i for i, t in enumerate(feature_names)}

    indptr, indices, data = [0], [], []

    for row in df["tfidf_json"]:
        if row is None:
            row = {}
        if isinstance(row, str):
            try:
                row = json.loads(row)
            except Exception:
                row = {}

        for term, val in row.items():
            j = vocab.get(term)
            if j is None:
                continue
            try:
                v = float(val)
            except Exception:
                continue
            if v != 0.0:
                indices.append(j)
                data.append(v)

        indptr.append(len(indices))

    X = csr_matrix(
        (np.array(data, dtype=np.float32), np.array(indices, dtype=np.int32), np.array(indptr, dtype=np.int64)),
        shape=(len(df), len(feature_names)),
        dtype=np.float32,
    )

    # L2 normalization (WAJIB)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A1
    norms[norms == 0] = 1.0

    # ⚠️ PENTING: paksa kembali ke CSR (agar slicing aman)
    X = X.multiply(1.0 / norms[:, None]).tocsr()

    return X


# ======================================================
# 🔗 kNN GRAPH + CONNECTED COMPONENTS
# ======================================================
def cluster_knn(X: csr_matrix, k: int, sim_threshold: float) -> np.ndarray:
    nn = NearestNeighbors(
        n_neighbors=min(k + 1, X.shape[0]),
        metric="cosine",
        algorithm="brute",
        n_jobs=-1
    )
    nn.fit(X)

    dists, neighs = nn.kneighbors(X)
    dist_thr = 1.0 - sim_threshold

    rows, cols = [], []

    for i in range(X.shape[0]):
        for j in range(1, neighs.shape[1]):
            if dists[i, j] <= dist_thr:
                rows.append(i)
                cols.append(int(neighs[i, j]))

    if not rows:
        return np.arange(X.shape[0], dtype=np.int32)

    A = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(X.shape[0], X.shape[0]))
    A = A.maximum(A.T)

    _, labels = connected_components(A, directed=False)
    return labels.astype(np.int32)


# ======================================================
# 📊 EVALUASI INTERNAL (PATCH: DBI SAFE)
# ======================================================
def _pick_clusters_for_dbi(ys: np.ndarray) -> Set[int]:
    """
    Pilih subset cluster untuk DBI agar tidak O(K^2) meledak memori.
    - Buang cluster < dbi_min_cluster_size
    - Ambil top-K cluster terbesar (dbi_max_clusters)
    """
    u, c = np.unique(ys, return_counts=True)

    # filter minimal ukuran cluster
    mask = c >= CFG.dbi_min_cluster_size
    u_big = u[mask]
    c_big = c[mask]

    if len(u_big) < 2:
        return set()

    # kalau masih kebanyakan, ambil top-K terbesar
    if len(u_big) > CFG.dbi_max_clusters:
        top_idx = np.argsort(-c_big)[:CFG.dbi_max_clusters]
        u_big = u_big[top_idx]

    return set(int(x) for x in u_big)


def evaluate(X: csr_matrix, labels: np.ndarray):
    # Reproducible sampling
    rng = np.random.default_rng(CFG.seed)

    uniq_all, cnt_all = np.unique(labels, return_counts=True)
    if len(uniq_all) < 2:
        return np.nan, np.nan

    # Sample untuk evaluasi
    n = len(labels)
    s = min(CFG.eval_sample_size, n)
    idx = rng.choice(n, size=s, replace=False)

    Xs = X[idx]
    ys = labels[idx]

    # Butuh >=2 cluster di sample
    if len(np.unique(ys)) < 2:
        return np.nan, np.nan

    # SVD dim guard
    n_features = Xs.shape[1]
    n_comp = min(CFG.svd_dim, n_features - 1) if n_features > 1 else 1
    if n_comp < 2:
        return np.nan, np.nan

    svd = TruncatedSVD(n_components=n_comp, random_state=CFG.seed)
    Z = svd.fit_transform(Xs)

    # Silhouette: cosine
    sil = silhouette_score(Z, ys, metric="cosine")

    # DBI: SAFE mode (subset cluster)
    keep_clusters = _pick_clusters_for_dbi(ys)
    if not keep_clusters:
        dbi = np.nan
    else:
        keep_mask = np.array([lab in keep_clusters for lab in ys], dtype=bool)
        Z_dbi = Z[keep_mask]
        y_dbi = ys[keep_mask]

        if len(np.unique(y_dbi)) < 2:
            dbi = np.nan
        else:
            dbi = davies_bouldin_score(Z_dbi, y_dbi)

    return float(sil), (float(dbi) if dbi == dbi else np.nan)


# ======================================================
# 🚀 MAIN EKSPERIMEN
# ======================================================
def run_knn_experiment():
    random.seed(CFG.seed)
    np.random.seed(CFG.seed)

    engine = get_engine()
    run_id = fetch_latest_run_id(engine)

    print(f"[INFO] run_id = {run_id}")

    feature_names = fetch_feature_names(engine, run_id)
    df = fetch_tfidf(engine, run_id, CFG.max_docs)

    print(f"[INFO] Total dokumen: {len(df):,}")
    print(f"[INFO] Total fitur: {len(feature_names):,}")

    X = build_sparse_matrix(df, feature_names)
    print(f"[INFO] Sparse matrix: shape={X.shape}, nnz={X.nnz:,}, type={type(X).__name__}")

    results = []

    for k in CFG.k_grid:
        labels = cluster_knn(X, k, CFG.sim_threshold)

        uniq, cnt = np.unique(labels, return_counts=True)
        n_clusters = len(uniq)
        n_singletons = int((cnt == 1).sum())
        max_cluster = int(cnt.max()) if len(cnt) else 0

        sil, dbi = evaluate(X, labels)

        results.append({
            "run_id": run_id,
            "k": k,
            "sim_threshold": CFG.sim_threshold,
            "n_docs": len(labels),
            "n_clusters_all": n_clusters,
            "n_singletons": n_singletons,
            "max_cluster_size": max_cluster,
            "silhouette": sil,
            "davies_bouldin": dbi,
            "dbi_note": (
                f"DBI computed on subset (min_size={CFG.dbi_min_cluster_size}, topK={CFG.dbi_max_clusters})"
                if (dbi == dbi) else
                "DBI skipped (insufficient large clusters in sample)"
            )
        })

        print(
            f"k={k:>2} | clusters={n_clusters:>6,} | singletons={n_singletons:>6,} | "
            f"max={max_cluster:>6,} | sil={sil:.4f} | dbi={'NaN' if dbi != dbi else f'{dbi:.4f}'}"
        )

    df_res = pd.DataFrame(results)
    df_res.to_csv("knn_sensitivity_results.csv", index=False)

    print("\n[OK] Hasil disimpan ke knn_sensitivity_results.csv")
    return df_res


if __name__ == "__main__":
    run_knn_experiment()
