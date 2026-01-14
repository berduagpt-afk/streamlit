# pages/labeling/labeling_topic_cluster.py
# ============================================================
# 🏷️ Topic Labeling per Cluster (Nomor 2B)
# Output:
# - lasis_djp.cluster_topic_labels
#   top_terms_json: list of {"kata": <str>, "score": <float>}
#
# Catatan:
# - Hindari penggunaan { } pada komentar DDL untuk mencegah parsing/formatting tak sengaja
# - time_col dibuat NOT NULL DEFAULT '' agar aman jadi bagian PK/ON CONFLICT
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer

# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"

# Input labels table (hasil pelabelan biner)
T_LABEL = f"{SCHEMA}.incident_labeling_results"

# Output topic label table
T_TOPIC = f"{SCHEMA}.cluster_topic_labels"


# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )


def read_df(engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def exec_sql(engine, sql: str, params: dict | None = None):
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def ensure_topic_table(engine):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {T_TOPIC} (
        created_at timestamptz NOT NULL DEFAULT now(),

        -- run identity
        jenis_pendekatan text NOT NULL,                 -- 'sintaksis' / 'semantik'
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        time_col text NOT NULL DEFAULT '',              -- '' untuk sintaksis
        cluster_id bigint NOT NULL,

        -- stats (copy from labeling aggregation)
        n_member_cluster bigint NOT NULL,
        n_episode_cluster bigint NOT NULL,

        -- topic outputs
        top_terms_json jsonb NOT NULL,                  -- list berisi pasangan kata dan score
        label_topic text NULL,                          -- optional: label tema (auto / manual)
        sample_incidents_json jsonb NULL,               -- contoh incident_number
        sample_text_json jsonb NULL,                    -- contoh teks (sedikit saja)
        params_json jsonb NOT NULL,                     -- param vectorizer + sumber teks

        CONSTRAINT cluster_topic_labels_pkey
            PRIMARY KEY (jenis_pendekatan, modeling_id, window_days, time_col, cluster_id)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_topic_modeling ON {T_TOPIC} (modeling_id, window_days);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_topic_cluster ON {T_TOPIC} (cluster_id);"))


# ======================================================
# 🧠 Core helpers
# ======================================================
def safe_ident(name: str) -> str:
    """
    Sanitizer sederhana untuk identifier SQL (schema.table / table / column).
    """
    n = name.strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$")
    if not n or any(c not in allowed for c in n):
        raise ValueError(
            "Nama tabel/kolom mengandung karakter tidak valid. Gunakan huruf/angka/_ dan opsional schema.table."
        )
    return n


def build_top_kata_for_cluster(
    texts: List[str],
    top_k: int,
    ngram_min: int,
    ngram_max: int,
    min_df: int,
    max_df: float,
    max_features: int,
) -> List[Dict]:
    """
    Fit TF-IDF dalam satu cluster, lalu ambil kata teratas berdasarkan jumlah bobot TF-IDF.
    Output: list of {"kata": <str>, "score": <float>}
    """
    docs = [t for t in texts if isinstance(t, str) and t.strip()]
    if len(docs) < 2:
        return []

    vec = TfidfVectorizer(
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        lowercase=True,
    )
    X = vec.fit_transform(docs)
    if X.shape[1] == 0:
        return []

    scores = np.asarray(X.sum(axis=0)).ravel()
    features = np.array(vec.get_feature_names_out())

    k = min(top_k, len(scores))
    idx = np.argpartition(-scores, kth=k - 1)[:k]
    idx = idx[np.argsort(-scores[idx])]

    out: List[Dict] = []
    for i in idx:
        out.append({"kata": str(features[i]), "score": float(scores[i])})
    return out


def auto_label_from_kata(top_kata: List[Dict], max_words: int = 6) -> str:
    if not top_kata:
        return ""
    kata_list = [d.get("kata", "") for d in top_kata[:max_words] if d.get("kata")]
    return ", ".join(kata_list)


# ======================================================
# 🧭 UI
# ======================================================
st.title("🏷️ Topic Labeling per Cluster (2B)")
st.caption("Membuat label tema cluster berdasarkan keyword TF-IDF dari anggota cluster.")

engine = get_engine()
ensure_topic_table(engine)

# --------- Select run from labeling table ----------
st.subheader("1) Pilih Run Pelabelan (incident_labeling_results)")

runs = read_df(
    engine,
    f"""
    SELECT DISTINCT
      jenis_pendekatan,
      modeling_id,
      window_days,
      time_col,
      COUNT(*) AS n_rows,
      COUNT(DISTINCT cluster_id) AS n_clusters,
      MAX(run_time) AS last_run_time
    FROM {T_LABEL}
    GROUP BY 1,2,3,4
    ORDER BY last_run_time DESC
    LIMIT 300
    """,
)

if runs.empty:
    st.error(f"Tabel {T_LABEL} kosong / belum ada hasil pelabelan. Jalankan pelabelan dulu.")
    st.stop()

run_label = runs.apply(
    lambda r: (
        f"{r['jenis_pendekatan']} | modeling={r['modeling_id']} | w={r['window_days']} | time_col='{r['time_col']}' "
        f"| rows={r['n_rows']} | clusters={r['n_clusters']} | last={r['last_run_time']}"
    ),
    axis=1,
)
idx = st.selectbox("Pilih run", options=list(range(len(runs))), format_func=lambda i: run_label.iloc[i])

jenis_pendekatan = str(runs.loc[idx, "jenis_pendekatan"])
modeling_id = str(runs.loc[idx, "modeling_id"])
window_days = int(runs.loc[idx, "window_days"])
time_col = str(runs.loc[idx, "time_col"] or "")

st.info(f"Run terpilih: {jenis_pendekatan} | modeling_id={modeling_id} | window_days={window_days} | time_col='{time_col}'")

st.markdown("---")

# --------- Text source selection ----------
st.subheader("2) Sumber Teks untuk Ekstraksi Topik")
st.caption("Masukkan tabel & kolom yang berisi teks (disarankan hasil normalisasi).")

default_table = f"{SCHEMA}.incident_normalized"
default_col_incident = "incident_number"
default_col_text = "text_norm"  # sesuaikan dengan kolom Anda

colL, colR = st.columns(2)
with colL:
    src_table = st.text_input("Nama tabel sumber teks", value=default_table)
    src_incident_col = st.text_input("Kolom ID tiket (incident_number)", value=default_col_incident)
with colR:
    src_text_col = st.text_input("Kolom teks (untuk TF-IDF)", value=default_col_text)
    extra_where = st.text_input("Filter tambahan (SQL WHERE tanpa kata WHERE) - opsional", value="")

try:
    src_table_safe = safe_ident(src_table)
    src_incident_col_safe = safe_ident(src_incident_col)
    src_text_col_safe = safe_ident(src_text_col)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.markdown("---")

# --------- TF-IDF params ----------
st.subheader("3) Parameter TF-IDF per Cluster")

c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
with c1:
    top_k = st.number_input("Top kata (K)", min_value=3, max_value=50, value=10, step=1)
with c2:
    ngram_min = st.number_input("ngram min", min_value=1, max_value=3, value=1, step=1)
with c3:
    ngram_max = st.number_input("ngram max", min_value=1, max_value=3, value=2, step=1)
with c4:
    min_df = st.number_input("min_df", min_value=1, max_value=50, value=2, step=1)
with c5:
    max_features = st.number_input("max_features", min_value=500, max_value=50000, value=10000, step=500)

max_df = st.slider("max_df (proporsi dokumen)", min_value=0.50, max_value=1.00, value=0.95, step=0.01)

st.markdown("---")

# --------- Cluster selection controls ----------
st.subheader("4) Pilih Cluster yang akan diproses")

colA, colB, colC = st.columns([1, 1, 1])
with colA:
    only_recurrent = st.checkbox("Hanya cluster dengan label_berulang=1", value=True)
with colB:
    top_n_clusters = st.number_input("Top-N cluster terbesar; 0 = semua", min_value=0, value=100, step=10)
with colC:
    replace_existing = st.checkbox("Replace hasil topic labeling untuk run ini", value=False)

do_write = st.checkbox("Tulis hasil ke database", value=True)

st.markdown("---")


def load_clusters_to_process() -> pd.DataFrame:
    where = """
      jenis_pendekatan = :jp
      AND modeling_id = CAST(:mid AS uuid)
      AND window_days = :w
      AND time_col = :tc
    """
    sql = f"""
    WITH base AS (
      SELECT
        cluster_id,
        MAX(n_member_cluster) AS n_member_cluster,
        MAX(n_episode_cluster) AS n_episode_cluster,
        MAX(label_berulang) AS max_label
      FROM {T_LABEL}
      WHERE {where}
      GROUP BY cluster_id
    )
    SELECT cluster_id, n_member_cluster, n_episode_cluster, max_label
    FROM base
    {"WHERE max_label=1" if only_recurrent else ""}
    ORDER BY n_member_cluster DESC
    {"LIMIT :topn" if int(top_n_clusters) > 0 else ""}
    """
    params = {"jp": jenis_pendekatan, "mid": modeling_id, "w": window_days, "tc": time_col}
    if int(top_n_clusters) > 0:
        params["topn"] = int(top_n_clusters)
    return read_df(engine, sql, params)


def load_texts_for_clusters(cluster_ids: List[int]) -> pd.DataFrame:
    if not cluster_ids:
        return pd.DataFrame()

    where_run = """
      l.jenis_pendekatan = :jp
      AND l.modeling_id = CAST(:mid AS uuid)
      AND l.window_days = :w
      AND l.time_col = :tc
      AND l.cluster_id = ANY(:cluster_ids)
    """
    extra = f" AND ({extra_where})" if extra_where.strip() else ""

    sql = f"""
    SELECT
      l.cluster_id,
      l.incident_number,
      s.{src_text_col_safe} AS text_value
    FROM {T_LABEL} l
    LEFT JOIN {src_table_safe} s
      ON s.{src_incident_col_safe} = l.incident_number
    WHERE {where_run}
      {extra}
    """
    params = {"jp": jenis_pendekatan, "mid": modeling_id, "w": window_days, "tc": time_col, "cluster_ids": cluster_ids}
    return read_df(engine, sql, params)


def upsert_topic_rows(rows: List[Dict]):
    if not rows:
        return
    sql = f"""
    INSERT INTO {T_TOPIC} (
      jenis_pendekatan, modeling_id, window_days, time_col, cluster_id,
      n_member_cluster, n_episode_cluster,
      top_terms_json, label_topic, sample_incidents_json, sample_text_json,
      params_json
    )
    VALUES (
      :jp, CAST(:mid AS uuid), :w, :tc, :cid,
      :n_member, :n_episode,
      CAST(:top_terms AS jsonb), :label_topic,
      CAST(:sample_incidents AS jsonb),
      CAST(:sample_text AS jsonb),
      CAST(:params AS jsonb)
    )
    ON CONFLICT (jenis_pendekatan, modeling_id, window_days, time_col, cluster_id)
    DO UPDATE SET
      created_at = now(),
      n_member_cluster = EXCLUDED.n_member_cluster,
      n_episode_cluster = EXCLUDED.n_episode_cluster,
      top_terms_json = EXCLUDED.top_terms_json,
      label_topic = EXCLUDED.label_topic,
      sample_incidents_json = EXCLUDED.sample_incidents_json,
      sample_text_json = EXCLUDED.sample_text_json,
      params_json = EXCLUDED.params_json;
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


# ======================================================
# ▶️ Run topic labeling
# ======================================================
if st.button("🚀 Generate Topic Label per Cluster", type="primary"):
    with st.spinner("Menyiapkan daftar cluster..."):
        df_clusters = load_clusters_to_process()

    if df_clusters.empty:
        st.warning("Tidak ada cluster yang memenuhi filter.")
        st.stop()

    st.success(f"Cluster terpilih: {len(df_clusters):,}")
    st.dataframe(df_clusters, use_container_width=True, height=220)

    if replace_existing and do_write:
        with st.spinner("Menghapus hasil topic labeling sebelumnya untuk run ini..."):
            exec_sql(
                engine,
                f"""
                DELETE FROM {T_TOPIC}
                WHERE jenis_pendekatan = :jp
                  AND modeling_id = CAST(:mid AS uuid)
                  AND window_days = :w
                  AND time_col = :tc
                """,
                {"jp": jenis_pendekatan, "mid": modeling_id, "w": window_days, "tc": time_col},
            )
        st.info("Replace: data topic labeling lama sudah dihapus.")

    cluster_ids = [int(x) for x in df_clusters["cluster_id"].tolist()]
    with st.spinner("Mengambil teks anggota cluster (join ke sumber teks)..."):
        df_text = load_texts_for_clusters(cluster_ids)

    if df_text.empty:
        st.error(
            "Tidak ada teks yang berhasil diambil. Cek nama tabel/kolom sumber teks, "
            "atau pastikan incident_number dapat di-join."
        )
        st.stop()

    n_missing = int(df_text["text_value"].isna().sum())
    st.write(
        {
            "rows_joined": int(len(df_text)),
            "unique_clusters": int(df_text["cluster_id"].nunique()),
            "missing_text_rows": n_missing,
        }
    )
    if n_missing > 0:
        st.warning("Ada text_value NULL. Topic untuk beberapa cluster bisa kurang optimal.")

    params_json = {
        "source_table": src_table_safe,
        "source_incident_col": src_incident_col_safe,
        "source_text_col": src_text_col_safe,
        "extra_where": extra_where.strip(),
        "tfidf": {
            "top_k": int(top_k),
            "ngram_range": [int(ngram_min), int(ngram_max)],
            "min_df": int(min_df),
            "max_df": float(max_df),
            "max_features": int(max_features),
            "method": "cluster_local_tfidf_sum",
        },
        "run": {
            "jenis_pendekatan": jenis_pendekatan,
            "modeling_id": modeling_id,
            "window_days": int(window_days),
            "time_col": time_col,
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        },
    }

    # Map stats per cluster
    stats_map = (
        df_clusters.set_index("cluster_id")[["n_member_cluster", "n_episode_cluster"]]
        .to_dict(orient="index")
    )

    results = []
    progress = st.progress(0)
    status = st.empty()

    group = df_text.groupby("cluster_id", sort=False)
    cluster_list = list(group.groups.keys())
    total = len(cluster_list)

    for i, cid in enumerate(cluster_list, start=1):
        status.text(f"Processing cluster {cid} ({i}/{total})")
        sub = group.get_group(cid)

        texts = sub["text_value"].dropna().astype(str).tolist()
        incs = sub["incident_number"].dropna().astype(str).tolist()

        top_kata = build_top_kata_for_cluster(
            texts=texts,
            top_k=int(top_k),
            ngram_min=int(ngram_min),
            ngram_max=int(ngram_max),
            min_df=int(min_df),
            max_df=float(max_df),
            max_features=int(max_features),
        )

        sample_incidents = incs[:10]
        sample_text = texts[:3]

        label_topic = auto_label_from_kata(top_kata, max_words=6) or None

        stt = stats_map.get(cid, {"n_member_cluster": len(incs), "n_episode_cluster": 0})

        results.append(
            {
                "jp": jenis_pendekatan,
                "mid": modeling_id,
                "w": int(window_days),
                "tc": time_col,
                "cid": int(cid),
                "n_member": int(stt["n_member_cluster"]),
                "n_episode": int(stt["n_episode_cluster"]),
                "top_terms": json.dumps(top_kata),  # list of {"kata","score"}
                "label_topic": label_topic,
                "sample_incidents": json.dumps(sample_incidents),
                "sample_text": json.dumps(sample_text),
                "params": json.dumps(params_json),
            }
        )

        progress.progress(int((i / total) * 100))

    status.text("Selesai memproses semua cluster.")
    progress.progress(100)

    # Preview output (no NameError)
    df_out_preview = pd.DataFrame(
        [
            {
                "cluster_id": r["cid"],
                "n_member": r["n_member"],
                "n_episode": r["n_episode"],
                "label_topic": r["label_topic"],
                "top_kata": ", ".join([k["kata"] for k in json.loads(r["top_terms"])[:10]]),
            }
            for r in results
        ]
    )

    st.subheader("Preview hasil topic labeling")
    st.dataframe(df_out_preview, use_container_width=True, height=420)

    if do_write:
        with st.spinner("Menyimpan hasil ke database (upsert)..."):
            upsert_topic_rows(results)
        st.success(f"✅ Tersimpan ke {T_TOPIC}")

        df_recap = read_df(
            engine,
            f"""
            SELECT
              COUNT(*) AS n_clusters_labeled,
              MAX(created_at) AS last_created_at
            FROM {T_TOPIC}
            WHERE jenis_pendekatan = :jp
              AND modeling_id = CAST(:mid AS uuid)
              AND window_days = :w
              AND time_col = :tc
            """,
            {"jp": jenis_pendekatan, "mid": modeling_id, "w": window_days, "tc": time_col},
        )
        st.subheader("Rekap tersimpan")
        st.dataframe(df_recap, use_container_width=True)
    else:
        st.info("Mode tulis dimatikan. Hasil hanya preview.")
