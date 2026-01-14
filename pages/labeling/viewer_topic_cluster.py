# pages/labeling/viewer_edit_topic_label.py
# ============================================================
# 🔎 Viewer & ✏️ Edit Topic Label per Cluster
# - Membaca hasil topic labeling: lasis_djp.cluster_topic_labels
# - Drilldown: tampilkan contoh incident + teks per cluster
# - Edit: update kolom label_topic (manual override) + optional save notes
#
# Dependensi:
# - T_LABEL: lasis_djp.incident_labeling_results (untuk daftar incident per cluster)
# - T_TOPIC: lasis_djp.cluster_topic_labels (untuk top_kata & label_topic)
# - Source text table: default lasis_djp.incident_normalized (sesuaikan kolom)
#
# Catatan:
# - Menggunakan key JSON "kata" di top_terms_json
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"

T_LABEL = f"{SCHEMA}.incident_labeling_results"
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


def safe_ident(name: str) -> str:
    n = name.strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.$")
    if not n or any(c not in allowed for c in n):
        raise ValueError("Nama tabel/kolom mengandung karakter tidak valid. Gunakan huruf/angka/_ dan opsional schema.table.")
    return n


def json_top_kata_to_str(top_terms_json: Optional[object], k: int = 12) -> str:
    """
    top_terms_json berisi list dict [{"kata":..., "score":...}, ...]
    """
    if top_terms_json is None:
        return ""
    try:
        if isinstance(top_terms_json, str):
            obj = json.loads(top_terms_json)
        else:
            obj = top_terms_json
        if not isinstance(obj, list):
            return ""
        kata = []
        for d in obj[:k]:
            if isinstance(d, dict) and d.get("kata"):
                kata.append(str(d["kata"]))
        return ", ".join(kata)
    except Exception:
        return ""


# ======================================================
# 🧭 UI
# ======================================================
st.title("🔎 Viewer & ✏️ Edit Topic Label per Cluster")
st.caption("Lihat hasil topic labeling per cluster, drilldown contoh tiket, dan edit label_topic secara manual.")

engine = get_engine()

# ------------------------------------------------------
# 1) Pilih run yang tersedia di cluster_topic_labels
# ------------------------------------------------------
st.subheader("1) Pilih Run Topic Labeling")

runs = read_df(
    engine,
    f"""
    SELECT
      jenis_pendekatan,
      modeling_id,
      window_days,
      time_col,
      COUNT(*) AS n_clusters,
      MAX(created_at) AS last_created_at
    FROM {T_TOPIC}
    GROUP BY 1,2,3,4
    ORDER BY last_created_at DESC
    LIMIT 300
    """,
)

if runs.empty:
    st.error(f"Tabel {T_TOPIC} kosong. Jalankan halaman topic labeling (2B) terlebih dahulu.")
    st.stop()

run_label = runs.apply(
    lambda r: (
        f"{r['jenis_pendekatan']} | modeling={r['modeling_id']} | w={r['window_days']} | time_col='{r['time_col']}' "
        f"| clusters={r['n_clusters']} | last={r['last_created_at']}"
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

# ------------------------------------------------------
# 2) Set sumber teks untuk drilldown (opsional tapi disarankan)
# ------------------------------------------------------
st.subheader("2) Sumber Teks untuk Drilldown (contoh tiket)")
st.caption("Digunakan untuk menampilkan teks normalisasi dari tiket dalam cluster.")

default_table = f"{SCHEMA}.incident_normalized"
default_col_incident = "incident_number"
default_col_text = "text_norm"  # sesuaikan dengan kolom di DB Anda

colA, colB = st.columns(2)
with colA:
    src_table = st.text_input("Tabel sumber teks", value=default_table)
    src_incident_col = st.text_input("Kolom incident_number", value=default_col_incident)
with colB:
    src_text_col = st.text_input("Kolom teks", value=default_col_text)
    extra_where = st.text_input("Filter tambahan (SQL WHERE tanpa kata WHERE) - opsional", value="")

try:
    src_table_safe = safe_ident(src_table)
    src_incident_col_safe = safe_ident(src_incident_col)
    src_text_col_safe = safe_ident(src_text_col)
except ValueError as e:
    st.error(str(e))
    st.stop()

st.markdown("---")

# ------------------------------------------------------
# 3) Load daftar cluster untuk run ini
# ------------------------------------------------------
st.subheader("3) Daftar Cluster")

colX, colY, colZ = st.columns([1, 1, 1])
with colX:
    only_recurrent = st.checkbox("Hanya cluster recurrent (label_berulang=1)", value=True)
with colY:
    min_member_filter = st.number_input("Filter min n_member_cluster", min_value=0, value=0, step=10)
with colZ:
    limit_clusters = st.number_input("Limit cluster list", min_value=20, value=200, step=20)

sql_clusters = f"""
WITH rec AS (
  SELECT cluster_id, MAX(label_berulang) AS max_label
  FROM {T_LABEL}
  WHERE jenis_pendekatan = :jp
    AND modeling_id = CAST(:mid AS uuid)
    AND window_days = :w
    AND time_col = :tc
  GROUP BY cluster_id
)
SELECT
  t.cluster_id,
  t.n_member_cluster,
  t.n_episode_cluster,
  r.max_label AS label_berulang,
  t.label_topic,
  t.top_terms_json,
  t.created_at
FROM {T_TOPIC} t
LEFT JOIN rec r USING (cluster_id)
WHERE t.jenis_pendekatan = :jp
  AND t.modeling_id = CAST(:mid AS uuid)
  AND t.window_days = :w
  AND t.time_col = :tc
  {"AND COALESCE(r.max_label,0)=1" if only_recurrent else ""}
  AND t.n_member_cluster >= :min_member
ORDER BY t.n_member_cluster DESC, t.n_episode_cluster DESC
LIMIT :lim
"""

df_clusters = read_df(
    engine,
    sql_clusters,
    {
        "jp": jenis_pendekatan,
        "mid": modeling_id,
        "w": window_days,
        "tc": time_col,
        "min_member": int(min_member_filter),
        "lim": int(limit_clusters),
    },
)

if df_clusters.empty:
    st.warning("Tidak ada cluster sesuai filter.")
    st.stop()

# Prepare display columns
df_view = df_clusters.copy()
df_view["top_kata"] = df_view["top_terms_json"].apply(lambda x: json_top_kata_to_str(x, k=12))
df_view = df_view.drop(columns=["top_terms_json"])

st.dataframe(df_view, use_container_width=True, height=360)

st.markdown("---")

# ------------------------------------------------------
# 4) Pilih cluster untuk drilldown + edit
# ------------------------------------------------------
st.subheader("4) Drilldown Cluster & Edit label_topic")

cluster_ids = df_clusters["cluster_id"].astype(int).tolist()
default_cid = cluster_ids[0]

cid = st.selectbox("Pilih cluster_id", options=cluster_ids, index=0)

# Fetch current topic row
topic_row = df_clusters[df_clusters["cluster_id"].astype(int) == int(cid)].iloc[0]
current_label_topic = "" if pd.isna(topic_row["label_topic"]) else str(topic_row["label_topic"])
top_kata_str = json_top_kata_to_str(topic_row["top_terms_json"], k=20)

c1, c2 = st.columns([1, 2])
with c1:
    st.metric("n_member", int(topic_row["n_member_cluster"]))
    st.metric("n_episode", int(topic_row["n_episode_cluster"]))
    st.metric("label_berulang", int(topic_row["label_berulang"] or 0))
with c2:
    st.markdown("**Top kata (TF-IDF)**")
    st.write(top_kata_str if top_kata_str else "-")

st.markdown("### ✏️ Edit label_topic (manual override)")
new_label_topic = st.text_input("label_topic", value=current_label_topic, help="Isi label tema yang lebih manusiawi/representatif.")

colS1, colS2, colS3 = st.columns([1, 1, 1])
with colS1:
    do_save = st.button("💾 Simpan label_topic", type="primary")
with colS2:
    clear_label = st.button("🧹 Kosongkan label_topic")
with colS3:
    refresh_btn = st.button("🔄 Refresh data")

if clear_label:
    new_label_topic = ""

if do_save:
    exec_sql(
        engine,
        f"""
        UPDATE {T_TOPIC}
        SET label_topic = :label_topic,
            created_at = now()
        WHERE jenis_pendekatan = :jp
          AND modeling_id = CAST(:mid AS uuid)
          AND window_days = :w
          AND time_col = :tc
          AND cluster_id = :cid
        """,
        {
            "label_topic": new_label_topic.strip() if new_label_topic is not None else None,
            "jp": jenis_pendekatan,
            "mid": modeling_id,
            "w": window_days,
            "tc": time_col,
            "cid": int(cid),
        },
    )
    st.success("✅ label_topic berhasil disimpan.")

if refresh_btn:
    st.rerun()

st.markdown("---")

# ------------------------------------------------------
# 5) Drilldown incidents: list example tickets + text
# ------------------------------------------------------
st.subheader("5) Contoh Tiket dalam Cluster Ini")

colD1, colD2, colD3 = st.columns([1, 1, 1])
with colD1:
    sample_n = st.number_input("Jumlah contoh tiket", min_value=5, max_value=50, value=10, step=5)
with colD2:
    only_episode = st.checkbox("Tampilkan per-episode (limit 1 tiket per episode)", value=False)
with colD3:
    show_raw_json = st.checkbox("Tampilkan JSON top_terms (debug)", value=False)

if show_raw_json:
    st.code(json.dumps(topic_row["top_terms_json"], ensure_ascii=False, indent=2))

# Load incidents for this cluster
extra = f" AND ({extra_where})" if extra_where.strip() else ""

if not only_episode:
    sql_inc = f"""
    SELECT
      l.incident_number,
      l.temporal_cluster_no,
      l.temporal_cluster_id,
      l.gap_days,
      l.site, l.assignee, l.modul, l.sub_modul,
      s.{src_text_col_safe} AS text_value
    FROM {T_LABEL} l
    LEFT JOIN {src_table_safe} s
      ON s.{src_incident_col_safe} = l.incident_number
    WHERE l.jenis_pendekatan = :jp
      AND l.modeling_id = CAST(:mid AS uuid)
      AND l.window_days = :w
      AND l.time_col = :tc
      AND l.cluster_id = :cid
      {extra}
    ORDER BY l.temporal_cluster_no, l.incident_number
    LIMIT :lim
    """
else:
    # one ticket per episode
    sql_inc = f"""
    WITH base AS (
      SELECT
        l.incident_number,
        l.temporal_cluster_no,
        l.temporal_cluster_id,
        l.gap_days,
        l.site, l.assignee, l.modul, l.sub_modul,
        s.{src_text_col_safe} AS text_value,
        ROW_NUMBER() OVER (PARTITION BY l.temporal_cluster_no ORDER BY l.incident_number) AS rn
      FROM {T_LABEL} l
      LEFT JOIN {src_table_safe} s
        ON s.{src_incident_col_safe} = l.incident_number
      WHERE l.jenis_pendekatan = :jp
        AND l.modeling_id = CAST(:mid AS uuid)
        AND l.window_days = :w
        AND l.time_col = :tc
        AND l.cluster_id = :cid
        {extra}
    )
    SELECT *
    FROM base
    WHERE rn = 1
    ORDER BY temporal_cluster_no
    LIMIT :lim
    """

df_inc = read_df(
    engine,
    sql_inc,
    {
        "jp": jenis_pendekatan,
        "mid": modeling_id,
        "w": window_days,
        "tc": time_col,
        "cid": int(cid),
        "lim": int(sample_n),
    },
)

if df_inc.empty:
    st.warning("Tidak ada tiket yang ditemukan untuk cluster ini.")
else:
    st.dataframe(df_inc, use_container_width=True, height=420)

    # Optional: show as expanders for readability
    st.markdown("### Ringkasan Teks (expand)")
    for _, r in df_inc.iterrows():
        title = f"{r.get('incident_number')} | ep={r.get('temporal_cluster_no')} | gap={r.get('gap_days')}"
        with st.expander(title):
            st.write(r.get("text_value") if pd.notna(r.get("text_value")) else "(text_value kosong / NULL)")
            meta = {
                "site": r.get("site"),
                "assignee": r.get("assignee"),
                "modul": r.get("modul"),
                "sub_modul": r.get("sub_modul"),
                "temporal_cluster_id": r.get("temporal_cluster_id"),
            }
            st.caption(meta)
