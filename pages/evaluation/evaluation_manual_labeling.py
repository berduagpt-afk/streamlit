# pages/evaluation/evaluation_manual_labeling.py
# ============================================================
# ✅ Evaluasi Manual Hasil Pelabelan (Sampling Review)
# Output: lasis_djp.labeling_manual_review
#
# FIX penting:
# - Jangan pakai { } di komentar DDL karena bisa memicu parsing/formatting tak sengaja
# ============================================================

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

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
T_REVIEW = f"{SCHEMA}.labeling_manual_review"


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


def ensure_review_table(engine):
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {T_REVIEW} (
        review_time timestamptz NOT NULL DEFAULT now(),

        -- run identity
        jenis_pendekatan text NOT NULL,
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        time_col text NOT NULL DEFAULT '',
        cluster_id bigint NOT NULL,

        -- reviewer identity
        reviewer text NOT NULL,

        -- review fields
        label_berulang integer NOT NULL,          -- label sistem (0/1)
        valid_recurrent boolean NULL,             -- apakah label berulang benar?
        valid_topic boolean NULL,                 -- apakah tema cluster konsisten?
        coherence_score integer NULL,             -- 1..5 (konsistensi tema/anggota)
        recurrence_score integer NULL,            -- 1..5 (indikasi berulang lintas episode)
        notes text NULL,

        -- snapshot (audit)
        n_member_cluster bigint NULL,
        n_episode_cluster bigint NULL,
        label_topic text NULL,
        top_kata_json jsonb NULL,
        sample_json jsonb NULL,                   -- snapshot berisi samples dan params

        CONSTRAINT labeling_manual_review_pkey
          PRIMARY KEY (jenis_pendekatan, modeling_id, window_days, time_col, cluster_id, reviewer)
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_review_run ON {T_REVIEW}(modeling_id, window_days);"))
        conn.execute(text(f"CREATE INDEX IF NOT EXISTS idx_review_reviewer ON {T_REVIEW}(reviewer);"))


def json_top_kata_to_str(top_terms_json: Optional[object], k: int = 12) -> str:
    if top_terms_json is None:
        return ""
    try:
        obj = json.loads(top_terms_json) if isinstance(top_terms_json, str) else top_terms_json
        if not isinstance(obj, list):
            return ""
        kata = []
        for d in obj[:k]:
            if isinstance(d, dict) and d.get("kata"):
                kata.append(str(d["kata"]))
        return ", ".join(kata)
    except Exception:
        return ""


def parse_radio(val: str) -> Optional[bool]:
    if val == "Ya":
        return True
    if val == "Tidak":
        return False
    return None


# ======================================================
# 🧭 UI
# ======================================================
st.title("✅ Evaluasi Manual Pelabelan (Sampling Review)")
st.caption("Sampling cluster untuk validasi label_berulang dan konsistensi tema, lalu simpan penilaian reviewer.")

engine = get_engine()
ensure_review_table(engine)

# ------------------------------------------------------
# 1) Pilih run pelabelan
# ------------------------------------------------------
st.subheader("1) Pilih Run Pelabelan")

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
    st.error(f"Tabel {T_LABEL} kosong. Jalankan pelabelan dulu.")
    st.stop()

run_label = runs.apply(
    lambda r: (
        f"{r['jenis_pendekatan']} | modeling={r['modeling_id']} | w={r['window_days']} | time_col='{r['time_col']}' "
        f"| rows={r['n_rows']} | clusters={r['n_clusters']} | last={r['last_run_time']}"
    ),
    axis=1,
)
idx = st.selectbox("Pilih run", options=list(range(len(runs))), format_func=lambda i: run_label.iloc[i])

jp = str(runs.loc[idx, "jenis_pendekatan"])
mid = str(runs.loc[idx, "modeling_id"])
w = int(runs.loc[idx, "window_days"])
tc = str(runs.loc[idx, "time_col"] or "")

st.info(f"Run terpilih: {jp} | modeling_id={mid} | window_days={w} | time_col='{tc}'")

reviewer = st.text_input("Nama reviewer", value=st.session_state.get("username", "") or "")
if not reviewer.strip():
    st.warning("Isi nama reviewer terlebih dahulu (dibutuhkan sebagai kunci penyimpanan).")

st.markdown("---")

# ------------------------------------------------------
# 2) Sumber teks
# ------------------------------------------------------
st.subheader("2) Sumber Teks untuk Contoh Tiket")

default_table = f"{SCHEMA}.incident_normalized"
default_col_incident = "incident_number"
default_col_text = "text_norm"

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
# 3) Sampling cluster
# ------------------------------------------------------
st.subheader("3) Sampling Cluster (Stratified)")

colS1, colS2, colS3, colS4 = st.columns([1, 1, 1, 1])
with colS1:
    sample_each_stratum = st.number_input("Sampel per strata", min_value=1, max_value=100, value=8, step=1)
with colS2:
    big_threshold = st.number_input("Batas cluster besar (n_member ≥)", min_value=10, value=200, step=10)
with colS3:
    focus_recurrent_only = st.checkbox("Fokus recurrent saja", value=False)
with colS4:
    random_seed = st.number_input("Random seed", min_value=0, value=7, step=1)

sample_per_episode = st.checkbox("Ambil 1 tiket per episode (lebih representatif)", value=True)
sample_n_tickets = st.number_input("Jumlah contoh tiket per cluster", min_value=5, max_value=50, value=10, step=5)

sql_clusters = f"""
WITH base AS (
  SELECT
    cluster_id,
    MAX(n_member_cluster)::bigint AS n_member_cluster,
    MAX(n_episode_cluster)::bigint AS n_episode_cluster,
    MAX(label_berulang)::int AS label_berulang
  FROM {T_LABEL}
  WHERE jenis_pendekatan = :jp
    AND modeling_id = CAST(:mid AS uuid)
    AND window_days = :w
    AND time_col = :tc
  GROUP BY cluster_id
),
topic AS (
  SELECT
    cluster_id,
    label_topic,
    top_terms_json
  FROM {T_TOPIC}
  WHERE jenis_pendekatan = :jp
    AND modeling_id = CAST(:mid AS uuid)
    AND window_days = :w
    AND time_col = :tc
)
SELECT
  b.*,
  t.label_topic,
  t.top_terms_json
FROM base b
LEFT JOIN topic t USING (cluster_id)
{"WHERE b.label_berulang = 1" if focus_recurrent_only else ""}
ORDER BY b.n_member_cluster DESC, b.n_episode_cluster DESC
"""

df_clusters = read_df(engine, sql_clusters, {"jp": jp, "mid": mid, "w": w, "tc": tc})
if df_clusters.empty:
    st.warning("Tidak ada cluster untuk run ini.")
    st.stop()

df_clusters["size_bucket"] = df_clusters["n_member_cluster"].apply(lambda x: "BIG" if int(x) >= int(big_threshold) else "SMALL")
df_clusters["stratum"] = df_clusters["label_berulang"].astype(str) + "_" + df_clusters["size_bucket"].astype(str)

recap = (
    df_clusters.groupby(["label_berulang", "size_bucket"], as_index=False)
    .agg(n_clusters=("cluster_id", "count"), avg_member=("n_member_cluster", "mean"), avg_episode=("n_episode_cluster", "mean"))
)
st.dataframe(recap, use_container_width=True, height=200)

def stratified_sample(df: pd.DataFrame, per_stratum: int, seed: int) -> pd.DataFrame:
    out = []
    rng = seed
    for s, g in df.groupby("stratum", sort=False):
        if g.empty:
            continue
        n = min(per_stratum, len(g))
        out.append(g.sample(n=n, random_state=rng))
        rng += 1
    return pd.concat(out, ignore_index=True) if out else df.head(0)

if "review_sample_df" not in st.session_state:
    st.session_state["review_sample_df"] = None

colB1, colB2 = st.columns([1, 1])
with colB1:
    btn_sample = st.button("🎲 Generate Sampel Cluster", type="primary")
with colB2:
    btn_reset = st.button("🧹 Reset Sampel")

if btn_reset:
    st.session_state["review_sample_df"] = None
    st.rerun()

if btn_sample:
    st.session_state["review_sample_df"] = stratified_sample(df_clusters, int(sample_each_stratum), int(random_seed))

sample_df = st.session_state.get("review_sample_df")
if sample_df is None or sample_df.empty:
    st.info("Klik **Generate Sampel Cluster** untuk membuat daftar cluster yang akan direview.")
    st.stop()

sample_view = sample_df.copy()
sample_view["top_kata"] = sample_view["top_terms_json"].apply(lambda x: json_top_kata_to_str(x, k=12))
st.success(f"Jumlah cluster sampel: {len(sample_view):,}")
st.dataframe(
    sample_view[["cluster_id", "label_berulang", "n_member_cluster", "n_episode_cluster", "label_topic", "top_kata", "stratum"]],
    use_container_width=True,
    height=260,
)

st.markdown("---")

# ------------------------------------------------------
# 4) Review per cluster
# ------------------------------------------------------
st.subheader("4) Review Cluster (Satu per Satu)")

cluster_ids = sample_df["cluster_id"].astype(int).tolist()
selected_cid = st.selectbox("Pilih cluster_id", options=cluster_ids, index=0)

row = sample_df[sample_df["cluster_id"].astype(int) == int(selected_cid)].iloc[0]
label_sys = int(row["label_berulang"])
n_member = int(row["n_member_cluster"])
n_episode = int(row["n_episode_cluster"])
label_topic = None if pd.isna(row.get("label_topic")) else str(row.get("label_topic"))
top_kata_str = json_top_kata_to_str(row.get("top_terms_json"), k=20)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Label sistem", label_sys)
c2.metric("n_member", n_member)
c3.metric("n_episode", n_episode)
c4.metric("Strata", str(row["stratum"]))

st.markdown("**label_topic:**")
st.write(label_topic or "-")
st.markdown("**Top kata:**")
st.write(top_kata_str or "-")

extra = f" AND ({extra_where})" if extra_where.strip() else ""

if sample_per_episode:
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
else:
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

df_inc = read_df(
    engine,
    sql_inc,
    {"jp": jp, "mid": mid, "w": w, "tc": tc, "cid": int(selected_cid), "lim": int(sample_n_tickets)},
)

st.markdown("### Contoh tiket")
if df_inc.empty:
    st.warning("Tidak ada contoh tiket ditemukan untuk cluster ini.")
else:
    st.dataframe(df_inc, use_container_width=True, height=320)
    st.markdown("### Ringkasan teks (expand)")
    for _, r in df_inc.iterrows():
        title = f"{r.get('incident_number')} | ep={r.get('temporal_cluster_no')} | gap={r.get('gap_days')}"
        with st.expander(title):
            st.write(r.get("text_value") if pd.notna(r.get("text_value")) else "(text kosong / NULL)")
            st.caption(
                {
                    "site": r.get("site"),
                    "assignee": r.get("assignee"),
                    "modul": r.get("modul"),
                    "sub_modul": r.get("sub_modul"),
                    "temporal_cluster_id": r.get("temporal_cluster_id"),
                }
            )

st.markdown("---")

# ------------------------------------------------------
# 5) Form penilaian + save
# ------------------------------------------------------
st.subheader("5) Form Penilaian Reviewer")

colF1, colF2 = st.columns(2)
with colF1:
    valid_recurrent = st.radio(
        "Apakah label berulang dari sistem sudah benar?",
        options=["Belum diisi", "Ya", "Tidak"],
        index=0,
        horizontal=True,
    )
    recurrence_score = st.slider("Skor indikasi berulang lintas episode (1..5)", min_value=1, max_value=5, value=3)
with colF2:
    valid_topic = st.radio(
        "Apakah tema/anggota cluster konsisten?",
        options=["Belum diisi", "Ya", "Tidak"],
        index=0,
        horizontal=True,
    )
    coherence_score = st.slider("Skor konsistensi tema/anggota (1..5)", min_value=1, max_value=5, value=3)

notes = st.text_area("Catatan reviewer (opsional)", value="", height=120)

btn_save = st.button("💾 Simpan Review", type="primary", disabled=not reviewer.strip())

if btn_save:
    vrec = parse_radio(valid_recurrent)
    vtop = parse_radio(valid_topic)

    sample_payload = {
        "samples": df_inc.to_dict(orient="records") if not df_inc.empty else [],
        "params": {
            "sample_per_episode": bool(sample_per_episode),
            "sample_n_tickets": int(sample_n_tickets),
            "source_table": src_table_safe,
            "source_text_col": src_text_col_safe,
            "extra_where": extra_where.strip(),
            "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        },
    }

    upsert_sql = f"""
    INSERT INTO {T_REVIEW} (
      jenis_pendekatan, modeling_id, window_days, time_col, cluster_id,
      reviewer,
      label_berulang, valid_recurrent, valid_topic,
      coherence_score, recurrence_score, notes,
      n_member_cluster, n_episode_cluster, label_topic, top_kata_json, sample_json
    )
    VALUES (
      :jp, CAST(:mid AS uuid), :w, :tc, :cid,
      :reviewer,
      :label_sys, :valid_recurrent, :valid_topic,
      :coherence_score, :recurrence_score, :notes,
      :n_member, :n_episode, :label_topic, CAST(:top_kata_json AS jsonb), CAST(:sample_json AS jsonb)
    )
    ON CONFLICT (jenis_pendekatan, modeling_id, window_days, time_col, cluster_id, reviewer)
    DO UPDATE SET
      review_time = now(),
      label_berulang = EXCLUDED.label_berulang,
      valid_recurrent = EXCLUDED.valid_recurrent,
      valid_topic = EXCLUDED.valid_topic,
      coherence_score = EXCLUDED.coherence_score,
      recurrence_score = EXCLUDED.recurrence_score,
      notes = EXCLUDED.notes,
      n_member_cluster = EXCLUDED.n_member_cluster,
      n_episode_cluster = EXCLUDED.n_episode_cluster,
      label_topic = EXCLUDED.label_topic,
      top_kata_json = EXCLUDED.top_kata_json,
      sample_json = EXCLUDED.sample_json;
    """

    exec_sql(
        engine,
        upsert_sql,
        {
            "jp": jp,
            "mid": mid,
            "w": int(w),
            "tc": tc,
            "cid": int(selected_cid),
            "reviewer": reviewer.strip(),
            "label_sys": int(label_sys),
            "valid_recurrent": vrec,
            "valid_topic": vtop,
            "coherence_score": int(coherence_score),
            "recurrence_score": int(recurrence_score),
            "notes": notes.strip() if notes.strip() else None,
            "n_member": int(n_member),
            "n_episode": int(n_episode),
            "label_topic": label_topic,
            "top_kata_json": json.dumps(row.get("top_terms_json")),
            "sample_json": json.dumps(sample_payload),
        },
    )
    st.success("✅ Review berhasil disimpan.")

st.markdown("---")
st.subheader("6) Rekap Review Tersimpan (Run ini)")

df_recap = read_df(
    engine,
    f"""
    SELECT
      reviewer,
      COUNT(*) AS n_reviewed_clusters,
      SUM(CASE WHEN valid_recurrent IS TRUE THEN 1 ELSE 0 END) AS n_valid_recurrent_true,
      SUM(CASE WHEN valid_recurrent IS FALSE THEN 1 ELSE 0 END) AS n_valid_recurrent_false,
      SUM(CASE WHEN valid_topic IS TRUE THEN 1 ELSE 0 END) AS n_valid_topic_true,
      SUM(CASE WHEN valid_topic IS FALSE THEN 1 ELSE 0 END) AS n_valid_topic_false,
      AVG(coherence_score) AS avg_coherence_score,
      AVG(recurrence_score) AS avg_recurrence_score,
      MAX(review_time) AS last_review_time
    FROM {T_REVIEW}
    WHERE jenis_pendekatan = :jp
      AND modeling_id = CAST(:mid AS uuid)
      AND window_days = :w
      AND time_col = :tc
    GROUP BY reviewer
    ORDER BY last_review_time DESC
    """,
    {"jp": jp, "mid": mid, "w": w, "tc": tc},
)

st.dataframe(df_recap, use_container_width=True)

df_detail = read_df(
    engine,
    f"""
    SELECT
      review_time, reviewer,
      cluster_id, label_berulang,
      valid_recurrent, valid_topic,
      coherence_score, recurrence_score,
      n_member_cluster, n_episode_cluster,
      label_topic
    FROM {T_REVIEW}
    WHERE jenis_pendekatan = :jp
      AND modeling_id = CAST(:mid AS uuid)
      AND window_days = :w
      AND time_col = :tc
    ORDER BY review_time DESC
    LIMIT 500
    """,
    {"jp": jp, "mid": mid, "w": w, "tc": tc},
)

with st.expander("Lihat detail review (maks 500)"):
    st.dataframe(df_detail, use_container_width=True, height=420)
