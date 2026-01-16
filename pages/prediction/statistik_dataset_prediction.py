# pages/prediction/statistik_dataset_supervised.py
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard Login (opsional - sesuaikan dengan sistem Anda)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"
T_DS = "dataset_supervised"

# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}",
        pool_pre_ping=True,
    )

engine = get_engine()

# ======================================================
# 🧱 Helpers
# ======================================================
def qdf(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def table_exists(schema: str, table: str) -> bool:
    df = qdf(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table
        ) AS ok
        """,
        {"schema": schema, "table": table},
    )
    return bool(df.iloc[0]["ok"]) if not df.empty else False

def fmt_int(x) -> str:
    try:
        return f"{int(x):,}"
    except Exception:
        return "-"

def fmt_pct(x) -> str:
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return "-"

def safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def is_truthy(v) -> bool:
    return bool(v) and str(v).lower() not in ("none", "nan", "nat", "")

# ======================================================
# 📌 Page
# ======================================================
# st.set_page_config(page_title="Statistik Dataset Supervised", layout="wide")
st.title("📊 Statistik Dataset Supervised")
st.caption(
    "Halaman ini menampilkan ringkasan statistik, distribusi label, split train/test, karakteristik temporal/cluster, "
    "serta ringkasan per modul dari tabel `lasis_djp.dataset_supervised`."
)

if not table_exists(SCHEMA, T_DS):
    st.error(f"Tabel `{SCHEMA}.{T_DS}` belum tersedia. Bangun dataset dulu dari halaman builder.")
    st.stop()

# ======================================================
# 🎛️ Filter Utama
# ======================================================
with st.expander("⚙️ Filter", expanded=True):
    # Ambil kombinasi unik untuk filter
    dims = qdf(
        f"""
        SELECT
            jenis_pendekatan,
            modeling_id::text AS modeling_id,
            window_days,
            time_col
        FROM {SCHEMA}."{T_DS}"
        GROUP BY 1,2,3,4
        ORDER BY window_days, jenis_pendekatan, modeling_id
        """
    )

    if dims.empty:
        st.warning("Tabel kosong.")
        st.stop()

    c1, c2, c3, c4, c5 = st.columns([1.2, 2.2, 1.0, 1.2, 1.2])

    with c1:
        jenis_opts = ["(all)"] + sorted(dims["jenis_pendekatan"].dropna().unique().tolist())
        jenis = st.selectbox("jenis_pendekatan", jenis_opts, index=0)

    dims2 = dims.copy()
    if jenis != "(all)":
        dims2 = dims2[dims2["jenis_pendekatan"] == jenis]

    with c2:
        mid_opts = ["(all)"] + sorted(dims2["modeling_id"].dropna().unique().tolist())
        modeling_id = st.selectbox("modeling_id", mid_opts, index=0)

    dims3 = dims2.copy()
    if modeling_id != "(all)":
        dims3 = dims3[dims3["modeling_id"] == modeling_id]

    with c3:
        wd_opts = ["(all)"] + sorted(dims3["window_days"].dropna().astype(int).unique().tolist())
        window_days = st.selectbox("window_days", wd_opts, index=0)

    dims4 = dims3.copy()
    if window_days != "(all)":
        dims4 = dims4[dims4["window_days"].astype(int) == int(window_days)]

    with c4:
        tc_opts = ["(all)"] + sorted(dims4["time_col"].dropna().unique().tolist())
        time_col = st.selectbox("time_col", tc_opts, index=0)

    with c5:
        sample_limit = st.number_input("Sample untuk preview", min_value=100, max_value=50000, value=5000, step=500)

    # tambahan filter ringan
    f1, f2, f3, f4 = st.columns([1.2, 1.2, 1.2, 1.4])
    with f1:
        only_train = st.checkbox("Hanya train", value=False)
    with f2:
        only_test = st.checkbox("Hanya test", value=False)
    with f3:
        only_berulang = st.checkbox("Hanya label=1", value=False)
    with f4:
        exclude_noise = st.checkbox("Exclude noise (cluster_id = -1)", value=False)

# ======================================================
# 🧩 WHERE builder
# ======================================================
where = ["1=1"]
params: Dict[str, Any] = {}

if jenis != "(all)":
    where.append("jenis_pendekatan = :jenis")
    params["jenis"] = jenis

if modeling_id != "(all)":
    where.append("modeling_id = CAST(:modeling_id AS uuid)")
    params["modeling_id"] = modeling_id

if window_days != "(all)":
    where.append("window_days = :window_days")
    params["window_days"] = int(window_days)

if time_col != "(all)":
    where.append("time_col = :time_col")
    params["time_col"] = time_col

if only_train and not only_test:
    where.append("split_name = 'train'")
elif only_test and not only_train:
    where.append("split_name = 'test'")

if only_berulang:
    where.append("label_berulang = 1")

if exclude_noise:
    where.append("(cluster_id IS NULL OR cluster_id <> -1)")

where_sql = " AND ".join(where)

# ======================================================
# ✅ Ringkasan cepat (tanpa tarik data besar)
# ======================================================
summary = qdf(
    f"""
    WITH base AS (
      SELECT *
      FROM {SCHEMA}."{T_DS}"
      WHERE {where_sql}
    )
    SELECT
      COUNT(*)::bigint AS total_rows,
      COUNT(*) FILTER (WHERE label_berulang = 1)::bigint AS n_berulang,
      COUNT(*) FILTER (WHERE label_berulang = 0)::bigint AS n_tidak_berulang,
      ROUND(100.0 * COUNT(*) FILTER (WHERE label_berulang = 1) / NULLIF(COUNT(*),0), 2) AS pct_berulang,

      COUNT(*) FILTER (WHERE split_name = 'train')::bigint AS n_train,
      COUNT(*) FILTER (WHERE split_name = 'test')::bigint AS n_test,

      COUNT(DISTINCT cluster_id) FILTER (WHERE cluster_id IS NOT NULL)::bigint AS n_clusters
    FROM base
    """,
    params,
)

row = summary.iloc[0].to_dict() if not summary.empty else {}

m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
m1.metric("Total baris", fmt_int(row.get("total_rows", 0)))
m2.metric("Label berulang (1)", fmt_int(row.get("n_berulang", 0)))
m3.metric("Label tidak berulang (0)", fmt_int(row.get("n_tidak_berulang", 0)))
m4.metric("% berulang", fmt_pct(row.get("pct_berulang", 0)))
m5.metric("Train", fmt_int(row.get("n_train", 0)))
m6.metric("Test", fmt_int(row.get("n_test", 0)))
m7.metric("Cluster unik", fmt_int(row.get("n_clusters", 0)))

st.divider()

# ======================================================
# 📌 Distribusi (label & split)
# ======================================================
cA, cB = st.columns(2)

label_dist = qdf(
    f"""
    SELECT
      label_berulang::int AS label_berulang,
      COUNT(*)::bigint AS n
    FROM {SCHEMA}."{T_DS}"
    WHERE {where_sql}
    GROUP BY 1
    ORDER BY 1
    """,
    params,
)
split_dist = qdf(
    f"""
    SELECT
      COALESCE(split_name, '(null)') AS split_name,
      COUNT(*)::bigint AS n
    FROM {SCHEMA}."{T_DS}"
    WHERE {where_sql}
    GROUP BY 1
    ORDER BY 2 DESC
    """,
    params,
)
cross = qdf(
    f"""
    SELECT
      COALESCE(split_name, '(null)') AS split_name,
      label_berulang::int AS label_berulang,
      COUNT(*)::bigint AS n
    FROM {SCHEMA}."{T_DS}"
    WHERE {where_sql}
    GROUP BY 1,2
    ORDER BY 1,2
    """,
    params,
)

with cA:
    st.subheader("Distribusi Label")
    if label_dist.empty:
        st.info("Tidak ada data untuk distribusi label.")
    else:
        ch = (
            alt.Chart(label_dist)
            .mark_bar()
            .encode(
                x=alt.X("label_berulang:O", title="label_berulang"),
                y=alt.Y("n:Q", title="Jumlah"),
                tooltip=["label_berulang", "n"],
            )
        )
        st.altair_chart(ch, use_container_width=True)
        st.dataframe(label_dist, use_container_width=True)

with cB:
    st.subheader("Distribusi Split")
    if split_dist.empty:
        st.info("Tidak ada data untuk distribusi split.")
    else:
        ch = (
            alt.Chart(split_dist)
            .mark_bar()
            .encode(
                x=alt.X("split_name:O", title="split_name"),
                y=alt.Y("n:Q", title="Jumlah"),
                tooltip=["split_name", "n"],
            )
        )
        st.altair_chart(ch, use_container_width=True)
        st.dataframe(split_dist, use_container_width=True)

st.subheader("Crosstab Split × Label")
st.dataframe(cross, use_container_width=True)

st.divider()

# ======================================================
# 📈 Statistik fitur numerik (cluster/episode/gap)
# ======================================================
st.subheader("Statistik Fitur Numerik")

num_stats = qdf(
    f"""
    WITH base AS (
      SELECT
        label_berulang::int AS label_berulang,
        n_member_cluster::float AS n_member_cluster,
        n_episode_cluster::float AS n_episode_cluster,
        n_member_episode::float AS n_member_episode,
        gap_days::float AS gap_days
      FROM {SCHEMA}."{T_DS}"
      WHERE {where_sql}
    )
    SELECT
      label_berulang,
      COUNT(*)::bigint AS n,
      AVG(n_member_cluster)::numeric(12,2) AS avg_n_member_cluster,
      AVG(n_episode_cluster)::numeric(12,2) AS avg_n_episode_cluster,
      AVG(n_member_episode)::numeric(12,2) AS avg_n_member_episode,
      AVG(gap_days)::numeric(12,2) AS avg_gap_days,

      PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY n_member_cluster) AS med_n_member_cluster,
      PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY n_episode_cluster) AS med_n_episode_cluster,
      PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY n_member_episode) AS med_n_member_episode,
      PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY gap_days) AS med_gap_days
    FROM base
    GROUP BY 1
    ORDER BY 1
    """,
    params,
)
st.dataframe(num_stats, use_container_width=True)

# Optional charts: gap_days by label (requires pulling sample)
st.subheader("Distribusi Gap Days (Sample)")
sample_df = qdf(
    f"""
    SELECT
      label_berulang::int AS label_berulang,
      gap_days::float AS gap_days
    FROM {SCHEMA}."{T_DS}"
    WHERE {where_sql}
      AND gap_days IS NOT NULL
    ORDER BY random()
    LIMIT :lim
    """,
    {**params, "lim": int(sample_limit)},
)
if sample_df.empty:
    st.info("Tidak ada data gap_days (non-null) pada filter saat ini.")
else:
    # histogram
    sample_df["gap_days"] = safe_num(sample_df["gap_days"])
    sample_df = sample_df.dropna(subset=["gap_days"])
    ch = (
        alt.Chart(sample_df)
        .mark_bar()
        .encode(
            x=alt.X("gap_days:Q", bin=alt.Bin(maxbins=40), title="gap_days"),
            y=alt.Y("count():Q", title="Jumlah"),
            tooltip=[alt.Tooltip("count():Q", title="Jumlah")],
            column=alt.Column("label_berulang:O", title="label_berulang"),
        )
    )
    st.altair_chart(ch, use_container_width=True)

st.divider()

# ======================================================
# 🧩 Statistik kategorikal (modul/sub_modul/site)
# ======================================================
st.subheader("Ringkasan Kategorikal (Top-N)")

topn = st.slider("Top-N", min_value=5, max_value=50, value=15, step=5)

cat_cols = st.columns(3)

def top_counts(colname: str) -> pd.DataFrame:
    return qdf(
        f"""
        SELECT
          COALESCE({colname}, '(null)') AS key,
          COUNT(*)::bigint AS n
        FROM {SCHEMA}."{T_DS}"
        WHERE {where_sql}
        GROUP BY 1
        ORDER BY 2 DESC
        LIMIT :topn
        """,
        {**params, "topn": int(topn)},
    )

def pct_berulang_by(colname: str) -> pd.DataFrame:
    return qdf(
        f"""
        SELECT
          COALESCE({colname}, '(null)') AS key,
          COUNT(*)::bigint AS total,
          COUNT(*) FILTER (WHERE label_berulang = 1)::bigint AS n_berulang,
          ROUND(100.0 * COUNT(*) FILTER (WHERE label_berulang = 1) / NULLIF(COUNT(*),0), 2) AS pct_berulang
        FROM {SCHEMA}."{T_DS}"
        WHERE {where_sql}
        GROUP BY 1
        HAVING COUNT(*) >= 10
        ORDER BY pct_berulang DESC, total DESC
        LIMIT :topn
        """,
        {**params, "topn": int(topn)},
    )

with cat_cols[0]:
    st.markdown("**Top Modul**")
    df_mod = top_counts("modul")
    st.dataframe(df_mod, use_container_width=True)
    if not df_mod.empty:
        st.altair_chart(
            alt.Chart(df_mod).mark_bar().encode(
                x=alt.X("n:Q", title="Jumlah"),
                y=alt.Y("key:N", sort="-x", title="modul"),
                tooltip=["key", "n"],
            ),
            use_container_width=True,
        )

with cat_cols[1]:
    st.markdown("**% Berulang per Modul (min 10 tiket)**")
    df_mod_pct = pct_berulang_by("modul")
    st.dataframe(df_mod_pct, use_container_width=True)
    if not df_mod_pct.empty:
        st.altair_chart(
            alt.Chart(df_mod_pct).mark_bar().encode(
                x=alt.X("pct_berulang:Q", title="% berulang"),
                y=alt.Y("key:N", sort="-x", title="modul"),
                tooltip=["key", "total", "n_berulang", "pct_berulang"],
            ),
            use_container_width=True,
        )

with cat_cols[2]:
    st.markdown("**Top Site**")
    df_site = top_counts("site")
    st.dataframe(df_site, use_container_width=True)
    if not df_site.empty:
        st.altair_chart(
            alt.Chart(df_site).mark_bar().encode(
                x=alt.X("n:Q", title="Jumlah"),
                y=alt.Y("key:N", sort="-x", title="site"),
                tooltip=["key", "n"],
            ),
            use_container_width=True,
        )

st.divider()

# ======================================================
# 🔎 Preview data & download (sample)
# ======================================================
st.subheader("Preview Data (Sample)")

preview = qdf(
    f"""
    SELECT *
    FROM {SCHEMA}."{T_DS}"
    WHERE {where_sql}
    ORDER BY COALESCE(event_time, tgl_submit) NULLS LAST
    LIMIT :lim
    """,
    {**params, "lim": int(min(sample_limit, 20000))},
)

if preview.empty:
    st.info("Tidak ada data untuk preview pada filter saat ini.")
else:
    st.dataframe(preview.head(200), use_container_width=True, height=420)

    # download sample
    csv_bytes = preview.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV (sample hasil filter)",
        data=csv_bytes,
        file_name=f"dataset_supervised_stats_sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

# ======================================================
# ℹ️ Catatan metodologis (untuk tesis)
# ======================================================
with st.expander("📝 Catatan untuk Bab IV (narasi singkat)", expanded=False):
    st.markdown(
        """
- **Distribusi label** biasanya tidak seimbang (class imbalance) karena insiden berulang hanya sebagian kecil dari total tiket.
- Evaluasi model prediksi sebaiknya memprioritaskan **Recall/F1/PR-AUC** untuk kelas `label_berulang = 1`.
- Variabel `n_episode_cluster`, `gap_days`, dan `n_member_cluster` dapat diposisikan sebagai **indikator intensitas pengulangan** dalam dimensi waktu.
        """.strip()
    )
