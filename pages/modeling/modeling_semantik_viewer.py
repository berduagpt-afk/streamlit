# pages/modeling_semantic_hdbscan_viewer.py
# ============================================================
# Viewer Hasil Clustering Semantik — HDBSCAN (Read-only)
#
# Sumber:
# - lasis_djp.modeling_semantic_hdbscan_runs
# - lasis_djp.modeling_semantic_hdbscan_clusters
# - lasis_djp.modeling_semantic_hdbscan_members
# + (opsional untuk teks) lasis_djp.incident_semantic (text_semantic)
#
# Fitur:
# - Pilih modeling_id (dropdown)
# - KPI ringkas: n_rows, n_clusters, n_noise, silhouette, DBI
# - Distribusi ukuran cluster (bar chart)
# - Tabel cluster + filter noise
# - Drilldown: pilih cluster_id -> tampilkan member + score + site + text_semantic (preview)
# - Export CSV (clusters & members)
#
# Patch:
# ✅ Hindari UnhashableParamError: arg Engine di cache pakai _engine
# ✅ Query aman + parameterized
# ✅ Tambah kolom site pada drilldown members
# ✅ Max limit drilldown dinaikkan menjadi 40.000
# ============================================================

from __future__ import annotations

import io
import json
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

# =========================
# 🔐 Guard login
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()

# =========================
# ⚙️ Konstanta
# =========================
SCHEMA = "lasis_djp"

T_RUNS = "modeling_semantic_hdbscan_runs"
T_CLUST = "modeling_semantic_hdbscan_clusters"
T_MEM = "modeling_semantic_hdbscan_members"

# optional join untuk tampilkan teks
T_SEM = "incident_semantic"
TEXT_COL = "text_semantic"
KEY_COL = "incident_number"

MAX_MEMBER_LIMIT = 40000  # ✅ sesuai permintaan


# =========================
# 🔌 DB
# =========================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    user = cfg.get("user") or cfg.get("username")  # fallback aman
    url = (
        f"postgresql+psycopg2://{user}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)

@st.cache_data(show_spinner=False, ttl=120)
def table_exists(_engine: Engine, schema: str, table: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = :schema AND table_name = :table
    LIMIT 1
    """
    df = pd.read_sql_query(text(q), _engine, params={"schema": schema, "table": table})
    return not df.empty

@st.cache_data(show_spinner=False, ttl=120)
def column_exists(_engine: Engine, schema: str, table: str, column: str) -> bool:
    q = """
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table AND column_name = :col
    LIMIT 1
    """
    df = pd.read_sql_query(text(q), _engine, params={"schema": schema, "table": table, "col": column})
    return not df.empty

@st.cache_data(show_spinner=False, ttl=120)
def load_runs(_engine: Engine, limit: int = 200) -> pd.DataFrame:
    q = f"""
    SELECT
      modeling_id::text AS modeling_id,
      embedding_run_id::text AS embedding_run_id,
      model_name,
      run_time,
      notes,
      n_rows,
      n_clusters,
      n_noise,
      silhouette,
      dbi
    FROM {SCHEMA}.{T_RUNS}
    ORDER BY run_time DESC
    LIMIT :lim
    """
    return pd.read_sql_query(text(q), _engine, params={"lim": int(limit)})

@st.cache_data(show_spinner=False, ttl=120)
def load_run_detail(_engine: Engine, modeling_id: str) -> pd.DataFrame:
    q = f"""
    SELECT
      modeling_id::text AS modeling_id,
      embedding_run_id::text AS embedding_run_id,
      model_name,
      run_time,
      params_json,
      notes,
      n_rows,
      n_clusters,
      n_noise,
      silhouette,
      dbi
    FROM {SCHEMA}.{T_RUNS}
    WHERE modeling_id = :mid
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id})

@st.cache_data(show_spinner=False, ttl=120)
def load_clusters(_engine: Engine, modeling_id: str) -> pd.DataFrame:
    q = f"""
    SELECT
      cluster_id,
      cluster_size,
      pct
    FROM {SCHEMA}.{T_CLUST}
    WHERE modeling_id = :mid
    ORDER BY
      CASE WHEN cluster_id = -1 THEN 1 ELSE 0 END,
      cluster_size DESC
    """
    return pd.read_sql_query(text(q), _engine, params={"mid": modeling_id})

@st.cache_data(show_spinner=False, ttl=120)
def load_members(
    _engine: Engine,
    modeling_id: str,
    cluster_id: int,
    limit: int = 2000,
    with_text: bool = True,
    with_site: bool = True,
) -> pd.DataFrame:
    """
    Load member per cluster.
    NOTE: Jika tipe incident_number sudah sama di kedua tabel, hapus ::text untuk performa index.
    """
    if with_text:
        # kolom site mungkin tidak ada, jadi kita gunakan flag with_site
        site_select = "s.site," if with_site else ""
        q = f"""
        SELECT
          m.incident_number,
          m.cluster_id,
          m.score,
          s.tgl_submit,
          {site_select}
          s.modul,
          s.sub_modul,
          s.{TEXT_COL} AS {TEXT_COL}
        FROM {SCHEMA}.{T_MEM} m
        LEFT JOIN {SCHEMA}.{T_SEM} s
          ON s.{KEY_COL}::text = m.incident_number::text
        WHERE m.modeling_id = :mid AND m.cluster_id = :cid
        ORDER BY m.score DESC NULLS LAST, m.incident_number
        LIMIT :lim
        """
    else:
        q = f"""
        SELECT incident_number, cluster_id, score
        FROM {SCHEMA}.{T_MEM}
        WHERE modeling_id = :mid AND cluster_id = :cid
        ORDER BY score DESC NULLS LAST, incident_number
        LIMIT :lim
        """
    return pd.read_sql_query(
        text(q),
        _engine,
        params={"mid": modeling_id, "cid": int(cluster_id), "lim": int(limit)},
    )

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


# =========================
# 🧭 UI
# =========================
st.title("🔎 Viewer Hasil HDBSCAN — Semantik")
st.caption("Menampilkan hasil clustering HDBSCAN dari embedding semantik (tanpa ground truth).")

engine = get_engine()

# sanity checks
required_ok = (
    table_exists(engine, SCHEMA, T_RUNS)
    and table_exists(engine, SCHEMA, T_CLUST)
    and table_exists(engine, SCHEMA, T_MEM)
)
if not required_ok:
    st.error(
        f"Tabel hasil HDBSCAN belum lengkap. Pastikan sudah menjalankan modeling HDBSCAN.\n\n"
        f"Required: `{SCHEMA}.{T_RUNS}`, `{SCHEMA}.{T_CLUST}`, `{SCHEMA}.{T_MEM}`"
    )
    st.stop()

has_sem_table = table_exists(engine, SCHEMA, T_SEM)
has_sem_text = has_sem_table and column_exists(engine, SCHEMA, T_SEM, TEXT_COL)
has_sem_site = has_sem_table and column_exists(engine, SCHEMA, T_SEM, "site")

df_runs = load_runs(engine, limit=300)
if df_runs.empty:
    st.warning("Belum ada run HDBSCAN tersimpan. Jalankan modeling dulu.")
    st.stop()

with st.sidebar:
    st.header("📌 Pilih Run HDBSCAN")

    idx = st.selectbox(
        "Run (modeling_id | model | rows)",
        options=list(range(len(df_runs))),
        format_func=lambda i: (
            f"{df_runs.loc[i,'modeling_id']} | "
            f"{df_runs.loc[i,'model_name']} | "
            f"rows={int(df_runs.loc[i,'n_rows'] or 0):,} | "
            f"clusters={int(df_runs.loc[i,'n_clusters'] or 0):,}"
        ),
    )
    modeling_id = str(df_runs.loc[idx, "modeling_id"])

    st.divider()
    st.subheader("Tampilan")
    show_noise = st.checkbox("Tampilkan noise (-1) di list cluster", value=False)

    member_limit = st.number_input(
        "Limit members (drilldown)",
        min_value=50,
        max_value=MAX_MEMBER_LIMIT,   # ✅ max 40.000
        value=min(2000, MAX_MEMBER_LIMIT),
        step=50,
        help=f"Naikkan jika ingin melihat lebih banyak member (maks {MAX_MEMBER_LIMIT:,}).",
    )

    show_text = st.checkbox(
        "Tampilkan text_semantic (butuh incident_semantic.text_semantic)",
        value=True and has_sem_text,
        disabled=not has_sem_text,
    )

    show_site = st.checkbox(
        "Tampilkan site (butuh incident_semantic.site)",
        value=True and has_sem_site,
        disabled=not has_sem_site,
    )

    if not has_sem_text:
        st.info(f"Kolom `{SCHEMA}.{T_SEM}.{TEXT_COL}` tidak tersedia → teks tidak ditampilkan.")
    if not has_sem_site:
        st.info(f"Kolom `{SCHEMA}.{T_SEM}.site` tidak tersedia → site tidak ditampilkan.")

# load detail
df_detail = load_run_detail(engine, modeling_id)
row = df_detail.iloc[0].to_dict() if not df_detail.empty else {}

# KPI
st.subheader("Ringkasan Run")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model", str(row.get("model_name", "-")))
c2.metric("Rows", f"{int(row.get('n_rows') or 0):,}")
c3.metric("Clusters", f"{int(row.get('n_clusters') or 0):,}")
c4.metric("Noise", f"{int(row.get('n_noise') or 0):,}")

c5, c6, c7 = st.columns(3)
sil = row.get("silhouette")
dbi = row.get("dbi")
c5.metric("Silhouette", "-" if sil is None else f"{float(sil):.4f}")
c6.metric("DBI", "-" if dbi is None else f"{float(dbi):.4f}")
c7.metric("Embedding run_id", str(row.get("embedding_run_id", "-")))

with st.expander("Lihat params_json & notes", expanded=False):
    st.write("**run_time:**", row.get("run_time"))
    st.write("**notes:**", row.get("notes") or "")
    pj = row.get("params_json")
    if isinstance(pj, str):
        try:
            pj = json.loads(pj)
        except Exception:
            pass
    st.write("**params_json:**")
    st.json(pj)

# clusters
df_clusters = load_clusters(engine, modeling_id)
if df_clusters.empty:
    st.warning("Tabel cluster kosong untuk run ini.")
    st.stop()

df_clusters_view = df_clusters.copy()
if not show_noise:
    df_clusters_view = df_clusters_view[df_clusters_view["cluster_id"] != -1]

st.subheader("Distribusi Ukuran Cluster")

# bar chart (top N)
top_n = min(80, len(df_clusters_view))
df_plot = df_clusters_view.head(top_n).copy()
df_plot["cluster_id"] = df_plot["cluster_id"].astype(int).astype(str)

chart = (
    alt.Chart(df_plot)
    .mark_bar()
    .encode(
        x=alt.X("cluster_id:O", sort=None, title="cluster_id (top by size)"),
        y=alt.Y("cluster_size:Q", title="cluster_size"),
        tooltip=["cluster_id", "cluster_size", alt.Tooltip("pct:Q", format=".2f")],
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)

st.subheader("Daftar Cluster")
st.dataframe(df_clusters_view, use_container_width=True, height=360)

# Export clusters
st.download_button(
    "⬇️ Download clusters CSV",
    data=df_to_csv_bytes(df_clusters),
    file_name=f"hdbscan_clusters_{modeling_id}.csv",
    mime="text/csv",
)

# Drilldown cluster
st.subheader("Drilldown Cluster (Members)")
cluster_ids = df_clusters_view["cluster_id"].astype(int).tolist()
if show_noise and (-1 not in cluster_ids) and (df_clusters["cluster_id"].astype(int).eq(-1).any()):
    cluster_ids = [-1] + cluster_ids

selected_cluster = st.selectbox("Pilih cluster_id", options=cluster_ids, index=0)

df_members = load_members(
    engine,
    modeling_id=modeling_id,
    cluster_id=int(selected_cluster),
    limit=int(member_limit),
    with_text=bool(show_text and has_sem_text),
    with_site=bool(show_site and has_sem_site),
)

if df_members.empty:
    st.info("Tidak ada member pada cluster ini.")
else:
    # simple stats
    m1, m2, m3 = st.columns(3)
    m1.metric("Members loaded", f"{len(df_members):,}")
    m2.metric("Score mean", "-" if df_members["score"].isna().all() else f"{df_members['score'].mean():.4f}")
    m3.metric("Score max", "-" if df_members["score"].isna().all() else f"{df_members['score'].max():.4f}")

    st.dataframe(df_members, use_container_width=True, height=520)

    st.download_button(
        f"⬇️ Download members CSV (limit {int(member_limit):,})",
        data=df_to_csv_bytes(df_members),
        file_name=f"hdbscan_members_{modeling_id}_cluster_{int(selected_cluster)}_limit_{int(member_limit)}.csv",
        mime="text/csv",
        help="Export sesuai tampilan (kena LIMIT).",
    )

    # preview text examples
    if show_text and has_sem_text and (TEXT_COL in df_members.columns):
        with st.expander("🧾 Contoh teks (top 10 by score)", expanded=False):
            df_ex = df_members.sort_values(["score", "incident_number"], ascending=[False, True]).head(10)
            for _, r in df_ex.iterrows():
                st.markdown(f"**{r['incident_number']}** | score={r.get('score')}")
                st.write(r.get(TEXT_COL, ""))

# Footer note
st.caption(
    "Catatan: Silhouette & DBI dihitung dengan mengecualikan noise (-1). "
    "Untuk interpretasi, periksa cluster terbesar dan contoh tiket ber-score tinggi."
)
