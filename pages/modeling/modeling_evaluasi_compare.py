# pages/modeling/modeling_evaluasi_compare.py
from __future__ import annotations

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

SCHEMA = "lasis_djp"

# --- Sintaksis temporal tables ---
T_SYN_SUM = "modeling_sintaksis_temporal_summary"
T_SYN_MEM = "modeling_sintaksis_temporal_members"

# --- Semantik temporal tables ---
T_SEM_SUM = "modeling_semantik_temporal_summary"
T_SEM_MEM = "modeling_semantik_temporal_members"


# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


engine = get_engine()


# ======================================================
# ✅ Helpers: schema introspection + time resolver
# ======================================================
@st.cache_data(show_spinner=False, ttl=300)
def table_has_column(schema: str, table: str, col: str) -> bool:
    q = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
          AND column_name = :col
        LIMIT 1
        """
    )
    with engine.begin() as conn:
        r = conn.execute(q, {"schema": schema, "table": table, "col": col}).fetchone()
    return r is not None


def resolve_time_col(schema: str, table: str, candidates: list[str]) -> str:
    for c in candidates:
        if table_has_column(schema, table, c):
            return c
    raise RuntimeError(f"Tidak menemukan kolom waktu di {schema}.{table}. Kandidat: {candidates}")


def build_optional_filter(table_name: str, col: str, param_name: str | None = None) -> str:
    """
    Return SQL fragment: ' AND col = :param' jika kolom ada; else ''.
    param_name default = col.
    """
    p = param_name or col
    return f" AND {col} = :{p}" if table_has_column(SCHEMA, table_name, col) else ""


# ======================================================
# ✅ Load options
# ======================================================
@st.cache_data(show_spinner=False, ttl=60)
def load_job_model_pairs_sintaksis() -> pd.DataFrame:
    q = text(
        f"""
        SELECT DISTINCT job_id::text AS job_id, modeling_id::text AS modeling_id
        FROM {SCHEMA}.{T_SYN_SUM}
        ORDER BY job_id::text DESC, modeling_id::text DESC
        """
    )
    return pd.read_sql(q, engine)


@st.cache_data(show_spinner=False, ttl=60)
def load_modeling_ids_semantik() -> pd.DataFrame:
    q = text(
        f"""
        SELECT DISTINCT modeling_id::text AS modeling_id
        FROM {SCHEMA}.{T_SEM_SUM}
        ORDER BY modeling_id::text DESC
        """
    )
    return pd.read_sql(q, engine)


@st.cache_data(show_spinner=False, ttl=60)
def load_syn_summary(job_id: str, modeling_id: str) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
          job_id::text AS job_id,
          modeling_id::text AS modeling_id,
          window_days,
          n_clusters_eligible,
          n_clusters_split,
          prop_clusters_split,
          n_clusters_stable,
          prop_clusters_stable,
          total_episodes,
          avg_episode_per_cluster,
          median_episode_per_cluster,
          run_time
        FROM {SCHEMA}.{T_SYN_SUM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
        ORDER BY window_days ASC
        """
    )
    return pd.read_sql(q, engine, params={"job_id": job_id, "modeling_id": modeling_id})


@st.cache_data(show_spinner=False, ttl=60)
def load_sem_summary(modeling_id: str, time_col: str, include_noise: bool) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
          modeling_id::text AS modeling_id,
          window_days,
          time_col,
          include_noise,
          eligible_rule,
          n_clusters_eligible,
          n_clusters_split,
          prop_clusters_split,
          n_clusters_stable,
          prop_clusters_stable,
          total_episodes,
          avg_episode_per_cluster,
          median_episode_per_cluster,
          run_time
        FROM {SCHEMA}.{T_SEM_SUM}
        WHERE modeling_id = CAST(:modeling_id AS uuid)
          AND time_col = :time_col
          AND include_noise = :include_noise
        ORDER BY window_days ASC
        """
    )
    return pd.read_sql(
        q,
        engine,
        params={"modeling_id": modeling_id, "time_col": time_col, "include_noise": include_noise},
    )


@st.cache_data(show_spinner=False, ttl=60)
def load_members_counts(table_name: str, where_sql: str, params: dict) -> pd.DataFrame:
    time_col = resolve_time_col(SCHEMA, table_name, ["event_time", "tgl_submit"])
    q = text(
        f"""
        SELECT
          cluster_id,
          COUNT(*) AS n_member,
          COUNT(DISTINCT temporal_cluster_no) AS n_episode,
          MIN({time_col}) AS min_time,
          MAX({time_col}) AS max_time
        FROM {SCHEMA}.{table_name}
        {where_sql}
        GROUP BY cluster_id
        """
    )
    df = pd.read_sql(q, engine, params=params)
    if not df.empty:
        df["min_time"] = pd.to_datetime(df["min_time"], errors="coerce")
        df["max_time"] = pd.to_datetime(df["max_time"], errors="coerce")
        df["span_days"] = (df["max_time"] - df["min_time"]).dt.days
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_episode_list(table_name: str, where_sql: str, params: dict) -> pd.DataFrame:
    q = text(
        f"""
        SELECT DISTINCT temporal_cluster_id, temporal_cluster_no
        FROM {SCHEMA}.{table_name}
        {where_sql}
        ORDER BY temporal_cluster_no ASC, temporal_cluster_id ASC
        """
    )
    return pd.read_sql(q, engine, params=params)


@st.cache_data(show_spinner=False, ttl=60)
def load_episode_members(table_name: str, where_sql: str, params: dict, limit_rows: int) -> pd.DataFrame:
    time_col = resolve_time_col(SCHEMA, table_name, ["event_time", "tgl_submit"])

    opt_cols = []
    for c in ["site", "assignee", "modul", "sub_modul", "gap_days"]:
        if table_has_column(SCHEMA, table_name, c):
            opt_cols.append(c)
    opt_sql = ", " + ", ".join(opt_cols) if opt_cols else ""

    q = text(
        f"""
        SELECT
          incident_number,
          {time_col} AS event_time,
          cluster_id,
          temporal_cluster_no,
          temporal_cluster_id
          {opt_sql}
        FROM {SCHEMA}.{table_name}
        {where_sql}
        ORDER BY {time_col} ASC NULLS LAST, incident_number ASC
        LIMIT :limit_rows
        """
    )
    p = dict(params)
    p["limit_rows"] = int(limit_rows)
    df = pd.read_sql(q, engine, params=p)
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_cluster_members_full(table_name: str, where_sql: str, params: dict, limit_rows: int) -> pd.DataFrame:
    """Load seluruh member dalam satu cluster (gabungan semua episode)."""
    time_col = resolve_time_col(SCHEMA, table_name, ["event_time", "tgl_submit"])

    opt_cols = []
    for c in ["site", "assignee", "modul", "sub_modul", "gap_days", "temporal_cluster_no", "temporal_cluster_id"]:
        if table_has_column(SCHEMA, table_name, c):
            opt_cols.append(c)
    opt_sql = ", " + ", ".join(opt_cols) if opt_cols else ""

    q = text(
        f"""
        SELECT
          incident_number,
          {time_col} AS event_time,
          cluster_id
          {opt_sql}
        FROM {SCHEMA}.{table_name}
        {where_sql}
        ORDER BY {time_col} ASC NULLS LAST, incident_number ASC
        LIMIT :limit_rows
        """
    )
    p = dict(params)
    p["limit_rows"] = int(limit_rows)
    df = pd.read_sql(q, engine, params=p)
    if not df.empty:
        df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    return df


# ===== PATCH: episode size distribution (jumlah tiket per temporal_cluster_id)
@st.cache_data(show_spinner=False, ttl=60)
def load_episode_size_distribution(table_name: str, where_sql: str, params: dict) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            temporal_cluster_id,
            COUNT(*) AS episode_size
        FROM {SCHEMA}.{table_name}
        {where_sql}
        GROUP BY temporal_cluster_id
        """
    )
    return pd.read_sql(q, engine, params=params)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ======================================================
# UI — Sidebar
# ======================================================
st.title("📊 Perbandingan Evaluasi Temporal — Sintaksis vs Semantik")

with st.sidebar:
    st.header("Parameter Sintaksis")
    syn_pairs = load_job_model_pairs_sintaksis()
    if syn_pairs.empty:
        st.warning("Tidak ada data sintaksis temporal summary.")
        st.stop()

    opt_pairs = [f"{r.job_id} | {r.modeling_id}" for r in syn_pairs.itertuples(index=False)]
    sel_pair = st.selectbox("job_id | modeling_id (sintaksis)", opt_pairs, index=0)
    job_id, syn_modeling_id = [x.strip() for x in sel_pair.split("|")]

    st.divider()
    st.header("Parameter Semantik")
    sem_mids = load_modeling_ids_semantik()
    if sem_mids.empty:
        st.warning("Tidak ada data semantik temporal summary.")
        st.stop()

    sem_modeling_id = st.selectbox("modeling_id (semantik)", sem_mids["modeling_id"].tolist(), index=0)

    sem_opts = pd.read_sql(
        text(
            f"""
            SELECT DISTINCT time_col, include_noise, eligible_rule
            FROM {SCHEMA}.{T_SEM_SUM}
            WHERE modeling_id = CAST(:mid AS uuid)
            ORDER BY time_col ASC, include_noise ASC, eligible_rule ASC
            """
        ),
        engine,
        params={"mid": sem_modeling_id},
    )

    if sem_opts.empty:
        st.warning("Tidak ada kombinasi time_col/include_noise pada summary semantik.")
        st.stop()

    time_col = st.selectbox("time_col", sem_opts["time_col"].tolist(), index=0)

    # include_noise options per time_col
    noise_candidates = sem_opts[sem_opts["time_col"] == time_col]["include_noise"].drop_duplicates().tolist()
    include_noise = st.selectbox("include_noise", noise_candidates, index=0)

    # eligible_rule options (opsional)
    rule_candidates = (
        sem_opts[(sem_opts["time_col"] == time_col) & (sem_opts["include_noise"] == include_noise)]["eligible_rule"]
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    sem_rule = None
    if rule_candidates:
        sem_rule = st.selectbox("eligible_rule", rule_candidates, index=0)

    st.divider()
    st.header("Opsi Tampilan")
    compare_mode = st.radio(
        "Mode perbandingan",
        options=["Episode (temporal)", "Cluster FULL"],
        index=0,
        help=(
            "Episode: bandingkan satu episode temporal tertentu.\n"
            "Cluster FULL: bandingkan seluruh tiket dalam satu cluster (gabungan semua episode)."
        ),
    )
    show_tables = st.checkbox("Tampilkan tabel ringkasan", value=True)
    show_cluster_dist = st.checkbox("Tampilkan distribusi episode/cluster", value=True)
    drill_limit = st.slider("Limit baris drilldown", 1_000, 50_000, 10_000, step=1_000)
    topn = st.slider("Top-N cluster (drilldown list)", 10, 200, 50, step=10)

# ======================================================
# Load summaries
# ======================================================
syn_sum = load_syn_summary(job_id, syn_modeling_id)
sem_sum = load_sem_summary(sem_modeling_id, time_col, bool(include_noise))

if syn_sum.empty:
    st.warning("Summary sintaksis kosong untuk pasangan job_id & modeling_id ini.")
    st.stop()
if sem_sum.empty:
    st.warning("Summary semantik kosong untuk parameter semantik yang dipilih.")
    st.stop()

win_common = sorted(set(syn_sum["window_days"]).intersection(set(sem_sum["window_days"])))
if not win_common:
    st.warning("Tidak ada irisan window_days antara sintaksis dan semantik.")
    st.stop()

# ======================================================
# KPI Compare
# ======================================================
st.subheader("KPI Perbandingan (window terpilih)")
win_sel = st.select_slider("Pilih window_days", options=win_common, value=win_common[0])

syn_row = syn_sum[syn_sum["window_days"] == win_sel].iloc[0]
sem_row = sem_sum[sem_sum["window_days"] == win_sel].iloc[0]

c1, c2 = st.columns(2)
with c1:
    st.markdown("### Sintaksis")
    a, b, c = st.columns(3)
    a.metric("Eligible", f"{int(syn_row['n_clusters_eligible']):,}")
    b.metric("Split", f"{int(syn_row['n_clusters_split']):,}", f"{syn_row['prop_clusters_split']*100:.1f}%")
    c.metric("Stabil", f"{int(syn_row['n_clusters_stable']):,}", f"{syn_row['prop_clusters_stable']*100:.1f}%")
    a2, b2, c2_ = st.columns(3)
    a2.metric("Total episodes", f"{int(syn_row['total_episodes']):,}")
    b2.metric("Avg ep/cluster", f"{syn_row['avg_episode_per_cluster']:.2f}")
    c2_.metric("Median ep/cluster", f"{syn_row['median_episode_per_cluster']:.2f}")

with c2:
    st.markdown("### Semantik")
    a, b, c = st.columns(3)
    a.metric("Eligible", f"{int(sem_row['n_clusters_eligible']):,}")
    b.metric("Split", f"{int(sem_row['n_clusters_split']):,}", f"{sem_row['prop_clusters_split']*100:.1f}%")
    c.metric("Stabil", f"{int(sem_row['n_clusters_stable']):,}", f"{sem_row['prop_clusters_stable']*100:.1f}%")
    a2, b2, c2_ = st.columns(3)
    a2.metric("Total episodes", f"{int(sem_row['total_episodes']):,}")
    b2.metric("Avg ep/cluster", f"{sem_row['avg_episode_per_cluster']:.2f}")
    c2_.metric("Median ep/cluster", f"{sem_row['median_episode_per_cluster']:.2f}")

    cap = f"time_col={time_col} | noise={include_noise}"
    if sem_rule is not None:
        cap += f" | eligible_rule={sem_rule}"
    st.caption(cap)

# ======================================================
# Trend charts across windows
# ======================================================
st.subheader("Tren Multi-window")

plot_df = pd.concat(
    [
        syn_sum.assign(approach="Sintaksis", key=syn_sum["job_id"] + " | " + syn_sum["modeling_id"])[
            [
                "approach",
                "key",
                "window_days",
                "prop_clusters_split",
                "avg_episode_per_cluster",
                "total_episodes",
                "n_clusters_eligible",
                "n_clusters_split",
            ]
        ],
        sem_sum.assign(approach="Semantik", key=sem_sum["modeling_id"] + f" | {time_col} | noise={include_noise}")[  # type: ignore
            [
                "approach",
                "key",
                "window_days",
                "prop_clusters_split",
                "avg_episode_per_cluster",
                "total_episodes",
                "n_clusters_eligible",
                "n_clusters_split",
            ]
        ],
    ],
    ignore_index=True,
)

cc1, cc2 = st.columns(2)
with cc1:
    ch = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("window_days:O", title="Window (hari)"),
            y=alt.Y("prop_clusters_split:Q", title="Proporsi Cluster Split"),
            color=alt.Color("approach:N", title="Pendekatan"),
            tooltip=[
                "approach",
                "key",
                "window_days",
                "n_clusters_eligible",
                "n_clusters_split",
                "prop_clusters_split",
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(ch, use_container_width=True)

with cc2:
    ch2 = (
        alt.Chart(plot_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("window_days:O", title="Window (hari)"),
            y=alt.Y("avg_episode_per_cluster:Q", title="Avg Episode/Cluster"),
            color=alt.Color("approach:N", title="Pendekatan"),
            tooltip=["approach", "key", "window_days", "avg_episode_per_cluster", "total_episodes"],
        )
        .properties(height=300)
    )
    st.altair_chart(ch2, use_container_width=True)

if show_tables:
    st.subheader("Tabel Ringkasan")
    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### Sintaksis")
        st.dataframe(syn_sum, use_container_width=True, hide_index=True)
    with t2:
        st.markdown("#### Semantik")
        st.dataframe(sem_sum, use_container_width=True, hide_index=True)

# ======================================================
# Cluster distributions + member counts (window selected)
# ======================================================
sem_noise_sql = build_optional_filter(T_SEM_MEM, "include_noise", "include_noise")
sem_rule_sql = build_optional_filter(T_SEM_MEM, "eligible_rule", "eligible_rule") if sem_rule is not None else ""

syn_cluster = load_members_counts(
    table_name=T_SYN_MEM,
    where_sql="""
      WHERE job_id = CAST(:job_id AS uuid)
        AND modeling_id = CAST(:mid AS uuid)
        AND window_days = :w
    """,
    params={"job_id": job_id, "mid": syn_modeling_id, "w": int(win_sel)},
)

sem_cluster = load_members_counts(
    table_name=T_SEM_MEM,
    where_sql=f"""
      WHERE modeling_id = CAST(:mid AS uuid)
        AND window_days = :w
        AND time_col = :tc
        {sem_noise_sql}
        {sem_rule_sql}
    """,
    params={
        "mid": sem_modeling_id,
        "w": int(win_sel),
        "tc": str(time_col),
        "include_noise": bool(include_noise),
        "eligible_rule": sem_rule,
    },
)

if show_cluster_dist:
    st.subheader(f"Distribusi Episode per Cluster (window={win_sel})")

    if syn_cluster.empty or sem_cluster.empty:
        st.info("Data cluster-level tidak lengkap untuk membuat distribusi episode.")
    else:
        syn_d = (
            syn_cluster.groupby("n_episode", as_index=False)
            .agg(n_cluster=("cluster_id", "count"))
            .assign(approach="Sintaksis")
        )
        sem_d = (
            sem_cluster.groupby("n_episode", as_index=False)
            .agg(n_cluster=("cluster_id", "count"))
            .assign(approach="Semantik")
        )
        dist_df = pd.concat([syn_d, sem_d], ignore_index=True)
        dist_df["n_episode"] = dist_df["n_episode"].astype(int)

        dist_chart = (
            alt.Chart(dist_df)
            .mark_bar()
            .encode(
                x=alt.X("n_episode:O", title="Jumlah Episode per Cluster"),
                y=alt.Y("n_cluster:Q", title="Jumlah Cluster"),
                color=alt.Color("approach:N", title="Pendekatan"),
                tooltip=["approach", "n_episode", "n_cluster"],
            )
            .properties(height=320)
        )
        st.altair_chart(dist_chart, use_container_width=True)

    # ======================================================
    # ✅ PATCH: Distribusi Ukuran Episode (1,2,3,... tiket per episode)
    # ======================================================
    st.subheader(f"Distribusi Ukuran Episode (Jumlah Tiket per Episode) — window={win_sel}")

    syn_ep_size = load_episode_size_distribution(
        table_name=T_SYN_MEM,
        where_sql="""
          WHERE job_id = CAST(:job_id AS uuid)
            AND modeling_id = CAST(:mid AS uuid)
            AND window_days = :w
        """,
        params={"job_id": job_id, "mid": syn_modeling_id, "w": int(win_sel)},
    )

    sem_ep_size = load_episode_size_distribution(
        table_name=T_SEM_MEM,
        where_sql=f"""
          WHERE modeling_id = CAST(:mid AS uuid)
            AND window_days = :w
            AND time_col = :tc
            {sem_noise_sql}
            {sem_rule_sql}
        """,
        params={
            "mid": sem_modeling_id,
            "w": int(win_sel),
            "tc": str(time_col),
            "include_noise": bool(include_noise),
            "eligible_rule": sem_rule,
        },
    )

    def prep_episode_size_dist(df: pd.DataFrame, label: str) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=["episode_size", "n_episode", "approach"])
        out = df.groupby("episode_size", as_index=False).agg(n_episode=("temporal_cluster_id", "count"))
        out["approach"] = label
        return out

    syn_esd = prep_episode_size_dist(syn_ep_size, "Sintaksis")
    sem_esd = prep_episode_size_dist(sem_ep_size, "Semantik")
    esd = pd.concat([syn_esd, sem_esd], ignore_index=True)

    if esd.empty:
        st.info("Distribusi ukuran episode tidak tersedia untuk parameter saat ini.")
    else:
        esd["episode_size"] = esd["episode_size"].astype(int)

        esd_chart = (
            alt.Chart(esd)
            .mark_bar()
            .encode(
                x=alt.X("episode_size:O", title="Ukuran Episode (jumlah tiket)", sort="ascending"),
                y=alt.Y("n_episode:Q", title="Jumlah Episode"),
                color=alt.Color("approach:N", title="Pendekatan"),
                tooltip=["approach", "episode_size", "n_episode"],
            )
            .properties(height=320)
        )
        st.altair_chart(esd_chart, use_container_width=True)

        st.caption(
            "Catatan: ukuran episode dihitung sebagai jumlah tiket dalam satu temporal_cluster_id. "
            "Episode ukuran 1 = episode tunggal (sporadis), sedangkan ukuran ≥2 mengindikasikan insiden berulang."
        )

# ======================================================
# 🧩 DRILLDOWN COMPARE (SIDE-BY-SIDE)
# ======================================================
st.subheader("🔎 Drilldown Compare — Sintaksis vs Semantik")

if syn_cluster.empty or sem_cluster.empty:
    st.info("Drilldown tidak tersedia karena data cluster-level kosong.")
    st.stop()

syn_list = syn_cluster.sort_values(["n_episode", "span_days", "n_member"], ascending=[False, False, False]).head(int(topn))
sem_list = sem_cluster.sort_values(["n_episode", "span_days", "n_member"], ascending=[False, False, False]).head(int(topn))

left, right = st.columns(2, gap="large")

with left:
    st.markdown("### 🅰️ Sintaksis")

    syn_cluster_id = st.selectbox(
        "Pilih cluster_id (Sintaksis)",
        syn_list["cluster_id"].tolist(),
        index=0,
        format_func=lambda x: (
            f"{x} (ep={int(syn_list[syn_list['cluster_id']==x]['n_episode'].iloc[0])}, "
            f"span={int(syn_list[syn_list['cluster_id']==x]['span_days'].iloc[0])}d, "
            f"n={int(syn_list[syn_list['cluster_id']==x]['n_member'].iloc[0])})"
        ),
        key="syn_cluster",
    )

    if compare_mode == "Episode (temporal)":
        syn_episodes = load_episode_list(
            table_name=T_SYN_MEM,
            where_sql="""
              WHERE job_id = CAST(:job_id AS uuid)
                AND modeling_id = CAST(:mid AS uuid)
                AND window_days = :w
                AND cluster_id = :cid
            """,
            params={"job_id": job_id, "mid": syn_modeling_id, "w": int(win_sel), "cid": int(syn_cluster_id)},
        )
        if syn_episodes.empty:
            st.warning("Episode list kosong untuk cluster ini.")
            st.stop()

        syn_episode_id = st.selectbox(
            "Pilih episode (temporal_cluster_id)",
            syn_episodes["temporal_cluster_id"].tolist(),
            index=0,
            key="syn_episode",
        )

        syn_df = load_episode_members(
            table_name=T_SYN_MEM,
            where_sql="""
              WHERE job_id = CAST(:job_id AS uuid)
                AND modeling_id = CAST(:mid AS uuid)
                AND window_days = :w
                AND temporal_cluster_id = :eid
            """,
            params={"job_id": job_id, "mid": syn_modeling_id, "w": int(win_sel), "eid": syn_episode_id},
            limit_rows=int(drill_limit),
        )
    else:
        syn_df = load_cluster_members_full(
            table_name=T_SYN_MEM,
            where_sql="""
              WHERE job_id = CAST(:job_id AS uuid)
                AND modeling_id = CAST(:mid AS uuid)
                AND window_days = :w
                AND cluster_id = :cid
            """,
            params={"job_id": job_id, "mid": syn_modeling_id, "w": int(win_sel), "cid": int(syn_cluster_id)},
            limit_rows=int(drill_limit),
        )

    st.metric("Jumlah tiket", f"{len(syn_df):,}")
    if not syn_df.empty:
        st.caption(f"Periode: {syn_df['event_time'].min()} — {syn_df['event_time'].max()}")
    st.dataframe(syn_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download CSV (Sintaksis)",
        data=syn_df.to_csv(index=False).encode("utf-8"),
        file_name=f"drill_syn_{job_id}_{syn_modeling_id}_w{win_sel}_cluster{syn_cluster_id}_{compare_mode.replace(' ','_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

with right:
    st.markdown("### 🅱️ Semantik")

    sem_cluster_id = st.selectbox(
        "Pilih cluster_id (Semantik)",
        sem_list["cluster_id"].tolist(),
        index=0,
        format_func=lambda x: (
            f"{x} (ep={int(sem_list[sem_list['cluster_id']==x]['n_episode'].iloc[0])}, "
            f"span={int(sem_list[sem_list['cluster_id']==x]['span_days'].iloc[0])}d, "
            f"n={int(sem_list[sem_list['cluster_id']==x]['n_member'].iloc[0])})"
        ),
        key="sem_cluster",
    )

    sem_noise_sql = build_optional_filter(T_SEM_MEM, "include_noise", "include_noise")
    sem_rule_sql = build_optional_filter(T_SEM_MEM, "eligible_rule", "eligible_rule") if sem_rule is not None else ""

    if compare_mode == "Episode (temporal)":
        sem_episodes = load_episode_list(
            table_name=T_SEM_MEM,
            where_sql=f"""
              WHERE modeling_id = CAST(:mid AS uuid)
                AND window_days = :w
                AND time_col = :tc
                AND cluster_id = :cid
                {sem_noise_sql}
                {sem_rule_sql}
            """,
            params={
                "mid": sem_modeling_id,
                "w": int(win_sel),
                "tc": str(time_col),
                "cid": int(sem_cluster_id),
                "include_noise": bool(include_noise),
                "eligible_rule": sem_rule,
            },
        )
        if sem_episodes.empty:
            st.warning("Episode list kosong untuk cluster ini.")
            st.stop()

        sem_episode_id = st.selectbox(
            "Pilih episode (temporal_cluster_id)",
            sem_episodes["temporal_cluster_id"].tolist(),
            index=0,
            key="sem_episode",
        )

        # ✅ FIXED INDENTATION + VALID PYTHON
        sem_df = load_episode_members(
            table_name=T_SEM_MEM,
            where_sql=f"""
              WHERE modeling_id = CAST(:mid AS uuid)
                AND window_days = :w
                AND time_col = :tc
                AND temporal_cluster_id = :eid
                {sem_noise_sql}
                {sem_rule_sql}
            """,
            params={
                "mid": sem_modeling_id,
                "w": int(win_sel),
                "tc": str(time_col),
                "eid": sem_episode_id,
                "include_noise": bool(include_noise),
                "eligible_rule": sem_rule,
            },
            limit_rows=int(drill_limit),
        )
    else:
        sem_df = load_cluster_members_full(
            table_name=T_SEM_MEM,
            where_sql=f"""
              WHERE modeling_id = CAST(:mid AS uuid)
                AND window_days = :w
                AND time_col = :tc
                AND cluster_id = :cid
                {sem_noise_sql}
                {sem_rule_sql}
            """,
            params={
                "mid": sem_modeling_id,
                "w": int(win_sel),
                "tc": str(time_col),
                "cid": int(sem_cluster_id),
                "include_noise": bool(include_noise),
                "eligible_rule": sem_rule,
            },
            limit_rows=int(drill_limit),
        )

    st.metric("Jumlah tiket", f"{len(sem_df):,}")
    if not sem_df.empty:
        st.caption(f"Periode: {sem_df['event_time'].min()} — {sem_df['event_time'].max()}")
    st.dataframe(sem_df, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download CSV (Semantik)",
        data=sem_df.to_csv(index=False).encode("utf-8"),
        file_name=f"drill_sem_{sem_modeling_id}_w{win_sel}_{time_col}_cluster{sem_cluster_id}_{compare_mode.replace(' ','_')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ======================================================
# Overlap + Similarity metrics
# ======================================================
st.subheader("🔁 Irisan (Overlap) — Incident Number")

syn_set = set(syn_df["incident_number"].astype(str).tolist()) if not syn_df.empty else set()
sem_set = set(sem_df["incident_number"].astype(str).tolist()) if not sem_df.empty else set()
inter = syn_set.intersection(sem_set)

na, nb = len(syn_set), len(sem_set)
overlap = len(inter)

jac = jaccard(syn_set, sem_set)
rate_a = overlap / na if na else 0.0
rate_b = overlap / nb if nb else 0.0
rate_min = overlap / min(na, nb) if min(na, nb) else 0.0

cA, cB, cC, cD, cE, cF = st.columns(6)
cA.metric("Tiket Sintaksis", f"{na:,}")
cB.metric("Tiket Semantik", f"{nb:,}")
cC.metric("Overlap", f"{overlap:,}")
cD.metric("Jaccard", f"{jac:.4f}")
cE.metric("Overlap/Sintaksis", f"{rate_a*100:.1f}%")
cF.metric("Overlap/Semantik", f"{rate_b*100:.1f}%")
st.caption(f"Coverage min-set (Overlap/min(|A|,|B|)): {rate_min*100:.1f}%")

# ======================================================
# Contoh Tiket (10 overlap vs 10 unik)
# ======================================================
st.subheader("🧾 Contoh Tiket — Overlap vs Unik (Top 10)")

only_syn = syn_set - sem_set
only_sem = sem_set - syn_set


def sample_rows(df: pd.DataFrame, id_list: list[str], label: str, n: int = 10) -> pd.DataFrame:
    if df.empty or not id_list:
        return pd.DataFrame(columns=["label", "incident_number"])
    colset = set(df.columns)
    prefer_cols = [
        "incident_number",
        "event_time",
        "cluster_id",
        "temporal_cluster_no",
        "temporal_cluster_id",
        "site",
        "modul",
        "sub_modul",
        "assignee",
        "gap_days",
    ]
    cols = [c for c in prefer_cols if c in colset]

    out = df[df["incident_number"].astype(str).isin(id_list)].copy()
    out["incident_number"] = out["incident_number"].astype(str)

    if "event_time" in out.columns:
        out = out.sort_values(["event_time", "incident_number"], ascending=[True, True])
    else:
        out = out.sort_values(["incident_number"], ascending=[True])

    out = out[cols].head(n)
    out.insert(0, "label", label)
    return out


overlap_ids = sorted(list(inter))[:10]
only_syn_ids = sorted(list(only_syn))[:10]
only_sem_ids = sorted(list(only_sem))[:10]

overlap_syn_df = sample_rows(syn_df, overlap_ids, "OVERLAP (dari Sintaksis)", n=10)
overlap_sem_df = sample_rows(sem_df, overlap_ids, "OVERLAP (dari Semantik)", n=10)
only_syn_df = sample_rows(syn_df, only_syn_ids, "UNIK Sintaksis", n=10)
only_sem_df = sample_rows(sem_df, only_sem_ids, "UNIK Semantik", n=10)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("#### ✅ 10 Tiket Overlap (contoh)")
    if overlap == 0:
        st.info("Tidak ada overlap pada pilihan saat ini.")
    else:
        st.caption("Ditampilkan dari sisi Sintaksis & Semantik (agar terlihat perbedaan episode/cluster).")
        st.dataframe(overlap_syn_df, use_container_width=True, hide_index=True)
        st.dataframe(overlap_sem_df, use_container_width=True, hide_index=True)

with col2:
    st.markdown("#### 🧩 10 Tiket Unik (contoh)")
    a, b = st.columns(2)
    with a:
        st.markdown("**Hanya Sintaksis**")
        if len(only_syn) == 0:
            st.info("Tidak ada tiket unik Sintaksis.")
        else:
            st.dataframe(only_syn_df, use_container_width=True, hide_index=True)
    with b:
        st.markdown("**Hanya Semantik**")
        if len(only_sem) == 0:
            st.info("Tidak ada tiket unik Semantik.")
        else:
            st.dataframe(only_sem_df, use_container_width=True, hide_index=True)

combined_samples = pd.concat([overlap_syn_df, overlap_sem_df, only_syn_df, only_sem_df], ignore_index=True)

st.download_button(
    "⬇️ Download CSV (contoh overlap & unik)",
    data=combined_samples.to_csv(index=False).encode("utf-8"),
    file_name=f"samples_overlap_unik_{compare_mode.replace(' ','_')}_w{win_sel}.csv",
    mime="text/csv",
)

with st.expander("Lihat daftar incident_number overlap"):
    if overlap == 0:
        st.info("Tidak ada overlap pada pilihan saat ini.")
    else:
        overlap_df = pd.DataFrame({"incident_number": sorted(inter)})
        st.dataframe(overlap_df, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download CSV (overlap incident_number)",
            data=overlap_df.to_csv(index=False).encode("utf-8"),
            file_name=f"overlap_{compare_mode.replace(' ','_')}_w{win_sel}.csv",
            mime="text/csv",
        )

st.caption(
    "Catatan: Overlap dihitung berdasarkan kesamaan incident_number pada pilihan saat ini. "
    "Mode Episode membandingkan satu episode temporal tertentu, sedangkan Mode Cluster FULL membandingkan seluruh member cluster. "
    f"Semantik difilter berdasarkan time_col={time_col}, include_noise={include_noise}"
    + (f", eligible_rule={sem_rule}" if sem_rule is not None else "")
    + " (filter hanya aktif jika kolom terkait tersedia di tabel member)."
)
