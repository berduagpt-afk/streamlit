# pages/evaluasi_threshold_temporal.py
# Evaluasi Variasi Cosine Similarity Threshold + Temporal (window_days)
# Membaca: lasis_djp.modeling_runs, lasis_djp.cluster_summary, lasis_djp.cluster_members

from __future__ import annotations

import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# ======================================================
# ⚙️ Konstanta
# ======================================================
SCHEMA = "lasis_djp"
T_RUNS = "modeling_runs"
T_SUMMARY = "cluster_summary"
T_MEMBERS = "cluster_members"

DEFAULT_PICK_N = 9  # ✅ default run yang dipilih


# ======================================================
# 🔌 DB Connection (secrets.toml)
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


engine = get_engine()


# ======================================================
# 🧪 Helpers
# ======================================================
def safe_has_column(conn, schema: str, table: str, column: str) -> bool:
    q = text("""
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND table_name = :table
          AND column_name = :column
        LIMIT 1
    """)
    return conn.execute(q, {"schema": schema, "table": table, "column": column}).fetchone() is not None


@st.cache_data(show_spinner=False, ttl=60)
def load_runs(schema: str, table: str) -> pd.DataFrame:
    with engine.begin() as conn:
        has_threshold = safe_has_column(conn, schema, table, "threshold")
        has_window = safe_has_column(conn, schema, table, "window_days")

    sel_cols = ["run_id", "run_time", "approach", "params_json", "data_range", "notes"]
    if has_threshold:
        sel_cols.insert(3, "threshold")
    else:
        sel_cols.insert(3, "NULL::float8 AS threshold")
    if has_window:
        sel_cols.insert(4, "window_days")
    else:
        sel_cols.insert(4, "NULL::int AS window_days")

    sql = f"""
        SELECT {", ".join(sel_cols)}
        FROM {schema}.{table}
        ORDER BY run_time DESC
    """
    df = pd.read_sql(sql, engine)
    df["run_time"] = pd.to_datetime(df["run_time"], errors="coerce")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["window_days"] = pd.to_numeric(df["window_days"], errors="coerce").astype("Int64")
    return df


@st.cache_data(show_spinner=False, ttl=60)
def load_aggregate_for_runs(schema: str, t_summary: str, t_members: str, run_ids: list[str]) -> pd.DataFrame:
    """
    Hitung metrik per run_id:
    - clusters: jumlah cluster (cluster_summary)
    - members: jumlah member rows (cluster_members)
    - avg_cluster_size
    - median_cluster_size
    - singleton_pct
    """
    if not run_ids:
        return pd.DataFrame()

    sql = text(f"""
        WITH s AS (
            SELECT
                run_id,
                COUNT(*) AS clusters,
                SUM(n_tickets) AS members_from_summary,
                AVG(n_tickets)::float8 AS avg_cluster_size,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY n_tickets)::float8 AS median_cluster_size,
                SUM(CASE WHEN n_tickets = 1 THEN 1 ELSE 0 END) AS singleton_clusters
            FROM {schema}.{t_summary}
            WHERE run_id = ANY(:run_ids)
            GROUP BY run_id
        ),
        m AS (
            SELECT
                run_id,
                COUNT(*) AS members
            FROM {schema}.{t_members}
            WHERE run_id = ANY(:run_ids)
            GROUP BY run_id
        )
        SELECT
            COALESCE(s.run_id, m.run_id) AS run_id,
            COALESCE(s.clusters, 0) AS clusters,
            COALESCE(m.members, 0) AS members,
            COALESCE(s.avg_cluster_size, 0)::float8 AS avg_cluster_size,
            COALESCE(s.median_cluster_size, 0)::float8 AS median_cluster_size,
            CASE
                WHEN COALESCE(s.clusters, 0) = 0 THEN 0
                ELSE (COALESCE(s.singleton_clusters, 0)::float8 / s.clusters::float8) * 100.0
            END AS singleton_pct
        FROM s
        FULL OUTER JOIN m USING (run_id)
        ORDER BY run_id
    """)
    return pd.read_sql(sql, engine, params={"run_ids": run_ids})


@st.cache_data(show_spinner=False, ttl=60)
def load_modul_breakdown(schema: str, t_summary: str, run_ids: list[str], min_clusters: int = 1) -> pd.DataFrame:
    if not run_ids:
        return pd.DataFrame()

    sql = text(f"""
        SELECT
            run_id,
            modul,
            COUNT(*) AS clusters,
            SUM(n_tickets) AS members,
            AVG(n_tickets)::float8 AS avg_cluster_size,
            CASE
                WHEN COUNT(*) = 0 THEN 0
                ELSE (SUM(CASE WHEN n_tickets = 1 THEN 1 ELSE 0 END)::float8 / COUNT(*)::float8) * 100.0
            END AS singleton_pct
        FROM {schema}.{t_summary}
        WHERE run_id = ANY(:run_ids)
        GROUP BY run_id, modul
        HAVING COUNT(*) >= :min_clusters
        ORDER BY modul, run_id
    """)
    return pd.read_sql(sql, engine, params={"run_ids": run_ids, "min_clusters": int(min_clusters)})


def _heatmap(df: pd.DataFrame, value_col: str, title: str):
    """Heatmap threshold x window_days."""
    base = df.dropna(subset=["threshold", "window_days"]).copy()
    if base.empty:
        st.info("Tidak ada data threshold/window_days yang lengkap untuk heatmap.")
        return

    base["window_days"] = base["window_days"].astype(int)

    chart = (
        alt.Chart(base)
        .mark_rect()
        .encode(
            x=alt.X("threshold:Q", title="Threshold"),
            y=alt.Y("window_days:O", title="Window (hari)"),
            tooltip=["run_id", "run_time", "threshold", "window_days", value_col],
            color=alt.Color(f"{value_col}:Q", title=title),
        )
        .properties(height=260)
    )

    # label angka di sel heatmap
    fmt = ".2f" if pd.api.types.is_float_dtype(base[value_col]) else ".0f"
    labels = (
        alt.Chart(base)
        .mark_text()
        .encode(
            x=alt.X("threshold:Q"),
            y=alt.Y("window_days:O"),
            text=alt.Text(f"{value_col}:Q", format=fmt),
        )
    )

    st.altair_chart(chart + labels, use_container_width=True)


# ======================================================
# 🧭 UI
# ======================================================
st.title("📊 Evaluasi Threshold + Temporal (window_days)")
st.caption("Bandingkan hasil modeling sintaksis berdasarkan nilai threshold dan window_days pada tabel modeling_runs.")

runs = load_runs(SCHEMA, T_RUNS)
if runs.empty:
    st.warning("Tabel modeling_runs kosong / belum ada data.")
    st.stop()

# ======================================================
# Sidebar Filters
# ======================================================
st.sidebar.header("Filter Run")

approaches = sorted([x for x in runs["approach"].dropna().unique().tolist()])
sel_approach = st.sidebar.selectbox("Approach", options=["(semua)"] + approaches, index=0)

runs_filtered = runs.copy()
if sel_approach != "(semua)":
    runs_filtered = runs_filtered[runs_filtered["approach"] == sel_approach].copy()

# threshold slider (kalau ada)
thr_series = runs_filtered["threshold"].dropna()
thr_range = None
if not thr_series.empty:
    thr_min, thr_max = float(thr_series.min()), float(thr_series.max())
    thr_range = st.sidebar.slider(
        "Range Threshold",
        min_value=thr_min,
        max_value=thr_max,
        value=(thr_min, thr_max),
    )

# window_days selector (kalau ada)
win_series = runs_filtered["window_days"].dropna()
sel_windows = None
if not win_series.empty:
    win_list = sorted(win_series.astype(int).unique().tolist())
    sel_windows = st.sidebar.multiselect("Window (hari)", options=win_list, default=win_list)

# apply filters
if thr_range is not None:
    runs_filtered = runs_filtered[
        (runs_filtered["threshold"].fillna(-1) >= thr_range[0]) &
        (runs_filtered["threshold"].fillna(-1) <= thr_range[1])
    ].copy()

if sel_windows is not None and len(sel_windows) > 0:
    runs_filtered = runs_filtered[runs_filtered["window_days"].isin(pd.Series(sel_windows, dtype="Int64"))].copy()

# max_show hanya untuk tampilan list ringkas (BUKAN untuk analisis)
max_show = st.sidebar.number_input("Maks run ditampilkan (tabel ringkas)", min_value=5, max_value=2000, value=90, step=5)

runs_view_head = runs_filtered.head(int(max_show)).copy()   # untuk tabel ringkas
runs_view_all = runs_filtered.copy()                        # untuk opsi analisis (semua hasil filter)

# ======================================================
# Pilih Run untuk Dibandingkan
# ======================================================
st.subheader("Pilih Run untuk Dibandingkan")

# siapkan default pilihan = 9 run terbaru (sesuai filter)
default_ids_9 = runs_view_all["run_id"].head(min(DEFAULT_PICK_N, len(runs_view_all))).tolist()

# simpan pilihan user di session_state supaya tombol bisa mengubah selection
if "sel_run_ids_eval" not in st.session_state:
    st.session_state["sel_run_ids_eval"] = default_ids_9

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    if st.button("✅ Pilih semua run (sesuai filter)"):
        st.session_state["sel_run_ids_eval"] = runs_view_all["run_id"].tolist()
with colB:
    if st.button(f"↩️ Reset default ({DEFAULT_PICK_N})"):
        st.session_state["sel_run_ids_eval"] = default_ids_9
with colC:
    st.caption("Catatan: 'sesuai filter' = mengikuti Approach/Range Threshold/Window. "
               "'Maks run ditampilkan' hanya membatasi tabel ringkas, bukan analisis.")

sel_run_ids = st.multiselect(
    "Run ID",
    options=runs_view_all["run_id"].tolist(),
    default=st.session_state["sel_run_ids_eval"],
    help="Pilih beberapa run_id untuk analisis. Untuk heatmap, idealnya ada beberapa kombinasi threshold × window_days.",
    key="sel_run_ids_eval",
)

with st.expander("Daftar run (ringkas)", expanded=False):
    st.dataframe(
        runs_view_head[["run_time", "run_id", "approach", "threshold", "window_days"]],
        use_container_width=True,
        height=300,
    )

if not sel_run_ids:
    st.info("Pilih minimal 1 run_id untuk menampilkan evaluasi.")
    st.stop()

# ======================================================
# 🧰 Auto-generate grid eksperimen (threshold × window_days)
# ======================================================
st.subheader("🧰 Auto-generate Grid Eksperimen (Threshold × Window)")

with st.expander("Buka generator command eksperimen", expanded=False):
    st.write("Generator ini membuat perintah untuk menjalankan banyak eksperimen kombinasi **threshold × window_days**.")
    st.caption("Tips: gunakan `--dry-run` untuk uji cepat koneksi/DDL tanpa insert besar.")

    colA, colB, colC = st.columns(3)

    with colA:
        mode_shell = st.selectbox("Format Shell", ["Windows CMD", "PowerShell", "Bash (Linux/Mac)"], index=0)
        run_script = st.text_input("Nama script", value="python run_modeling.py")

    with colB:
        thr_start = st.number_input("Threshold mulai", value=0.65, step=0.01, format="%.2f")
        thr_end = st.number_input("Threshold akhir", value=0.85, step=0.01, format="%.2f")
        thr_step = st.number_input("Step threshold", value=0.05, step=0.01, format="%.2f")

    with colC:
        windows_str = st.text_input("Daftar window_days (pisahkan koma)", value="7,14,30")
        min_cluster_size = st.number_input("min_cluster_size", value=2, min_value=1, max_value=50, step=1)

    colD, colE, colF = st.columns(3)
    with colD:
        use_dburl = st.checkbox("Sertakan --db-url", value=False)
        db_url = st.text_input("db-url", value='postgresql+psycopg2://user:pass@host:5432/db') if use_dburl else ""

    with colE:
        start_date = st.text_input("start (YYYY-MM-DD)", value="")
        end_date = st.text_input("end (YYYY-MM-DD)", value="")

    with colF:
        modul = st.text_input("modul (opsional)", value="")
        site = st.text_input("site (opsional)", value="")

    colG, colH, colI = st.columns(3)
    with colG:
        max_rows = st.text_input("max_rows (opsional)", value="")
        full_matrix_limit = st.number_input("full_matrix_limit", value=2500, min_value=100, max_value=20000, step=100)

    with colH:
        nn_topk = st.number_input("nn_topk (base)", value=30, min_value=5, max_value=500, step=5)
        dry_run = st.checkbox("Tambahkan --dry-run", value=False)

    with colI:
        max_features = st.number_input("max_features", value=20000, min_value=1000, max_value=200000, step=1000)
        ngram_min = st.number_input("ngram_min", value=1, min_value=1, max_value=5, step=1)
        ngram_max = st.number_input("ngram_max", value=2, min_value=1, max_value=5, step=1)

    def build_thresholds(a: float, b: float, step: float):
        if step <= 0:
            return []
        vals = []
        x = a
        while x <= b + 1e-9:
            vals.append(round(x, 4))
            x += step
        return vals

    thresholds = build_thresholds(float(thr_start), float(thr_end), float(thr_step))

    windows = []
    for part in str(windows_str).split(","):
        part = part.strip()
        if part:
            try:
                windows.append(int(part))
            except Exception:
                pass
    windows = sorted(list(set(windows)))

    if not thresholds or not windows:
        st.warning("Threshold atau window_days belum valid. Pastikan step > 0 dan window_days berupa angka.")
        st.stop()

    common_args = []
    if use_dburl and db_url.strip():
        common_args += ["--db-url", f'"{db_url.strip()}"' if " " in db_url.strip() else db_url.strip()]
    if start_date.strip():
        common_args += ["--start", start_date.strip()]
    if end_date.strip():
        common_args += ["--end", end_date.strip()]
    if modul.strip():
        common_args += ["--modul", f'"{modul.strip()}"' if " " in modul.strip() else modul.strip()]
    if site.strip():
        common_args += ["--site", f'"{site.strip()}"' if " " in site.strip() else site.strip()]
    if max_rows.strip():
        common_args += ["--max-rows", max_rows.strip()]

    common_args += [
        "--min-cluster-size", str(int(min_cluster_size)),
        "--full-matrix-limit", str(int(full_matrix_limit)),
        "--nn-topk", str(int(nn_topk)),
        "--max-features", str(int(max_features)),
        "--ngram-min", str(int(ngram_min)),
        "--ngram-max", str(int(ngram_max)),
    ]
    if dry_run:
        common_args += ["--dry-run"]

    commands = []
    for w in windows:
        for thr in thresholds:
            args = common_args + ["--window", str(int(w)), "--threshold", str(thr)]
            cmd = " ".join([run_script] + args)
            commands.append(cmd)

    st.markdown(f"**Total eksperimen:** `{len(commands):,}` (window={windows}, threshold={thresholds})")

    if mode_shell == "Windows CMD":
        rendered = []
        for cmd in commands:
            toks = cmd.split()
            if len(toks) <= 1:
                rendered.append(cmd)
                continue
            base = toks[0] + " " + toks[1] if toks[0].lower() == "python" and len(toks) > 1 else toks[0]
            rest = toks[2:] if toks[0].lower() == "python" and len(toks) > 2 else toks[1:]
            lines = [base + " ^"]
            for t in rest[:-1]:
                lines.append(f"  {t} ^")
            lines.append(f"  {rest[-1]}")
            rendered.append("\n".join(lines))
        out = "\n\n".join(rendered)

    elif mode_shell == "PowerShell":
        rendered = []
        for cmd in commands:
            toks = cmd.split()
            base = toks[0] + " " + toks[1] if toks[0].lower() == "python" and len(toks) > 1 else toks[0]
            rest = toks[2:] if toks[0].lower() == "python" and len(toks) > 2 else toks[1:]
            lines = [base + " `"]
            for t in rest[:-1]:
                lines.append(f"  {t} `")
            lines.append(f"  {rest[-1]}")
            rendered.append("\n".join(lines))
        out = "\n\n".join(rendered)

    else:
        out = "\n".join(commands)

    st.code(out, language="bash" if mode_shell == "Bash (Linux/Mac)" else "text")

    st.markdown("### Versi loop (lebih ringkas)")
    if mode_shell == "Bash (Linux/Mac)":
        thr_list = " ".join([str(t) for t in thresholds])
        win_list = " ".join([str(w) for w in windows])
        base = " ".join([run_script] + common_args).strip()
        loop = (
            f"for w in {win_list}; do\n"
            f"  for t in {thr_list}; do\n"
            f"    {base} --window $w --threshold $t\n"
            f"  done\n"
            f"done"
        )
        st.code(loop, language="bash")
    elif mode_shell == "PowerShell":
        thr_list = ", ".join([str(t) for t in thresholds])
        win_list = ", ".join([str(w) for w in windows])
        base = " ".join([run_script] + common_args).strip()
        loop = (
            f"$windows = @({win_list})\n"
            f"$ths = @({thr_list})\n"
            f"foreach ($w in $windows) {{\n"
            f"  foreach ($t in $ths) {{\n"
            f"    {base} --window $w --threshold $t\n"
            f"  }}\n"
            f"}}"
        )
        st.code(loop, language="powershell")
    else:
        thr_list = " ".join([str(t) for t in thresholds])
        win_list = " ".join([str(w) for w in windows])
        base = " ".join([run_script] + common_args).strip()
        loop = (
            f"for %%W in ({win_list}) do (\n"
            f"  for %%T in ({thr_list}) do (\n"
            f"    {base} --window %%W --threshold %%T\n"
            f"  )\n"
            f")"
        )
        st.code(loop, language="bat")

    st.caption("Catatan: Untuk dataset besar, lebih aman jalankan batch bertahap (mis. per window dulu, baru per threshold).")

# ======================================================
# 📈 Aggregates (untuk run terpilih)
# ======================================================
agg = load_aggregate_for_runs(SCHEMA, T_SUMMARY, T_MEMBERS, sel_run_ids)
runs_sel = runs_view_all[runs_view_all["run_id"].isin(sel_run_ids)].copy()

df_eval = runs_sel.merge(agg, on="run_id", how="left")
df_eval = df_eval.sort_values(["window_days", "threshold", "run_time"], ascending=[True, True, False])

# KPI cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Run terpilih", f"{len(df_eval):,}")
c2.metric("Total cluster", f"{int(df_eval['clusters'].fillna(0).sum()):,}")
c3.metric("Total members", f"{int(df_eval['members'].fillna(0).sum()):,}")
c4.metric("Rata-rata singleton (%)", f"{df_eval['singleton_pct'].fillna(0).mean():.2f}")

st.subheader("Ringkasan Per Run")
st.dataframe(
    df_eval[[
        "run_time", "run_id", "threshold", "window_days",
        "clusters", "members", "avg_cluster_size", "median_cluster_size", "singleton_pct"
    ]],
    use_container_width=True,
    height=340
)

# ======================================================
# 🔥 Heatmap (threshold x window_days)
# ======================================================
st.subheader("Heatmap: Threshold × Window (hari)")

metric_opt = st.selectbox(
    "Metrik heatmap",
    options=[
        ("clusters", "Jumlah Cluster"),
        ("avg_cluster_size", "Rata-rata Ukuran Cluster"),
        ("singleton_pct", "% Singleton Cluster"),
        ("members", "Jumlah Members"),
    ],
    format_func=lambda x: x[1],
)
value_col, title = metric_opt[0], metric_opt[1]
_heatmap(df_eval, value_col=value_col, title=title)

# ======================================================
# 📉 Line charts (lebih detail)
# ======================================================
st.subheader("Grafik Detail")
mode = st.radio(
    "Mode Grafik",
    options=["Per Window (garis per window_days)", "Per Threshold (garis per threshold)"],
    horizontal=True
)

plot_df = df_eval.dropna(subset=["threshold", "window_days"]).copy()
plot_df["window_days"] = plot_df["window_days"].astype(int)

y_metric = st.selectbox(
    "Metrik sumbu-Y",
    options=[
        ("clusters", "Jumlah Cluster"),
        ("avg_cluster_size", "Rata-rata Ukuran Cluster"),
        ("median_cluster_size", "Median Ukuran Cluster"),
        ("singleton_pct", "% Singleton Cluster"),
        ("members", "Jumlah Members"),
    ],
    format_func=lambda x: x[1]
)
y_col = y_metric[0]

if plot_df.empty:
    st.info("Tidak ada data threshold/window_days yang lengkap untuk grafik.")
else:
    if mode.startswith("Per Window"):
        ch = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("threshold:Q", title="Threshold"),
                y=alt.Y(f"{y_col}:Q", title=y_metric[1]),
                color=alt.Color("window_days:N", title="Window (hari)"),
                tooltip=["run_id", "run_time", "threshold", "window_days", y_col, "clusters", "members", "singleton_pct"]
            )
            .properties(height=280)
        )
        st.altair_chart(ch, use_container_width=True)
    else:
        ch = (
            alt.Chart(plot_df)
            .mark_line(point=True)
            .encode(
                x=alt.X("window_days:O", title="Window (hari)"),
                y=alt.Y(f"{y_col}:Q", title=y_metric[1]),
                color=alt.Color("threshold:N", title="Threshold"),
                tooltip=["run_id", "run_time", "threshold", "window_days", y_col, "clusters", "members", "singleton_pct"]
            )
            .properties(height=280)
        )
        st.altair_chart(ch, use_container_width=True)

# ======================================================
# 📚 Modul breakdown (opsional)
# ======================================================
st.subheader("Breakdown per Modul (opsional)")
min_clusters = st.number_input("Minimal cluster per modul (filter)", min_value=1, max_value=500, value=10, step=1)

modul_df = load_modul_breakdown(SCHEMA, T_SUMMARY, sel_run_ids, min_clusters=int(min_clusters))
if modul_df.empty:
    st.info("Tidak ada data modul breakdown (atau semua modul cluster-nya < filter minimal).")
else:
    modul_list = sorted([m for m in modul_df["modul"].dropna().unique().tolist() if str(m).strip() != ""])
    sel_modul = st.selectbox("Pilih modul", options=["(semua modul)"] + modul_list, index=0)

    view = modul_df.copy()
    if sel_modul != "(semua modul)":
        view = view[view["modul"] == sel_modul].copy()

    view2 = view.merge(
        runs_sel[["run_id", "threshold", "window_days", "run_time"]],
        on="run_id",
        how="left"
    )
    view2["threshold"] = pd.to_numeric(view2["threshold"], errors="coerce")
    view2["window_days"] = pd.to_numeric(view2["window_days"], errors="coerce").astype("Int64")

    st.dataframe(view2, use_container_width=True, height=320)

    st.markdown("**Heatmap per modul (clusters)**")
    _heatmap(view2.rename(columns={"clusters": "clusters_modul"}), value_col="clusters_modul", title="Clusters (Modul)")

# ======================================================
# 🧾 Detail params & data_range per run
# ======================================================
st.subheader("Detail Parameter Run")
with st.expander("Lihat params_json & data_range", expanded=False):
    for _, r in df_eval.iterrows():
        st.markdown(
            f"**run_id:** `{r['run_id']}`  \n"
            f"**threshold:** `{r['threshold']}`  \n"
            f"**window_days:** `{r['window_days']}`  \n"
            f"**run_time:** `{r['run_time']}`"
        )
        st.write("params_json:", r.get("params_json"))
        st.write("data_range:", r.get("data_range"))
        st.divider()
