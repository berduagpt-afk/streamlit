# pages/evaluation_labeling_utility.py
from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)

# ======================================================
# 🔐 Guard Login (opsional)
# ======================================================
if not st.session_state.get("logged_in", True):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

#st.set_page_config(page_title="Evaluasi Label & Utility Prediksi", layout="wide")

SCHEMA = "lasis_djp"

# --- Tables ---
T_SYN_SUM = f"{SCHEMA}.modeling_sintaksis_temporal_summary"
T_SYN_MEM = f"{SCHEMA}.modeling_sintaksis_temporal_members"

T_SEM_SUM = f"{SCHEMA}.modeling_semantik_temporal_summary"
T_SEM_MEM = f"{SCHEMA}.modeling_semantik_temporal_members"


# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@"
        f"{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )


# ======================================================
# 🧠 Helpers: load runs
# ======================================================
@st.cache_data(show_spinner=False, ttl=300)
def load_sintaksis_runs():
    eng = get_engine()
    q = f"""
    SELECT DISTINCT job_id, modeling_id, window_days
    FROM {T_SYN_MEM}
    ORDER BY window_days, job_id, modeling_id
    """
    return pd.read_sql(q, eng)


@st.cache_data(show_spinner=False, ttl=300)
def load_semantik_runs():
    eng = get_engine()
    q = f"""
    SELECT DISTINCT modeling_id, window_days, time_col, include_noise
    FROM {T_SEM_SUM}
    ORDER BY window_days, modeling_id, time_col, include_noise
    """
    return pd.read_sql(q, eng)


# ======================================================
# 🏷️ Label builder (rule-based + temporal)
# Definisi:
# - episode valid = (cluster_id, temporal_cluster_no) dengan n_tickets > min_tickets_per_episode
# - cluster recurrent kuat = n_valid_episodes >= min_valid_episodes
# Label tiket: 1 jika cluster_id masuk recurrent kuat; else 0
# ======================================================
@st.cache_data(show_spinner=False, ttl=300)
def build_labels_sintaksis(job_id: str, modeling_id: str, window_days: int,
                          min_tickets_per_episode: int, min_valid_episodes: int):
    eng = get_engine()

    # ✅ PATCH: gunakan CAST(:param AS uuid), bukan :param::uuid
    q_mem = text(f"""
        SELECT
            job_id, modeling_id, window_days, cluster_id, incident_number,
            tgl_submit, site, assignee, modul, sub_modul, gap_days,
            temporal_cluster_no, temporal_cluster_id
        FROM {T_SYN_MEM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :wd
    """)
    mem = pd.read_sql(
        q_mem, eng,
        params={"job_id": job_id, "modeling_id": modeling_id, "wd": int(window_days)}
    )
    if mem.empty:
        return mem, pd.DataFrame(), pd.DataFrame()

    # episode counts
    ep = (
        mem.groupby(["cluster_id", "temporal_cluster_no"], as_index=False)
           .agg(n_tickets=("incident_number", "count"))
    )
    ep_valid = ep[ep["n_tickets"] > int(min_tickets_per_episode)].copy()

    # cluster stats: number of valid episodes
    cl = (
        ep_valid.groupby("cluster_id", as_index=False)
                .agg(n_valid_episodes=("temporal_cluster_no", "nunique"),
                     total_tickets_valid=("n_tickets", "sum"),
                     avg_tickets_per_valid_episode=("n_tickets", "mean"))
    )
    cl["is_recurrent_cluster"] = (cl["n_valid_episodes"] >= int(min_valid_episodes)).astype(int)

    # label per ticket (join by cluster_id)
    out = mem.merge(cl[["cluster_id", "is_recurrent_cluster", "n_valid_episodes"]],
                    on="cluster_id", how="left")
    out["is_recurrent_cluster"] = out["is_recurrent_cluster"].fillna(0).astype(int)
    out["n_valid_episodes"] = out["n_valid_episodes"].fillna(0).astype(int)
    out = out.rename(columns={"is_recurrent_cluster": "is_recurrent"})
    out["approach"] = "sintaksis"
    out["event_time"] = pd.to_datetime(out["tgl_submit"], errors="coerce")

    return out, ep_valid, cl


@st.cache_data(show_spinner=False, ttl=300)
def load_sintaksis_summary(job_id: str, modeling_id: str, window_days: int):
    eng = get_engine()
    # ✅ PATCH
    q = text(f"""
        SELECT *
        FROM {T_SYN_SUM}
        WHERE job_id = CAST(:job_id AS uuid)
          AND modeling_id = CAST(:mid AS uuid)
          AND window_days = :wd
        LIMIT 1
    """)
    return pd.read_sql(q, eng, params={"job_id": job_id, "mid": modeling_id, "wd": int(window_days)})


@st.cache_data(show_spinner=False, ttl=300)
def build_labels_semantik(modeling_id: str, window_days: int, time_col: str, include_noise: bool,
                          min_tickets_per_episode: int, min_valid_episodes: int):
    eng = get_engine()

    # ✅ PATCH: CAST(:mid AS uuid)
    # ⚠️ Kolom di members semantik bisa sedikit berbeda tergantung DDL Anda.
    # Kode ini mengasumsikan ada kolom: modeling_id, window_days, cluster_id, incident_number,
    # time_col, event_time, site, assignee, modul, sub_modul, gap_days, temporal_cluster_no, temporal_cluster_id.
    q_mem = text(f"""
        SELECT
            modeling_id, window_days, cluster_id, incident_number,
            time_col, event_time, site, assignee, modul, sub_modul, gap_days,
            temporal_cluster_no, temporal_cluster_id
        FROM {T_SEM_MEM}
        WHERE modeling_id = CAST(:mid AS uuid)
          AND window_days = :wd
          AND time_col = :tcol
    """)
    mem = pd.read_sql(q_mem, eng, params={"mid": modeling_id, "wd": int(window_days), "tcol": str(time_col)})
    if mem.empty:
        return mem, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # episode counts
    ep = (
        mem.groupby(["cluster_id", "temporal_cluster_no"], as_index=False)
           .agg(n_tickets=("incident_number", "count"))
    )
    ep_valid = ep[ep["n_tickets"] > int(min_tickets_per_episode)].copy()

    # cluster stats
    cl = (
        ep_valid.groupby("cluster_id", as_index=False)
                .agg(n_valid_episodes=("temporal_cluster_no", "nunique"),
                     total_tickets_valid=("n_tickets", "sum"),
                     avg_tickets_per_valid_episode=("n_tickets", "mean"))
    )
    cl["is_recurrent_cluster"] = (cl["n_valid_episodes"] >= int(min_valid_episodes)).astype(int)

    out = mem.merge(cl[["cluster_id", "is_recurrent_cluster", "n_valid_episodes"]],
                    on="cluster_id", how="left")
    out["is_recurrent_cluster"] = out["is_recurrent_cluster"].fillna(0).astype(int)
    out["n_valid_episodes"] = out["n_valid_episodes"].fillna(0).astype(int)
    out = out.rename(columns={"is_recurrent_cluster": "is_recurrent"})
    out["approach"] = "semantik"

    # summary row untuk validasi temporal stable/split
    # ✅ PATCH
    q_sum = text(f"""
        SELECT *
        FROM {T_SEM_SUM}
        WHERE modeling_id = CAST(:mid AS uuid)
          AND window_days = :wd
          AND time_col = :tcol
          AND include_noise = :inc
        LIMIT 1
    """)
    summ = pd.read_sql(q_sum, eng, params={"mid": modeling_id, "wd": int(window_days), "tcol": str(time_col), "inc": bool(include_noise)})

    return out, ep_valid, cl, summ


# ======================================================
# 📈 Prediction (utility test) - time-based split
# ======================================================
def build_features(df: pd.DataFrame, time_col: str = "event_time") -> pd.DataFrame:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    d["dow"] = d[time_col].dt.dayofweek
    d["month"] = d[time_col].dt.month
    d["hour"] = d[time_col].dt.hour
    d["has_time"] = d[time_col].notna().astype(int)

    # Categorical columns (fallback if missing)
    for c in ["site", "assignee", "modul", "sub_modul"]:
        if c not in d.columns:
            d[c] = np.nan

    keep = ["site", "assignee", "modul", "sub_modul", "dow", "month", "hour", "has_time", "is_recurrent", time_col]
    return d[keep]


def train_eval_time_split(df_feat: pd.DataFrame, time_col: str, test_ratio: float = 0.2):
    df = df_feat.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
    if df.empty or df["is_recurrent"].nunique() < 2:
        return None, "Data tidak cukup (kolom waktu kosong atau label hanya satu kelas)."

    n = len(df)
    split_idx = int(np.floor((1 - test_ratio) * n))
    split_idx = max(1, min(split_idx, n - 1))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    X_train = train.drop(columns=["is_recurrent"])
    y_train = train["is_recurrent"].astype(int)

    X_test = test.drop(columns=["is_recurrent"])
    y_test = test["is_recurrent"].astype(int)

    cat_cols = ["site", "assignee", "modul", "sub_modul"]
    num_cols = ["dow", "month", "hour", "has_time"]

    pre = ColumnTransformer(
        transformers=[
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), cat_cols),
            ("num", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
            ]), num_cols),
        ],
        remainder="drop"
    )

    clf = LogisticRegression(
        max_iter=200,
        class_weight="balanced"
    )

    pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_score = pipe.predict_proba(X_test)[:, 1]

    rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    ap = average_precision_score(y_test, y_score)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])

    prec, rec, _thr = precision_recall_curve(y_test, y_score)
    pr_df = pd.DataFrame({"precision": prec, "recall": rec})

    meta = {
        "n_train": len(train),
        "n_test": len(test),
        "test_start": test[time_col].min(),
        "test_end": test[time_col].max(),
        "avg_precision": ap,
        "report": rep,
    }
    return (pipe, meta, cm_df, pr_df), None


# ======================================================
# UI
# ======================================================
st.title("Evaluasi Label (Ringkas) + Utility Prediksi (Time-based)")

with st.sidebar:
    st.header("Konfigurasi")

    approach = st.radio("Pendekatan", ["sintaksis", "semantik"], horizontal=True)
    window_days = st.number_input("window_days", min_value=1, max_value=365, value=7, step=1)

    st.subheader("Aturan Label (Episode Valid)")
    min_tickets_per_episode = st.number_input("Min tiket per episode (>)", min_value=0, max_value=1000, value=1, step=1)
    min_valid_episodes = st.number_input("Min episode valid per cluster (≥)", min_value=1, max_value=50, value=2, step=1)

    st.subheader("Prediksi (Time-based split)")
    test_ratio = st.slider("Porsi test (akhir periode)", min_value=0.1, max_value=0.4, value=0.2, step=0.05)


# Load selector based on approach
if approach == "sintaksis":
    runs = load_sintaksis_runs()
    if runs.empty:
        st.warning("Tidak ada data runs sintaksis pada tabel temporal_members.")
        st.stop()

    with st.sidebar:
        st.subheader("Run Sintaksis")
        job_id = st.selectbox("job_id", runs["job_id"].astype(str).unique())
        mids = runs.loc[runs["job_id"].astype(str) == job_id, "modeling_id"].astype(str).unique()
        modeling_id = st.selectbox("modeling_id", mids)

    df_lab, df_ep_valid, df_cluster_stats = build_labels_sintaksis(
        job_id=job_id,
        modeling_id=modeling_id,
        window_days=int(window_days),
        min_tickets_per_episode=int(min_tickets_per_episode),
        min_valid_episodes=int(min_valid_episodes),
    )
    df_sum = load_sintaksis_summary(job_id, modeling_id, int(window_days))
    time_col_name = "event_time"

else:
    runs = load_semantik_runs()
    if runs.empty:
        st.warning("Tidak ada data runs semantik pada tabel temporal_summary.")
        st.stop()

    with st.sidebar:
        st.subheader("Run Semantik")
        modeling_id = st.selectbox("modeling_id", runs["modeling_id"].astype(str).unique())
        tcols = runs.loc[runs["modeling_id"].astype(str) == modeling_id, "time_col"].astype(str).unique()
        time_col = st.selectbox("time_col", tcols)
        inc_vals = runs.loc[
            (runs["modeling_id"].astype(str) == modeling_id) & (runs["time_col"].astype(str) == time_col),
            "include_noise"
        ].unique()
        include_noise = st.selectbox("include_noise", list(map(bool, inc_vals)))

    df_lab, df_ep_valid, df_cluster_stats, df_sum = build_labels_semantik(
        modeling_id=modeling_id,
        window_days=int(window_days),
        time_col=time_col,
        include_noise=bool(include_noise),
        min_tickets_per_episode=int(min_tickets_per_episode),
        min_valid_episodes=int(min_valid_episodes),
    )
    time_col_name = "event_time"

if df_lab.empty:
    st.warning("Tidak ada data label untuk konfigurasi ini. Periksa parameter run/window_days.")
    st.stop()

# ======================================================
# SECTION 1: Statistik Label
# ======================================================
st.subheader("1) Statistik Label")
c1, c2, c3, c4 = st.columns(4)

n_total = len(df_lab)
n_pos = int(df_lab["is_recurrent"].sum())
n_neg = n_total - n_pos
pos_rate = (n_pos / n_total) if n_total else 0.0

n_clusters_total = df_lab["cluster_id"].nunique()
weak = int((df_cluster_stats["n_valid_episodes"] == 1).sum()) if (df_cluster_stats is not None and not df_cluster_stats.empty and "n_valid_episodes" in df_cluster_stats.columns) else 0
strong = int((df_cluster_stats["n_valid_episodes"] >= int(min_valid_episodes)).sum()) if (df_cluster_stats is not None and not df_cluster_stats.empty and "n_valid_episodes" in df_cluster_stats.columns) else 0

c1.metric("Total tiket", f"{n_total:,}")
c2.metric("Tiket recurrent (1)", f"{n_pos:,}", f"{pos_rate*100:.2f}%")
c3.metric("Tiket non-recurrent (0)", f"{n_neg:,}")
c4.metric("Cluster recurrent kuat", f"{strong:,}")

with st.expander("Detail statistik (tabel)"):
    st.dataframe(pd.DataFrame([{
        "approach": approach,
        "window_days": int(window_days),
        "min_tickets_per_episode(>)": int(min_tickets_per_episode),
        "min_valid_episodes(≥)": int(min_valid_episodes),
        "n_tickets_total": n_total,
        "n_tickets_recurrent": n_pos,
        "prop_recurrent": pos_rate,
        "n_clusters_total": int(n_clusters_total),
        "n_clusters_recurrent_strong": strong,
        "n_clusters_recurrent_weak": weak,
    }]), use_container_width=True, hide_index=True)

dist_df = pd.DataFrame({"label": ["0", "1"], "count": [n_neg, n_pos]})
st.altair_chart(
    alt.Chart(dist_df).mark_bar().encode(
        x=alt.X("label:N", title="Label is_recurrent"),
        y=alt.Y("count:Q", title="Jumlah tiket")
    ).properties(height=250),
    use_container_width=True
)

# ======================================================
# SECTION 2: Validasi Temporal
# ======================================================
st.subheader("2) Validasi Temporal sebagai Justifikasi Label")

left, right = st.columns([1.1, 0.9])

with left:
    st.markdown("**Ringkasan temporal (stable vs split)**")
    if df_sum is not None and not df_sum.empty:
        keys = [
            "n_clusters_eligible", "n_clusters_split", "prop_clusters_split",
            "n_clusters_stable", "prop_clusters_stable", "total_episodes",
            "avg_episode_per_cluster", "median_episode_per_cluster", "run_time"
        ]
        show = df_sum[[c for c in keys if c in df_sum.columns]].copy()
        st.dataframe(show, use_container_width=True, hide_index=True)
    else:
        st.info("Tabel summary tidak ditemukan/empty untuk konfigurasi ini.")

with right:
    st.markdown("**Distribusi jumlah episode valid per cluster**")
    if df_cluster_stats is not None and not df_cluster_stats.empty and "n_valid_episodes" in df_cluster_stats.columns:
        tmp = df_cluster_stats.copy()
        tmp["bucket"] = tmp["n_valid_episodes"].astype(int).astype(str)
        st.altair_chart(
            alt.Chart(tmp).mark_bar().encode(
                x=alt.X("bucket:N", title="n_valid_episodes per cluster"),
                y=alt.Y("count():Q", title="Jumlah cluster")
            ).properties(height=250),
            use_container_width=True
        )
    else:
        st.info("Tidak ada data episode valid yang dapat dihitung.")

st.markdown("**Analisis gap_days (opsional) pada tiket recurrent**")
gap_df = df_lab.loc[df_lab["is_recurrent"] == 1, ["gap_days"]].dropna()
if not gap_df.empty:
    st.dataframe(pd.DataFrame([{
        "n_rows_with_gap": int(len(gap_df)),
        "gap_days_mean": float(gap_df["gap_days"].mean()),
        "gap_days_median": float(gap_df["gap_days"].median()),
        "gap_days_p90": float(gap_df["gap_days"].quantile(0.9)),
    }]), use_container_width=True, hide_index=True)
else:
    st.caption("Tidak ada gap_days terisi untuk tiket recurrent pada konfigurasi ini.")

with st.expander("Episode valid (n_tickets > min) - detail"):
    if df_ep_valid is not None and not df_ep_valid.empty:
        st.dataframe(
            df_ep_valid.sort_values(["cluster_id", "temporal_cluster_no"]),
            use_container_width=True, hide_index=True
        )
    else:
        st.write("Tidak ada episode yang memenuhi syarat.")

with st.expander("Cluster stats (episode valid)"):
    if df_cluster_stats is not None and not df_cluster_stats.empty:
        st.dataframe(
            df_cluster_stats.sort_values(["n_valid_episodes", "total_tickets_valid"], ascending=False),
            use_container_width=True, hide_index=True
        )
    else:
        st.write("Tidak ada statistik cluster.")

# ======================================================
# SECTION 3: Utility Prediksi
# ======================================================
st.subheader("3) Time-based split + Prediksi sebagai Uji Kegunaan Label")

df_feat = build_features(df_lab, time_col=time_col_name)

with st.expander("Preview dataset prediksi (fitur + label)"):
    st.dataframe(df_feat.head(50), use_container_width=True, hide_index=True)

run_pred = st.button("Jalankan baseline prediksi (LogReg, class_weight=balanced)")
if run_pred:
    result, err = train_eval_time_split(df_feat, time_col=time_col_name, test_ratio=float(test_ratio))
    if err:
        st.error(err)
    else:
        _pipe, meta, cm_df, pr_df = result

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train size", f"{meta['n_train']:,}")
        c2.metric("Test size", f"{meta['n_test']:,}")
        c3.metric("PR-AUC (Avg Precision)", f"{meta['avg_precision']:.4f}")
        c4.metric("Test period", f"{meta['test_start'].date()} → {meta['test_end'].date()}")

        st.markdown("**Confusion Matrix (test set)**")
        st.dataframe(cm_df)

        rep = meta["report"]
        cls1 = rep.get("1", {})
        st.markdown("**Metrik fokus kelas recurrent (1)**")
        st.dataframe(pd.DataFrame([{
            "precision_1": cls1.get("precision", 0.0),
            "recall_1": cls1.get("recall", 0.0),
            "f1_1": cls1.get("f1-score", 0.0),
            "support_1": int(cls1.get("support", 0)),
        }]), use_container_width=True, hide_index=True)

        st.markdown("**Precision–Recall Curve (test set)**")
        st.altair_chart(
            alt.Chart(pr_df).mark_line().encode(
                x=alt.X("recall:Q", title="Recall"),
                y=alt.Y("precision:Q", title="Precision")
            ).properties(height=280),
            use_container_width=True
        )

        st.info(
            "Catatan: baseline ini diposisikan sebagai uji kegunaan (utility test) label operasional. "
            "Jika Anda ingin prediksi berbasis teks, dataset harus menyediakan kolom judul/deskripsi."
        )

st.caption("Halaman ini menampilkan statistik label + justifikasi temporal + utility test prediksi berbasis time-split.")
