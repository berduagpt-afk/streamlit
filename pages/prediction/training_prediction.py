# pages/prediction/training_prediction.py
# ======================================================
# Training Prediksi Insiden Berulang (Supervised)
# - Model: LogReg (baseline), RandomForest, XGBoost (opsional), SVM (LinearSVC)
# - Evaluasi utama: PR-AUC (Average Precision)
# - Threshold tuning: dilakukan di VALIDATION set (lebih akademis ketat)
# - TEST set: evaluasi final (tanpa "mengintip" saat tuning)
# ======================================================

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# XGBoost (opsional)
try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False

# ======================================================
# 🔐 Guard Login (opsional - sesuaikan dengan sistem Anda)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# st.set_page_config(page_title="Training Prediksi Insiden Berulang", layout="wide")

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


def fmt_float(x, nd=4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "-"


def safe_cols(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [c for c in cols if c in df.columns]


def make_ohe():
    """
    Kompatibel scikit-learn lama & baru:
    - sklearn lama: OneHotEncoder(..., sparse=True/False)
    - sklearn baru: OneHotEncoder(..., sparse_output=True/False)
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def predict_score(pipe: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Menghasilkan skor kontinu untuk kelas positif (label=1).
    - Jika ada predict_proba -> ambil proba[:,1]
    - Jika ada decision_function -> sigmoid(decision) agar berada di (0,1)
    """
    if hasattr(pipe, "predict_proba"):
        p = pipe.predict_proba(X)
        return np.asarray(p[:, 1], dtype=float)

    if hasattr(pipe, "decision_function"):
        s = pipe.decision_function(X)
        s = np.asarray(s, dtype=float)
        return 1.0 / (1.0 + np.exp(-s))  # sigmoid

    return np.asarray(pipe.predict(X), dtype=float)


def build_pr_df(y_true: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    prec, rec, thr = precision_recall_curve(y_true, y_score)
    df = pd.DataFrame({"precision": prec, "recall": rec})
    df["threshold"] = np.append(thr, np.nan)
    return df


def choose_threshold_from_pr(
    pr_df: pd.DataFrame,
    mode: str,
    min_precision: float,
    min_recall: float,
    manual_thr: float,
) -> float:
    df = pr_df.dropna(subset=["threshold"]).copy()
    if df.empty:
        return 0.5

    if mode == "Manual":
        return float(manual_thr)

    if mode == "Max F1":
        f1 = 2 * (df["precision"] * df["recall"]) / (df["precision"] + df["recall"] + 1e-12)
        idx = int(f1.idxmax())
        return float(df.loc[idx, "threshold"])

    if mode == "Max Recall @ Precision≥p":
        df2 = df[df["precision"] >= float(min_precision)].copy()
        if df2.empty:
            return choose_threshold_from_pr(pr_df, "Max F1", min_precision, min_recall, manual_thr)
        idx = int(df2["recall"].idxmax())
        return float(df2.loc[idx, "threshold"])

    if mode == "Max Precision @ Recall≥r":
        df2 = df[df["recall"] >= float(min_recall)].copy()
        if df2.empty:
            return choose_threshold_from_pr(pr_df, "Max F1", min_precision, min_recall, manual_thr)
        idx = int(df2["precision"].idxmax())
        return float(df2.loc[idx, "threshold"])

    return 0.5


def confusion_and_report(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Tuple[pd.DataFrame, str]:
    y_pred = (y_score >= float(thr)).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"])
    rep = classification_report(y_true, y_pred, digits=4)
    return cm_df, rep


def pr_chart(pr_df: pd.DataFrame, thr_used: float, title: str) -> alt.Chart:
    """
    PR curve + titik threshold (nearest) — TANPA facet, jadi aman untuk layering.
    """
    base = (
        alt.Chart(pr_df)
        .mark_line()
        .encode(
            x=alt.X("recall:Q", title="Recall"),
            y=alt.Y("precision:Q", title="Precision"),
            tooltip=["precision", "recall", "threshold"],
        )
        .properties(height=360, title=title)
    )

    df_thr = pr_df.dropna(subset=["threshold"]).copy()
    if not df_thr.empty:
        df_thr["diff"] = (df_thr["threshold"] - float(thr_used)).abs()
        pt = df_thr.sort_values("diff").head(1)
        dot = alt.Chart(pt).mark_point(size=120).encode(x="recall:Q", y="precision:Q")
        return alt.layer(base, dot)

    return base


# ======================================================
# ✅ Page Header
# ======================================================
st.title("🤖 Training Prediksi Insiden Berulang")
st.caption(
    "Melatih model klasifikasi untuk memprediksi `label_berulang` dari `dataset_supervised`. "
    "Evaluasi utama menggunakan PR-AUC (Average Precision). Threshold ditentukan pada VALIDATION dan diuji pada TEST."
)

if not table_exists(SCHEMA, T_DS):
    st.error(f"Tabel `{SCHEMA}.{T_DS}` tidak ditemukan. Bangun dataset dulu (dataset_supervised_builder).")
    st.stop()

# ======================================================
# 🎛️ Filter dataset (jenis/model/window/time_col)
# ======================================================
with st.expander("⚙️ Filter Dataset", expanded=True):
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
        st.warning("dataset_supervised kosong.")
        st.stop()

    c1, c2, c3, c4, c5, c6 = st.columns([1.3, 2.1, 1.0, 1.2, 1.4, 1.4])

    with c1:
        jenis_opts = ["(all)"] + sorted(dims["jenis_pendekatan"].dropna().unique().tolist())
        jenis = st.selectbox("jenis_pendekatan", jenis_opts, 0)

    dims2 = dims if jenis == "(all)" else dims[dims["jenis_pendekatan"] == jenis]
    with c2:
        mid_opts = ["(all)"] + sorted(dims2["modeling_id"].dropna().unique().tolist())
        modeling_id = st.selectbox("modeling_id", mid_opts, 0)

    dims3 = dims2 if modeling_id == "(all)" else dims2[dims2["modeling_id"] == modeling_id]
    with c3:
        wd_opts = ["(all)"] + sorted(dims3["window_days"].dropna().astype(int).unique().tolist())
        window_days = st.selectbox("window_days", wd_opts, 0)

    dims4 = dims3 if window_days == "(all)" else dims3[dims3["window_days"].astype(int) == int(window_days)]
    with c4:
        tc_opts = ["(all)"] + sorted(dims4["time_col"].dropna().unique().tolist())
        time_col = st.selectbox("time_col", tc_opts, 0)

    with c5:
        exclude_noise = st.checkbox("Exclude noise (cluster_id=-1)", value=True)

    with c6:
        max_rows = st.number_input("Maks baris (0=semua)", min_value=0, max_value=5_000_000, value=0, step=5000)

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
if exclude_noise:
    where.append("(cluster_id IS NULL OR cluster_id <> -1)")

where_sql = " AND ".join(where)

# ======================================================
# 🧩 Fitur, split, threshold tuning
# ======================================================
with st.expander("🧩 Fitur, Split, Threshold Tuning", expanded=True):
    default_num = ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode", "temporal_cluster_no"]
    default_cat = ["site", "modul", "sub_modul", "assignee", "jenis_pendekatan"]

    c1, c2, c3 = st.columns([1.6, 1.8, 1.8])
    with c1:
        use_text = st.checkbox("Tambahkan fitur teks (TF-IDF)", value=False)
        st.caption("Jika aktif: gunakan `text_col_1` + `text_col_2` sebagai fitur TF-IDF.")
    with c2:
        eval_mode = st.radio(
            "Metode evaluasi",
            ["Gunakan split_name (train/test) + val dari train", "Stratified split (train/val/test)"],
            index=0,
            horizontal=True,
        )
    with c3:
        seed = st.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

    cc1, cc2, cc3 = st.columns([1.2, 1.2, 1.8])
    with cc1:
        test_size = st.slider("Proporsi test", 0.10, 0.40, 0.20, 0.05)
    with cc2:
        val_size = st.slider("Proporsi val (dari train)", 0.10, 0.40, 0.20, 0.05)
    with cc3:
        thr_mode = st.selectbox(
            "Strategi tuning threshold (berdasarkan VALIDATION)",
            ["Max Recall @ Precision≥p", "Max F1", "Max Precision @ Recall≥r", "Manual"],
            index=0,
        )

    ctp1, ctp2, ctp3 = st.columns([1.2, 1.2, 1.2])
    with ctp1:
        min_precision = st.slider("p (min precision)", 0.10, 0.95, 0.50, 0.05)
    with ctp2:
        min_recall = st.slider("r (min recall)", 0.10, 0.99, 0.80, 0.05)
    with ctp3:
        manual_thr = st.slider("Manual threshold", 0.01, 0.99, 0.50, 0.01)

# ======================================================
# 📥 Load data
# ======================================================
@st.cache_data(show_spinner=False, ttl=300)
def load_dataset(where_sql: str, params: Dict[str, Any], use_text: bool, max_rows: int) -> pd.DataFrame:
    cols = [
        "dataset_id",
        "jenis_pendekatan",
        "modeling_id",
        "window_days",
        "time_col",
        "incident_number",
        "event_time",
        "tgl_submit",
        "cluster_id",
        "temporal_cluster_no",
        "gap_days",
        "n_member_cluster",
        "n_episode_cluster",
        "n_member_episode",
        "site",
        "assignee",
        "modul",
        "sub_modul",
        "label_berulang",
        "split_name",
    ]
    if use_text:
        cols += ["text_col_1", "text_col_2"]

    sql = f"""
    SELECT {", ".join([f'"{c}"' for c in cols])}
    FROM {SCHEMA}."{T_DS}"
    WHERE {where_sql}
    ORDER BY COALESCE(event_time, tgl_submit) NULLS LAST
    """
    p = dict(params)
    if max_rows and max_rows > 0:
        sql += " LIMIT :lim"
        p["lim"] = int(max_rows)
    return qdf(sql, p)


with st.spinner("Memuat dataset supervised..."):
    df = load_dataset(where_sql, params, use_text, int(max_rows))

if df.empty:
    st.warning("Tidak ada data sesuai filter.")
    st.stop()

df["label_berulang"] = pd.to_numeric(df["label_berulang"], errors="coerce").fillna(0).astype(int)

pos = int((df["label_berulang"] == 1).sum())
neg = int((df["label_berulang"] == 0).sum())
st.info(f"Data: **{len(df):,}** | Positif (1): **{pos:,}** | Negatif (0): **{neg:,}**")

# Pastikan kolom teks ada jika user mengaktifkan TF-IDF
if use_text:
    if "text_col_1" not in df.columns:
        df["text_col_1"] = None
    if "text_col_2" not in df.columns:
        df["text_col_2"] = None

# ======================================================
# 🧠 Tentukan kolom fitur
# ======================================================
num_cols = safe_cols(df, ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode", "temporal_cluster_no"])
cat_cols = safe_cols(df, ["site", "modul", "sub_modul", "assignee", "jenis_pendekatan"])

# ======================================================
# 🧱 Preprocessor
# ======================================================
def build_preprocessor(num_cols: List[str], cat_cols: List[str], use_text: bool) -> ColumnTransformer:
    transformers = []

    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))

    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("ohe", make_ohe()),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))

    if use_text:

        def _merge_text(X: pd.DataFrame) -> np.ndarray:
            t1 = X["text_col_1"].fillna("").astype(str) if "text_col_1" in X else ""
            t2 = X["text_col_2"].fillna("").astype(str) if "text_col_2" in X else ""
            return (t1 + " " + t2).str.strip().values

        text_pipe = Pipeline(
            steps=[
                ("merge", FunctionTransformer(_merge_text, validate=False)),
                ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2)),
            ]
        )
        transformers.append(("txt", text_pipe, ["text_col_1", "text_col_2"]))

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.3)


preprocessor = build_preprocessor(num_cols, cat_cols, use_text)
svm_scaler = StandardScaler(with_mean=False)

# ======================================================
# 🎛️ Model & hyperparameter ringkas
# ======================================================
with st.expander("🤖 Model & Hyperparameter", expanded=False):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        lr_C = st.slider("LogReg C", 0.01, 10.0, 1.0, 0.01)
    with c2:
        rf_n = st.slider("RF n_estimators", 100, 2000, 400, 50)
    with c3:
        rf_depth = st.slider("RF max_depth (0=none)", 0, 60, 0, 5)
    with c4:
        svm_C = st.slider("LinearSVM C", 0.01, 10.0, 1.0, 0.01)

    if HAS_XGB:
        cx1, cx2, cx3, cx4 = st.columns(4)
        with cx1:
            xgb_n = st.slider("XGB n_estimators", 200, 3000, 600, 50)
        with cx2:
            xgb_lr = st.slider("XGB learning_rate", 0.01, 0.5, 0.1, 0.01)
        with cx3:
            xgb_depth = st.slider("XGB max_depth", 2, 12, 6, 1)
        with cx4:
            xgb_sub = st.slider("XGB subsample", 0.5, 1.0, 0.9, 0.05)
    else:
        st.warning("xgboost tidak terpasang → model XGBoost dinonaktifkan.")


def make_models(y_train_for_weight: np.ndarray) -> Dict[str, Any]:
    models: Dict[str, Any] = {}

    models["LogReg (balanced)"] = LogisticRegression(
        C=float(lr_C),
        max_iter=3000,
        class_weight="balanced",
        solver="liblinear",
    )

    models["RandomForest (balanced_subsample)"] = RandomForestClassifier(
        n_estimators=int(rf_n),
        max_depth=None if int(rf_depth) == 0 else int(rf_depth),
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=int(seed),
    )

    models["LinearSVM (balanced)"] = LinearSVC(
        C=float(svm_C),
        class_weight="balanced",
        random_state=int(seed),
    )

    if HAS_XGB:
        pos_ = max(1, int((y_train_for_weight == 1).sum()))
        neg_ = max(1, int((y_train_for_weight == 0).sum()))
        spw = neg_ / pos_

        models["XGBoost (scale_pos_weight)"] = XGBClassifier(
            n_estimators=int(xgb_n),
            learning_rate=float(xgb_lr),
            max_depth=int(xgb_depth),
            subsample=float(xgb_sub),
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=float(spw),
            n_jobs=-1,
            random_state=int(seed),
        )

    return models


# ======================================================
# 🧪 Split data: Train / Val / Test
# ======================================================
y_all = df["label_berulang"].values.astype(int)


def split_train_val_test(df_in: pd.DataFrame, y_in: np.ndarray):
    """
    Dua mode:
    1) split_name: test = split_name=='test', train_full = split_name=='train',
       lalu val di-split dari train_full secara stratified.
    2) stratified split: train_temp/test, lalu train/val dari train_temp.
    """
    if eval_mode.startswith("Gunakan split_name") and "split_name" in df_in.columns:
        train_full = df_in[df_in["split_name"].fillna("train") == "train"].copy()
        test_df = df_in[df_in["split_name"].fillna("train") == "test"].copy()

        if train_full.empty or test_df.empty:
            # fallback stratified
            train_temp, test_df2, y_train_temp, y_test2 = train_test_split(
                df_in, y_in, test_size=float(test_size), random_state=int(seed), stratify=y_in
            )
            train_df2, val_df2, y_train2, y_val2 = train_test_split(
                train_temp, y_train_temp, test_size=float(val_size), random_state=int(seed), stratify=y_train_temp
            )
            return train_df2, val_df2, test_df2, y_train2, y_val2, y_test2

        y_train_full = train_full["label_berulang"].values.astype(int)
        y_test2 = test_df["label_berulang"].values.astype(int)

        train_df2, val_df2, y_train2, y_val2 = train_test_split(
            train_full, y_train_full, test_size=float(val_size), random_state=int(seed), stratify=y_train_full
        )
        return train_df2, val_df2, test_df, y_train2, y_val2, y_test2

    # stratified split default
    train_temp, test_df2, y_train_temp, y_test2 = train_test_split(
        df_in, y_in, test_size=float(test_size), random_state=int(seed), stratify=y_in
    )
    train_df2, val_df2, y_train2, y_val2 = train_test_split(
        train_temp, y_train_temp, test_size=float(val_size), random_state=int(seed), stratify=y_train_temp
    )
    return train_df2, val_df2, test_df2, y_train2, y_val2, y_test2


train_df, val_df, test_df, y_train, y_val, y_test = split_train_val_test(df, y_all)

st.write(
    f"Split: train **{len(train_df):,}** (pos={int((y_train==1).sum()):,}) | "
    f"val **{len(val_df):,}** (pos={int((y_val==1).sum()):,}) | "
    f"test **{len(test_df):,}** (pos={int((y_test==1).sum()):,})"
)

# ======================================================
# 🏁 Train & Evaluate
# ======================================================
st.subheader("🏁 Training & Evaluasi")
run_btn = st.button("🚀 Train & Evaluate", type="primary")
if not run_btn:
    st.stop()

models = make_models(y_train)

results: List[Dict[str, Any]] = []
details: Dict[str, Dict[str, Any]] = {}

for name, clf in models.items():
    if "SVM" in name:
        pipe = Pipeline(steps=[("prep", preprocessor), ("scaler", svm_scaler), ("clf", clf)])
    else:
        pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    with st.spinner(f"Training {name}..."):
        pipe.fit(train_df, y_train)

    # score on val & test
    val_score = predict_score(pipe, val_df)
    test_score = predict_score(pipe, test_df)

    # PR-AUC
    pr_auc_val = average_precision_score(y_val, val_score)
    pr_auc_test = average_precision_score(y_test, test_score)

    # ROC-AUC (opsional)
    try:
        roc_auc_val = roc_auc_score(y_val, val_score)
        roc_auc_test = roc_auc_score(y_test, test_score)
    except Exception:
        roc_auc_val = np.nan
        roc_auc_test = np.nan

    # threshold tuning on VAL
    pr_df_val = build_pr_df(y_val, val_score)
    thr = choose_threshold_from_pr(
        pr_df_val, thr_mode, float(min_precision), float(min_recall), float(manual_thr)
    )

    # final on TEST with tuned threshold
    cm_df, rep = confusion_and_report(y_test, test_score, thr)

    results.append(
        dict(
            model=name,
            pr_auc_val=float(pr_auc_val),
            pr_auc_test=float(pr_auc_test),
            roc_auc_val=float(roc_auc_val) if np.isfinite(roc_auc_val) else None,
            roc_auc_test=float(roc_auc_test) if np.isfinite(roc_auc_test) else None,
            threshold_from_val=float(thr),
            n_test=int(len(y_test)),
            pos_test=int((y_test == 1).sum()),
        )
    )

    details[name] = dict(
        pipeline=pipe,
        val_pr_df=pr_df_val,
        val_score=val_score,
        test_score=test_score,
        cm=cm_df,
        report=rep,
    )

res_df = pd.DataFrame(results).sort_values("pr_auc_test", ascending=False).reset_index(drop=True)

st.subheader("📌 Ringkasan Hasil (urut PR-AUC TEST, threshold dari VAL)")
st.dataframe(res_df, use_container_width=True)

best_model = str(res_df.iloc[0]["model"])
st.success(f"Model terbaik (PR-AUC TEST tertinggi): **{best_model}**")

# ======================================================
# 📈 Kurva PR (VAL & TEST)
# ======================================================
st.subheader("📈 Precision–Recall Curve")

pick = st.selectbox("Pilih model untuk detail", res_df["model"].tolist(), index=0)
thr_used = float(res_df.loc[res_df["model"] == pick, "threshold_from_val"].iloc[0])

val_pr_df = details[pick]["val_pr_df"].copy()
test_score_pick = np.asarray(details[pick]["test_score"], dtype=float)
val_score_pick = np.asarray(details[pick]["val_score"], dtype=float)

test_pr_df = build_pr_df(y_test, test_score_pick)

c1, c2 = st.columns(2)
with c1:
    st.altair_chart(
        pr_chart(val_pr_df, thr_used, "PR Curve (VALIDATION) — dipakai untuk tuning threshold"),
        use_container_width=True,
    )
with c2:
    st.altair_chart(
        pr_chart(test_pr_df, thr_used, "PR Curve (TEST) — evaluasi final"),
        use_container_width=True,
    )

st.markdown(f"**Threshold terpilih (dari VALIDATION):** `{thr_used:.4f}`")

# ======================================================
# 🧪 Confusion Matrix & Report on TEST
# ======================================================
st.subheader("🧪 Confusion Matrix & Classification Report (TEST, threshold dari VAL)")
cm_df_pick = details[pick]["cm"]
rep_pick = details[pick]["report"]

cc1, cc2 = st.columns([1.1, 1.9])
with cc1:
    st.dataframe(cm_df_pick, use_container_width=True)
with cc2:
    st.code(rep_pick)

# ======================================================
# 🔍 Score distribution (TEST) — FIX: no facet + safe layering
# ======================================================
st.subheader("🔍 Distribusi Skor Prediksi (TEST)")

dist_df = pd.DataFrame({"score": test_score_pick, "label": y_test.astype(int)})
dist_df["label"] = dist_df["label"].map({0: "0 (tidak berulang)", 1: "1 (berulang)"})

hist = (
    alt.Chart(dist_df)
    .mark_bar(opacity=0.6)
    .encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=40), title="Skor (kelas 1)"),
        y=alt.Y("count():Q", title="Jumlah"),
        color=alt.Color("label:N", legend=alt.Legend(title="Label")),
        tooltip=[
            alt.Tooltip("label:N", title="Label"),
            alt.Tooltip("count():Q", title="Jumlah"),
        ],
    )
    .properties(height=320)
)

rule = alt.Chart(pd.DataFrame({"thr": [thr_used]})).mark_rule(strokeWidth=3).encode(
    x=alt.X("thr:Q", title="")
)

st.altair_chart(alt.layer(hist, rule), use_container_width=True)

# ======================================================
# 🧾 Narasi siap copas (Bab IV)
# ======================================================
with st.expander("🧾 Narasi Bab IV (siap copas)", expanded=False):
    best_row = res_df.iloc[0].to_dict()
    st.markdown(
        f"""
Pemodelan prediksi insiden berulang dilakukan menggunakan beberapa algoritma klasifikasi, yaitu *Logistic Regression* sebagai baseline, *Random Forest*, *Support Vector Machine*, serta *XGBoost* (jika tersedia). Mengingat distribusi label pada data bersifat tidak seimbang, evaluasi kinerja model difokuskan pada metrik *Precision–Recall Area Under Curve (PR-AUC)* karena lebih representatif untuk menilai kemampuan model dalam mendeteksi kelas minoritas (label insiden berulang).

Untuk menghindari bias optimistik, penentuan ambang keputusan (*threshold*) tidak menggunakan nilai default 0,5, melainkan ditetapkan melalui proses *threshold tuning* pada data **validation** berdasarkan strategi **{thr_mode}**. Threshold terpilih kemudian diterapkan pada data **test** untuk evaluasi final.

Berdasarkan hasil pengujian, model dengan PR-AUC tertinggi pada data test adalah **{best_row['model']}** dengan nilai **PR-AUC(test) = {best_row['pr_auc_test']:.4f}** dan threshold hasil tuning sebesar **{best_row['threshold_from_val']:.4f}**. Confusion matrix dan classification report pada data test digunakan untuk menilai trade-off antara *precision* dan *recall*, dengan perhatian khusus pada *false negative* karena berpotensi menyebabkan insiden berulang tidak terdeteksi dan problem laten tidak tertangani.
        """.strip()
    )

st.caption(
    "Catatan: Threshold ditentukan di VALIDATION (lebih ketat untuk tesis), lalu dievaluasi di TEST. "
    "Jika Anda ingin, tahap berikutnya adalah menyimpan output prediksi ke tabel `incident_prediction_results` "
    "untuk kebutuhan Problem Management."
)
