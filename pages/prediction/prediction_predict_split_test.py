# pages/prediction/prediction_predict_rawtext_split_test.py
# ============================================================
# 🔮 Prediksi Split TEST (tanpa leakage) — RAW FEATURES + TEXT
# - TRAIN: dataset_supervised WHERE split_name='train' (ambil label) JOIN incident_kelayakan (raw fields)
# - TEST : dataset_supervised WHERE split_name='test'  (tanpa label) JOIN incident_kelayakan (raw fields)
# - RAW fields dipakai:
#   incident_number, tgl_submit, site, assignee, modul, sub_modul, detailed_decription
# - Model: Random Forest, XGBoost (opsional), SVM (LinearSVC)
# - Feature:
#   * Text TF-IDF: detailed_decription
#   * Cat OHE: site/assignee/modul/sub_modul
#   * Date features aman: day-of-week, month, day (dari tgl_submit)
# - Output:
#   * Evaluasi internal (holdout dari TRAIN) + Confusion Matrix per model
#   * Prediksi untuk TEST split (tanpa label) + download CSV
# ============================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import LinearSVC

# XGBoost opsional
try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False


# ============================================================
# 🔐 Guard Login
# ============================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ============================================================
# 🔧 PARAMETER DATABASE
# ============================================================
SCHEMA = "lasis_djp"
T_DS = f"{SCHEMA}.dataset_supervised"
T_K = f"{SCHEMA}.incident_kelayakan"

# kolom mentah yang dipakai (sesuai permintaan)
ID_COL = "incident_number"
DATE_COL = "tgl_submit"
TEXT_COL = "detailed_decription"  # mengikuti penamaan user
CAT_COLS = ["site", "assignee", "modul", "sub_modul"]
TARGET = "label_berulang"


# ============================================================
# 🔌 DB Connection
# ============================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


def read_df(engine, sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


engine = get_engine()


# ============================================================
# 🧱 Load runs / configs
# ============================================================
@st.cache_data(ttl=120)
def fetch_runs(_engine) -> pd.DataFrame:
    q = f"""
    SELECT
      jenis_pendekatan,
      modeling_id,
      window_days,
      time_col,
      COALESCE(include_noise,false) AS include_noise,
      COALESCE(eligible_rule,'') AS eligible_rule,
      COALESCE(split_name,'') AS split_name,
      MAX(run_time) AS last_run_time,
      COUNT(*) AS n_rows
    FROM {T_DS}
    GROUP BY 1,2,3,4,5,6,7
    ORDER BY last_run_time DESC
    """
    return pd.read_sql(text(q), _engine)


def _latest_incident_subquery(split_name: str) -> str:
    # Ambil 1 baris terbaru per incident_number untuk konfigurasi yang dipilih
    return f"""
    WITH latest AS (
      SELECT DISTINCT ON (incident_number)
        incident_number,
        run_time,
        label_berulang
      FROM {T_DS}
      WHERE COALESCE(split_name,'') = :split_name
        AND jenis_pendekatan = :jenis_pendekatan
        AND modeling_id = CAST(:modeling_id AS uuid)
        AND window_days = :window_days
        AND time_col = :time_col
        AND COALESCE(include_noise,false) = :include_noise
        AND COALESCE(eligible_rule,'') = :eligible_rule
      ORDER BY incident_number, run_time DESC
    )
    """


@st.cache_data(ttl=120)
def load_train_raw(_engine, meta: dict, limit: int) -> pd.DataFrame:
    q = (
        _latest_incident_subquery("train")
        + f"""
    SELECT
      k.{ID_COL} AS incident_number,
      k.{DATE_COL} AS tgl_submit,
      k.site,
      k.assignee,
      k.modul,
      k.sub_modul,
      k.{TEXT_COL} AS detailed_decription,
      l.label_berulang
    FROM latest l
    JOIN {T_K} k
      ON k.{ID_COL} = l.incident_number
    ORDER BY COALESCE(k.{DATE_COL}, l.run_time) NULLS LAST
    LIMIT :limit
    """
    )
    params = dict(
        split_name="train",
        jenis_pendekatan=str(meta["jenis_pendekatan"]),
        modeling_id=str(meta["modeling_id"]),
        window_days=int(meta["window_days"]),
        time_col=str(meta["time_col"]),
        include_noise=bool(meta["include_noise"]),
        eligible_rule=str(meta["eligible_rule"]),
        limit=int(limit),
    )
    df = pd.read_sql(text(q), _engine, params=params)
    return df


@st.cache_data(ttl=120)
def load_test_raw(_engine, meta: dict, limit: int) -> pd.DataFrame:
    q = (
        _latest_incident_subquery("test")
        + f"""
    SELECT
      k.{ID_COL} AS incident_number,
      k.{DATE_COL} AS tgl_submit,
      k.site,
      k.assignee,
      k.modul,
      k.sub_modul,
      k.{TEXT_COL} AS detailed_decription
    FROM latest l
    JOIN {T_K} k
      ON k.{ID_COL} = l.incident_number
    ORDER BY COALESCE(k.{DATE_COL}, l.run_time) NULLS LAST
    LIMIT :limit
    """
    )
    params = dict(
        split_name="test",
        jenis_pendekatan=str(meta["jenis_pendekatan"]),
        modeling_id=str(meta["modeling_id"]),
        window_days=int(meta["window_days"]),
        time_col=str(meta["time_col"]),
        include_noise=bool(meta["include_noise"]),
        eligible_rule=str(meta["eligible_rule"]),
        limit=int(limit),
    )
    df = pd.read_sql(text(q), _engine, params=params)
    return df


# ============================================================
# 🧠 Feature engineering (aman, tanpa histori)
# ============================================================
def prepare_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # basic cleaning
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    for c in CAT_COLS:
        out[c] = out[c].fillna("UNKNOWN").astype(str).str.strip()

    # text
    out[TEXT_COL] = out[TEXT_COL].fillna("").astype(str)

    # date features (tanpa histori)
    out["submit_dow"] = out[DATE_COL].dt.dayofweek.fillna(-1).astype(int)  # 0=Mon ... 6=Sun
    out["submit_month"] = out[DATE_COL].dt.month.fillna(0).astype(int)
    out["submit_day"] = out[DATE_COL].dt.day.fillna(0).astype(int)

    return out


def build_preprocessor(
    cat_cols: List[str],
    text_col: str,
    date_cols: List[str],
    max_features: int,
    ngram_max: int,
) -> ColumnTransformer:
    tfidf = TfidfVectorizer(
        max_features=int(max_features),
        ngram_range=(1, int(ngram_max)),
        lowercase=True,
    )

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("txt", tfidf, text_col),
            ("dt", "passthrough", date_cols),
        ],
        remainder="drop",
    )


@dataclass
class EvalRow:
    model: str
    precision_pos: float
    recall_pos: float
    f1_pos: float
    tp: int
    fn: int
    fp: int
    tn: int


def evaluate_binary(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> Tuple[EvalRow, np.ndarray]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    p = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    r = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1v = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    return (
        EvalRow(
            model=model_name,
            precision_pos=float(p),
            recall_pos=float(r),
            f1_pos=float(f1v),
            tp=int(tp),
            fn=int(fn),
            fp=int(fp),
            tn=int(tn),
        ),
        cm,
    )


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# 🧭 UI
# ============================================================
st.title("🔮 Prediksi Split TEST (No Leakage) — RAW + Text TF-IDF")
st.caption(
    "TRAIN diambil dari `dataset_supervised` split='train' lalu JOIN `incident_kelayakan` (fitur mentah + teks). "
    "TEST diambil dari split='test' dan diprediksi tanpa label."
)

runs = fetch_runs(engine)
# hanya tampilkan konfigurasi yang punya train dan test
runs2 = runs[runs["split_name"].isin(["train", "test"])].copy()
if runs2.empty:
    st.warning("Tidak ada data split train/test pada dataset_supervised.")
    st.stop()

# --- pilih konfigurasi berdasar split train/test (ambil yang punya test sebagai anchor)
runs_test = runs2[runs2["split_name"] == "test"].copy()
if runs_test.empty:
    st.warning("Tidak ada split_name='test' pada dataset_supervised.")
    st.stop()

with st.expander("Ringkasan run (split=test)", expanded=False):
    st.dataframe(runs_test, use_container_width=True, hide_index=True)

st.subheader("1) Pilih Konfigurasi (berdasarkan split='test')")
c1, c2, c3 = st.columns(3)
with c1:
    jenis = st.selectbox("jenis_pendekatan", sorted(runs_test["jenis_pendekatan"].unique().tolist()))
sub = runs_test[runs_test["jenis_pendekatan"] == jenis].copy()

with c2:
    modeling_id = st.selectbox("modeling_id", sub["modeling_id"].astype(str).unique().tolist())
sub = sub[sub["modeling_id"].astype(str) == str(modeling_id)].copy()

with c3:
    window_days = st.selectbox("window_days", sorted(sub["window_days"].unique().tolist()))
sub = sub[sub["window_days"] == int(window_days)].copy()

c4, c5, c6 = st.columns([1.2, 1.0, 1.2])
with c4:
    time_col = st.selectbox("time_col", sorted(sub["time_col"].astype(str).unique().tolist()))
sub = sub[sub["time_col"].astype(str) == str(time_col)].copy()

with c5:
    include_noise = st.selectbox("include_noise", [False, True], index=0)
sub = sub[sub["include_noise"].astype(bool) == bool(include_noise)].copy()

with c6:
    eligible_rule = st.selectbox("eligible_rule", sorted(sub["eligible_rule"].astype(str).unique().tolist()))
sub = sub[sub["eligible_rule"].astype(str) == str(eligible_rule)].copy()

meta = dict(
    jenis_pendekatan=str(jenis),
    modeling_id=str(modeling_id),
    window_days=int(window_days),
    time_col=str(time_col),
    include_noise=bool(include_noise),
    eligible_rule=str(eligible_rule),
)

st.divider()

st.subheader("2) Konfigurasi Model & Fitur")
k1, k2, k3, k4 = st.columns(4)
with k1:
    limit_train = st.number_input("Limit TRAIN (split='train')", min_value=1000, max_value=500000, value=200000, step=10000)
with k2:
    limit_test = st.number_input("Limit TEST (split='test')", min_value=100, max_value=200000, value=50000, step=1000)
with k3:
    test_size = st.slider("Holdout dari TRAIN (untuk evaluasi internal)", 0.1, 0.5, 0.2, 0.05)
with k4:
    random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)

t1, t2, t3 = st.columns(3)
with t1:
    max_features = st.number_input("TF-IDF max_features", min_value=2000, max_value=200000, value=50000, step=5000)
with t2:
    ngram_max = st.selectbox("TF-IDF ngram max", [1, 2, 3], index=1)
with t3:
    min_pos = st.number_input("Min label=1 di TRAIN", min_value=1, max_value=5000, value=20, step=1)

m1, m2, m3 = st.columns(3)
with m1:
    use_rf = st.checkbox("Random Forest", value=True)
with m2:
    use_xgb = st.checkbox("XGBoost", value=True, disabled=not XGB_OK)
    if not XGB_OK:
        st.caption("xgboost tidak terpasang → XGBoost dinonaktifkan.")
with m3:
    use_svm = st.checkbox("SVM (LinearSVC)", value=True)

run_btn = st.button("🚀 Train (split=train) + Evaluate (holdout) + Predict (split=test)", type="primary")

if run_btn:
    if not (use_rf or use_xgb or use_svm):
        st.error("Pilih minimal 1 model.")
        st.stop()

    with st.spinner("Memuat TRAIN (split=train) & TEST (split=test) dari incident_kelayakan..."):
        df_train_raw = load_train_raw(engine, meta, limit=int(limit_train))
        df_test_raw = load_test_raw(engine, meta, limit=int(limit_test))

    if df_train_raw.empty:
        st.error("TRAIN kosong. Pastikan split_name='train' tersedia untuk konfigurasi ini.")
        st.stop()

    if df_test_raw.empty:
        st.error("TEST kosong. Pastikan split_name='test' tersedia untuk konfigurasi ini.")
        st.stop()

    # prepare features
    df_train = prepare_raw_features(df_train_raw)
    df_test = prepare_raw_features(df_test_raw)

    # label
    df_train[TARGET] = pd.to_numeric(df_train[TARGET], errors="coerce").fillna(0).astype(int)
    pos = int((df_train[TARGET] == 1).sum())
    st.info(f"TRAIN: {len(df_train):,} baris | pos(label=1): {pos:,} • TEST: {len(df_test):,} baris")

    if pos < int(min_pos):
        st.warning(f"Label=1 di TRAIN hanya {pos:,} (< {int(min_pos)}). Model bisa cenderung memprediksi 0.")

    if df_train[TARGET].nunique() < 2:
        st.error("TRAIN tidak memiliki variasi label (hanya 0 atau hanya 1).")
        st.stop()

    # build X,y
    date_cols = ["submit_dow", "submit_month", "submit_day"]
    feature_cols = CAT_COLS + [TEXT_COL] + date_cols

    X_all = df_train[feature_cols].copy()
    y_all = df_train[TARGET].astype(int).values

    X_tr, X_va, y_tr, y_va = train_test_split(
        X_all,
        y_all,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y_all,
    )

    pre = build_preprocessor(
        cat_cols=CAT_COLS,
        text_col=TEXT_COL,
        date_cols=date_cols,
        max_features=int(max_features),
        ngram_max=int(ngram_max),
    )

    # train + eval internal
    eval_rows: List[EvalRow] = []
    cms: Dict[str, np.ndarray] = {}

    models: List[Tuple[str, Pipeline]] = []

    if use_rf:
        rf = Pipeline(
            [
                ("pre", pre),
                ("clf", RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced_subsample",
                    random_state=int(random_state),
                    n_jobs=-1,
                )),
            ]
        )
        models.append(("Random Forest", rf))

    if use_xgb and XGB_OK:
        pos_tr = int((y_tr == 1).sum())
        neg_tr = int((y_tr == 0).sum())
        spw = float(neg_tr / pos_tr) if pos_tr > 0 else 1.0

        xgb = Pipeline(
            [
                ("pre", pre),
                ("clf", XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    scale_pos_weight=spw,
                    random_state=int(random_state),
                    n_jobs=-1,
                )),
            ]
        )
        models.append(("XGBoost", xgb))

    if use_svm:
        svm = Pipeline([("pre", pre), ("clf", LinearSVC(class_weight="balanced"))])
        models.append(("SVM", svm))

    with st.spinner("Training & evaluasi internal (holdout dari TRAIN)..."):
        for name, mdl in models:
            mdl.fit(X_tr, y_tr)
            yhat = mdl.predict(X_va).astype(int)
            row, cm = evaluate_binary(y_va, yhat, name)
            eval_rows.append(row)
            cms[name] = cm

    st.subheader("3) Hasil Evaluasi Internal (Holdout dari TRAIN)")
    out_eval = pd.DataFrame(
        [
            {
                "Model Prediktif": r.model,
                "Precision (Berulang)": round(r.precision_pos, 4),
                "Recall (Berulang)": round(r.recall_pos, 4),
                "F1-Score (Berulang)": round(r.f1_pos, 4),
                "TP": r.tp,
                "FN": r.fn,
                "FP": r.fp,
                "TN": r.tn,
            }
            for r in eval_rows
        ]
    ).sort_values(["F1-Score (Berulang)", "Recall (Berulang)"], ascending=False)

    st.dataframe(out_eval, use_container_width=True, hide_index=True)

    st.markdown("### Confusion Matrix per Model")
    cols = st.columns(len(cms))
    for i, (name, cm) in enumerate(cms.items()):
        with cols[i]:
            st.write(f"**{name}**")
            st.dataframe(
                pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
                use_container_width=True,
            )

    # Fit ulang pada seluruh TRAIN lalu prediksi TEST
    X_train_full = df_train[feature_cols].copy()
    y_train_full = y_all
    X_test = df_test[feature_cols].copy()

    st.divider()
    st.subheader("4) Prediksi untuk TEST Split (tanpa label)")

    with st.spinner("Fit ulang pada seluruh TRAIN lalu prediksi TEST..."):
        out = df_test_raw.copy()  # output dasar dari incident_kelayakan (mentah)
        out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

        # siapkan juga fitur untuk test (yang sudah diprepare)
        X_test_feat = df_test[feature_cols].copy()

        for name, mdl in models:
            mdl.fit(X_train_full, y_train_full)

            pred = mdl.predict(X_test_feat).astype(int)
            out[f"pred_{name}"] = pred

            # score/prob
            if hasattr(mdl, "predict_proba"):
                try:
                    prob = mdl.predict_proba(X_test_feat)[:, 1].astype(float)
                    out[f"prob_{name}"] = prob
                except Exception:
                    out[f"prob_{name}"] = np.nan
            else:
                # LinearSVC: decision_function → sigmoid agar lebih “mirip probabilitas” (tetap pseudo)
                try:
                    s = mdl.decision_function(X_test_feat).astype(float)
                    out[f"score_{name}"] = s
                    out[f"prob_{name}"] = sigmoid(s)
                except Exception:
                    out[f"score_{name}"] = np.nan
                    out[f"prob_{name}"] = np.nan

    # KPI per model
    st.markdown("### Ringkasan Prediksi (TEST)")
    kcols = st.columns(len(models))
    for i, (name, _) in enumerate(models):
        with kcols[i]:
            n1 = int((out[f"pred_{name}"] == 1).sum())
            n0 = int((out[f"pred_{name}"] == 0).sum())
            st.metric(f"{name} – Pred=1", f"{n1:,}")
            st.caption(f"Pred=0: {n0:,}")

    show_cols = [
        "incident_number",
        "tgl_submit",
        "site",
        "assignee",
        "modul",
        "sub_modul",
        TEXT_COL,
    ]
    # tambahkan kolom pred/prob/score
    extra = [c for c in out.columns if c.startswith("pred_") or c.startswith("prob_") or c.startswith("score_")]
    show_cols = [c for c in show_cols if c in out.columns] + extra

    st.dataframe(out[show_cols], use_container_width=True, height=520)

    st.subheader("5) Unduh Hasil")
    csv = out[show_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download prediksi TEST split (CSV)",
        data=csv,
        file_name="prediksi_split_test_raw_text_no_leakage.csv",
        mime="text/csv",
    )

    with st.expander("📘 Catatan Metodologis", expanded=False):
        st.markdown(
            """
- Halaman ini **menghindari leakage** dengan cara:
  - TRAIN hanya dari `split_name='train'` (label dipakai).
  - TEST dari `split_name='test'` dan **tidak menggunakan label**.
  - Fitur yang dipakai hanya **kolom mentah** dari `incident_kelayakan` + TF-IDF teks, serta fitur tanggal sederhana dari `tgl_submit`.
- “Evaluasi internal” dilakukan lewat holdout dari TRAIN (stratified) untuk melihat gambaran performa, bukan evaluasi pada TEST.
- Untuk SVM (LinearSVC), kolom `prob_SVM` adalah **pseudo-probability** (sigmoid dari decision score).
"""
        )
