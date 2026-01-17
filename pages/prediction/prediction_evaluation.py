# pages/prediction/prediction_evaluation_from_dataset_supervised_test.py
# ============================================================
# 📊 Evaluasi Model Prediktif (Train/Test dari dataset_supervised)
# - Train: split_name='train'
# - Test : split_name='test'
# - Robust time: time_eval = COALESCE(tgl_submit, event_time) (tgl_submit bisa NULL)
# - Evaluasi: Confusion Matrix + Precision/Recall/F1 (kelas berulang=1)
# - Model: Random Forest, XGBoost (opsional), SVM
# - Anti-leakage: leakage guards memastikan label test tidak masuk fitur
# ============================================================

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# XGBoost opsional
try:
    from xgboost import XGBClassifier  # type: ignore

    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ============================================================
# 🔐 Guard Login
# ============================================================
if not st.session_state.get("logged_in", True):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

# st.set_page_config(page_title="Evaluasi Model (Test Split)", layout="wide")
st.title("📊 Evaluasi Model Prediktif — dataset_supervised (split_name = test)")
st.caption(
    "Model dilatih pada split train dan dievaluasi pada split test. "
    "Waktu evaluasi menggunakan time_eval = COALESCE(tgl_submit, event_time). "
    "Halaman ini dilengkapi leakage guards untuk memastikan label tidak masuk fitur."
)

# ============================================================
# 🔌 DB Connection (konsisten)
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
# 🎯 Kolom & konfigurasi
# ============================================================
TIME_COL = "time_eval"
TARGET_COL = "label_berulang"

CAT_COLS = ["site", "assignee", "modul", "sub_modul"]
NUM_COLS = ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode", "window_days"]
FEATURE_COLS = CAT_COLS + NUM_COLS

REQ_COLS = ["incident_number", "tgl_submit", "event_time", TIME_COL] + FEATURE_COLS + [TARGET_COL, "split_name"]


# ============================================================
# 🧠 Preprocessor (remainder=drop agar kolom lain dibuang)
# ============================================================
def build_preprocessor() -> ColumnTransformer:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, CAT_COLS),
            ("num", num_pipe, NUM_COLS),
        ],
        remainder="drop",
    )


# ============================================================
# 🧪 Models
# ============================================================
def build_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "SVM (LinearSVC)": LinearSVC(
            random_state=random_state,
            class_weight="balanced",
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
    return models


# ============================================================
# 📥 Load dataset (robust time column, cache-safe)
# ============================================================
@st.cache_data(ttl=300)
def load_supervised(_engine, include_noise: str, window_days: Optional[int], limit: int = 0) -> pd.DataFrame:
    where = ["1=1"]
    params: Dict = {}

    if include_noise != "(tanpa filter)":
        where.append("include_noise = :include_noise")
        params["include_noise"] = (include_noise == "true")

    if window_days is not None:
        where.append("window_days = :window_days")
        params["window_days"] = int(window_days)

    q = f"""
    SELECT
      incident_number,
      tgl_submit,
      event_time,
      COALESCE(tgl_submit, event_time) AS time_eval,
      site,
      assignee,
      modul,
      sub_modul,
      gap_days,
      n_member_cluster,
      n_episode_cluster,
      n_member_episode,
      window_days,
      label_berulang,
      split_name,
      include_noise
    FROM lasis_djp.dataset_supervised
    WHERE {" AND ".join(where)}
    """
    if limit and limit > 0:
        q += " LIMIT :limit"
        params["limit"] = int(limit)

    df = pd.read_sql(text(q), _engine, params=params)

    # Konversi waktu & robust drop
    df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df["time_eval"] = pd.to_datetime(df["time_eval"], errors="coerce")

    # Diagnostik sebelum drop
    total_before = len(df)
    n_tgl_null = int(df["tgl_submit"].isna().sum())
    n_evt_null = int(df["event_time"].isna().sum())
    n_time_eval_null = int(df["time_eval"].isna().sum())

    # Drop hanya jika time_eval tetap NULL
    df = df.dropna(subset=["time_eval"]).copy()
    df = df.sort_values("time_eval").reset_index(drop=True)

    # Validasi kolom
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Kolom wajib tidak ditemukan pada dataset_supervised: {missing}")

    # Label numeric robust
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").astype("Int64")

    # Simpan meta untuk UI
    df.attrs["total_before"] = total_before
    df.attrs["n_tgl_null"] = n_tgl_null
    df.attrs["n_evt_null"] = n_evt_null
    df.attrs["n_time_eval_null_before_drop"] = n_time_eval_null
    df.attrs["total_after"] = len(df)

    return df


# ============================================================
# 📌 Metrics
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[1], average=None, zero_division=0
    )
    return {
        "TP": int(tp),
        "FN": int(fn),
        "FP": int(fp),
        "TN": int(tn),
        "Precision (Berulang=1)": float(p[0]),
        "Recall (Berulang=1)": float(r[0]),
        "F1-Score (Berulang=1)": float(f1[0]),
    }


def make_cm_df(tn: int, fp: int, fn: int, tp: int) -> pd.DataFrame:
    return pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["Aktual: Berulang (1)", "Aktual: Tidak Berulang (0)"],
        columns=["Pred: Berulang (1)", "Pred: Tidak Berulang (0)"],
    )


# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.header("Pengaturan Evaluasi")

    include_noise = st.selectbox("Filter include_noise (opsional)", ["(tanpa filter)", "true", "false"], index=0)
    wd_opt = st.text_input("Filter window_days (kosong=tanpa filter)", value="")
    window_days = int(wd_opt) if wd_opt.strip().isdigit() else None

    limit = st.number_input("Limit baris (0=all)", min_value=0, value=0, step=1000)
    random_state = st.number_input("Random seed", min_value=0, value=42, step=1)

    run_btn = st.button("▶️ Jalankan Evaluasi", use_container_width=True)

# ============================================================
# Load data
# ============================================================
try:
    df = load_supervised(engine, include_noise=include_noise, window_days=window_days, limit=int(limit))
except Exception as e:
    st.error(f"Gagal memuat dataset_supervised: {e}")
    st.stop()

if df.empty:
    st.error("Data kosong setelah drop time_eval NULL. Periksa ketersediaan event_time/tgl_submit pada tabel.")
    st.stop()

st.caption(
    f"Diagnostik waktu: total={df.attrs.get('total_before','-')}, "
    f"tgl_submit NULL={df.attrs.get('n_tgl_null','-')}, "
    f"event_time NULL={df.attrs.get('n_evt_null','-')}, "
    f"time_eval NULL (sebelum drop)={df.attrs.get('n_time_eval_null_before_drop','-')}, "
    f"tersisa={df.attrs.get('total_after','-')}."
)

# Split
df_train = df[df["split_name"] == "train"].copy()
df_test = df[df["split_name"] == "test"].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total data (valid time_eval)", f"{len(df):,}")
c2.metric("Train", f"{len(df_train):,}")
c3.metric("Test", f"{len(df_test):,}")
pos_rate_test = float((df_test[TARGET_COL] == 1).mean()) if len(df_test) else np.nan
c4.metric("Proporsi label=1 (Test)", f"{pos_rate_test:.3f}" if not np.isnan(pos_rate_test) else "-")

if len(df_train) == 0 or len(df_test) == 0:
    st.error("Split train/test tidak lengkap. Pastikan dataset_supervised memiliki split_name 'train' dan 'test'.")
    st.stop()

# Distribusi kelas (lebih informatif daripada unique)
st.subheader("Distribusi Kelas (label_berulang)")
colA, colB = st.columns(2)
with colA:
    st.markdown("**Train (split=train)**")
    vc_train = df_train[TARGET_COL].value_counts(dropna=False).sort_index()
    st.dataframe(vc_train.rename("count").to_frame(), use_container_width=True)
with colB:
    st.markdown("**Test (split=test)**")
    vc_test = df_test[TARGET_COL].value_counts(dropna=False).sort_index()
    st.dataframe(vc_test.rename("count").to_frame(), use_container_width=True)

# Rentang waktu berdasarkan time_eval
t1, t2 = st.columns(2)
t1.write(f"**Rentang waktu Train:** {str(df_train[TIME_COL].min())[:19]} → {str(df_train[TIME_COL].max())[:19]}")
t2.write(f"**Rentang waktu Test :** {str(df_test[TIME_COL].min())[:19]} → {str(df_test[TIME_COL].max())[:19]}")

# ============================================================
# Run evaluation
# ============================================================
if not run_btn:
    st.info("Klik **Jalankan Evaluasi** untuk melatih model pada train dan mengevaluasi pada test.")
    st.stop()

# ----- X/y split (EXPLICIT) -----
X_train = df_train[FEATURE_COLS].copy()
y_train = df_train[TARGET_COL].astype(int).to_numpy()

X_test = df_test[FEATURE_COLS].copy()
y_test = df_test[TARGET_COL].astype(int).to_numpy()

# ============================================================
# 🔒 LEAKAGE GUARDS (fail-fast)
# ============================================================
assert TARGET_COL not in X_train.columns, "LEAKAGE: label_berulang masuk ke X_train!"
assert TARGET_COL not in X_test.columns, "LEAKAGE: label_berulang masuk ke X_test!"
for forbidden in ["split_name", "incident_number", "tgl_submit", "event_time", "time_eval"]:
    assert forbidden not in X_train.columns, f"LEAKAGE: {forbidden} masuk ke X_train!"
    assert forbidden not in X_test.columns, f"LEAKAGE: {forbidden} masuk ke X_test!"
assert len(X_train) == len(y_train), "Mismatch: X_train != y_train"
assert len(X_test) == len(y_test), "Mismatch: X_test != y_test"
assert set(np.unique(y_train)).issubset({0, 1}), f"Label train tidak valid: {set(np.unique(y_train))}"
assert set(np.unique(y_test)).issubset({0, 1}), f"Label test tidak valid: {set(np.unique(y_test))}"

st.success("Leakage guard: OK — label_berulang tidak masuk fitur, dan split train/test valid.")

# Build pipeline
pre = build_preprocessor()
models = build_models(random_state=int(random_state))

st.divider()
st.subheader("Hasil Evaluasi per Model")

tabs = st.tabs(list(models.keys()))
rows = []

for (model_name, clf), tab in zip(models.items(), tabs):
    with tab:
        pipe = Pipeline(steps=[("pre", pre), ("clf", clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        m = compute_metrics(y_test, y_pred)
        rows.append({"Model": model_name, **m})

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        st.markdown("**Confusion Matrix**")
        st.dataframe(make_cm_df(int(tn), int(fp), int(fn), int(tp)), use_container_width=True)

        st.markdown("**Metrik Kelas Berulang (label=1)**")
        a, b, c = st.columns(3)
        a.metric("Precision (1)", f"{m['Precision (Berulang=1)']:.4f}")
        b.metric("Recall (1)", f"{m['Recall (Berulang=1)']:.4f}")
        c.metric("F1 (1)", f"{m['F1-Score (Berulang=1)']:.4f}")

        st.markdown("**Komponen Confusion Matrix**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("TP", f"{m['TP']}")
        k2.metric("FN", f"{m['FN']}")
        k3.metric("FP", f"{m['FP']}")
        k4.metric("TN", f"{m['TN']}")

st.divider()
st.subheader("Ringkasan (Tabel Final)")
df_res = pd.DataFrame(rows)
st.dataframe(df_res, use_container_width=True)

best = df_res.sort_values("F1-Score (Berulang=1)", ascending=False).iloc[0]
st.success(
    f"Model terbaik berdasarkan F1 kelas berulang (1): **{best['Model']}** "
    f"(F1={best['F1-Score (Berulang=1)']:.4f}, "
    f"Precision={best['Precision (Berulang=1)']:.4f}, "
    f"Recall={best['Recall (Berulang=1)']:.4f})."
)

with st.expander("Audit Fitur yang Masuk ke Model", expanded=False):
    st.write("Kolom fitur (FEATURE_COLS):", FEATURE_COLS)
    st.write("Kolom pada X_train:", list(X_train.columns))
    st.write("Kolom pada X_test :", list(X_test.columns))

with st.expander("Catatan Metodologis (untuk tesis/sidang)", expanded=False):
    st.write(
        "- Evaluasi dilakukan pada split test (unseen), sedangkan pelatihan hanya pada split train.\n"
        "- time_eval dibentuk dari COALESCE(tgl_submit, event_time) agar pemisahan berbasis waktu tetap konsisten.\n"
        "- Implementasi dilengkapi leakage guards untuk memastikan label (`label_berulang`) tidak pernah masuk sebagai fitur.\n"
        "- Metrik yang ditampilkan berfokus pada kelas berulang (label=1) sesuai tujuan problem management."
    )
