# pages/prediction/prediction_model_evaluation.py
# ============================================================
# 🔮 Training, Evaluasi, SAVE MODEL & SAVE METRICS
# - Random Forest, XGBoost (opsional), SVM
# - Sumber data: lasis_djp.dataset_supervised
# - Output DB  : lasis_djp.prediction_evaluation_results
# - Output file: models/<modeling_id>_<model_name>.joblib
# ============================================================

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
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
if not st.session_state.get("logged_in", True):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ============================================================
# 🔧 KONFIGURASI
# ============================================================
SCHEMA = "lasis_djp"
T_DATA = f"{SCHEMA}.dataset_supervised"
T_EVAL = f"{SCHEMA}.prediction_evaluation_results"

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

CAT_COLS = ["site", "modul", "sub_modul", "assignee"]
NUM_COLS = ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode", "window_days"]


# ============================================================
# 🔌 DB CONNECTION (SAMAKAN dgn halaman lain)
# ============================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


def read_df(engine, sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def exec_sql(engine, sql: str, params: dict | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


engine = get_engine()


# ============================================================
# ✅ Ensure source + output tables
# ============================================================
def assert_source_table_exists(engine) -> None:
    q = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = :schema AND table_name = 'dataset_supervised'
    LIMIT 1
    """
    df = read_df(engine, q, {"schema": SCHEMA})
    if df.empty:
        st.error(f"Tabel sumber tidak ditemukan: `{T_DATA}`. Periksa koneksi DB (st.secrets).")
        st.stop()


def ensure_eval_table(engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {T_EVAL}
    (
        eval_id uuid DEFAULT gen_random_uuid(),
        run_time timestamptz NOT NULL DEFAULT now(),

        jenis_pendekatan text NOT NULL,
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        time_col text NOT NULL DEFAULT '',
        include_noise boolean,
        eligible_rule text,
        split_name text,

        model_name text NOT NULL,
        test_size double precision NOT NULL,
        random_state integer NOT NULL,

        precision_pos double precision NOT NULL,
        recall_pos double precision NOT NULL,
        f1_pos double precision NOT NULL,

        tp bigint NOT NULL,
        fn bigint NOT NULL,
        fp bigint NOT NULL,
        tn bigint NOT NULL,

        CONSTRAINT prediction_evaluation_results_pkey
            PRIMARY KEY (
                jenis_pendekatan,
                modeling_id,
                window_days,
                time_col,
                split_name,
                model_name,
                test_size,
                random_state
            )
    );
    """
    exec_sql(engine, ddl)
    exec_sql(engine, f"CREATE INDEX IF NOT EXISTS idx_pred_eval_run ON {T_EVAL} (run_time DESC);")


assert_source_table_exists(engine)
ensure_eval_table(engine)


# ============================================================
# 📥 Load dataset (run selection)
# ============================================================
@st.cache_data(ttl=120)
def fetch_runs(_engine) -> pd.DataFrame:
    q = f"""
    SELECT DISTINCT
        jenis_pendekatan,
        modeling_id,
        window_days,
        time_col,
        COALESCE(include_noise,false) AS include_noise,
        COALESCE(eligible_rule,'') AS eligible_rule,
        COALESCE(split_name,'') AS split_name
    FROM {T_DATA}
    ORDER BY window_days ASC
    """
    return pd.read_sql(text(q), _engine)


@st.cache_data(ttl=120)
def load_dataset(
    _engine,
    jenis_pendekatan: str,
    modeling_id: str,
    window_days: int,
    time_col: str,
    include_noise: bool,
    eligible_rule: str,
    split_name: str,
    limit: int,
) -> pd.DataFrame:
    sql = f"""
    SELECT
        site, modul, sub_modul, assignee,
        gap_days, n_member_cluster, n_episode_cluster, n_member_episode,
        window_days,
        label_berulang
    FROM {T_DATA}
    WHERE jenis_pendekatan = :jenis
      AND modeling_id = CAST(:modeling_id AS uuid)
      AND window_days = :window_days
      AND time_col = :time_col
      AND COALESCE(include_noise,false) = :include_noise
      AND COALESCE(eligible_rule,'') = :eligible_rule
      AND COALESCE(split_name,'') = :split_name
    LIMIT :lim
    """
    df = pd.read_sql(
        text(sql),
        _engine,
        params=dict(
            jenis=jenis_pendekatan,
            modeling_id=modeling_id,
            window_days=int(window_days),
            time_col=str(time_col),
            include_noise=bool(include_noise),
            eligible_rule=str(eligible_rule),
            split_name=str(split_name),
            lim=int(limit),
        ),
    )

    # normalisasi type
    df["label_berulang"] = pd.to_numeric(df["label_berulang"], errors="coerce").fillna(0).astype(int)
    for c in CAT_COLS:
        df[c] = df[c].fillna("UNKNOWN").astype(str)
    for c in NUM_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(float)

    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
            ("num", StandardScaler(with_mean=False), NUM_COLS),
        ],
        remainder="drop",
    )


# ============================================================
# 💾 Save metrics (UPSERT) with NATIVE CAST (NO numpy types!)
# ============================================================
def upsert_metrics(
    engine,
    meta: dict,
    model_name: str,
    p: float,
    r: float,
    f1: float,
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> None:
    sql = f"""
    INSERT INTO {T_EVAL} (
        jenis_pendekatan, modeling_id, window_days, time_col,
        include_noise, eligible_rule, split_name,
        model_name, test_size, random_state,
        precision_pos, recall_pos, f1_pos,
        tp, fn, fp, tn
    )
    VALUES (
        :jenis, CAST(:modeling_id AS uuid), :window, :time_col,
        :include_noise, :eligible_rule, :split_name,
        :model, :test_size, :rs,
        :p, :r, :f1,
        :tp, :fn, :fp, :tn
    )
    ON CONFLICT (
        jenis_pendekatan, modeling_id, window_days,
        time_col, split_name, model_name, test_size, random_state
    )
    DO UPDATE SET
        run_time = now(),
        precision_pos = EXCLUDED.precision_pos,
        recall_pos = EXCLUDED.recall_pos,
        f1_pos = EXCLUDED.f1_pos,
        tp = EXCLUDED.tp,
        fn = EXCLUDED.fn,
        fp = EXCLUDED.fp,
        tn = EXCLUDED.tn;
    """

    # ✅ CAST KRUSIAL: ubah semua jadi tipe Python native
    params = {
        "jenis": str(meta["jenis_pendekatan"]),
        "modeling_id": str(meta["modeling_id"]),
        "window": int(meta["window_days"]),
        "time_col": str(meta["time_col"]),
        "include_noise": bool(meta["include_noise"]),
        "eligible_rule": str(meta["eligible_rule"]),
        "split_name": str(meta["split_name"]),
        "model": str(model_name),
        "test_size": float(meta["test_size"]),
        "rs": int(meta["random_state"]),
        "p": float(p),
        "r": float(r),
        "f1": float(f1),
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
    }

    with engine.begin() as conn:
        conn.execute(text(sql), params)


# ============================================================
# 🧭 UI
# ============================================================
# st.set_page_config(page_title="Training & Evaluasi Model Prediksi", layout="wide")
st.title("🔮 Training & Evaluasi Model Prediksi Insiden Berulang")
st.caption(f"Sumber: `{T_DATA}` • Output metrik: `{T_EVAL}` • Output model: `{MODEL_DIR}/`")

runs = fetch_runs(engine)
if runs.empty:
    st.warning("Tidak ada data pada dataset_supervised.")
    st.stop()

idx = st.selectbox(
    "Pilih dataset supervised (run)",
    runs.index,
    format_func=lambda i: (
        f"{runs.loc[i,'jenis_pendekatan']} | modeling={runs.loc[i,'modeling_id']} | "
        f"w={runs.loc[i,'window_days']} | time_col={runs.loc[i,'time_col']} | "
        f"noise={runs.loc[i,'include_noise']} | split={runs.loc[i,'split_name']}"
    ),
)

sel = runs.loc[idx]

limit = st.number_input("Limit data", min_value=1000, max_value=500000, value=100000, step=10000)
test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.number_input("Random state", min_value=0, max_value=9999, value=42, step=1)

c1, c2, c3 = st.columns(3)
with c1:
    use_rf = st.checkbox("Random Forest", value=True)
with c2:
    use_xgb = st.checkbox("XGBoost", value=True, disabled=not XGB_OK)
    if not XGB_OK:
        st.caption("xgboost tidak terpasang → XGBoost dinonaktifkan.")
with c3:
    use_svm = st.checkbox("SVM (LinearSVC)", value=True)

run_btn = st.button("🚀 Train + Evaluasi + Simpan Model & Metrik", type="primary")


# ============================================================
# ▶️ RUN
# ============================================================
if run_btn:
    if not (use_rf or use_svm or use_xgb):
        st.error("Pilih minimal satu model.")
        st.stop()

    df = load_dataset(
        engine,
        jenis_pendekatan=str(sel["jenis_pendekatan"]),
        modeling_id=str(sel["modeling_id"]),
        window_days=int(sel["window_days"]),
        time_col=str(sel["time_col"]),
        include_noise=bool(sel["include_noise"]),
        eligible_rule=str(sel["eligible_rule"]),
        split_name=str(sel["split_name"]),
        limit=int(limit),
    )

    pos = int((df["label_berulang"] == 1).sum())
    neg = int((df["label_berulang"] == 0).sum())
    st.write(f"Jumlah data: **{len(df):,}** | Positif(1): **{pos:,}** | Negatif(0): **{neg:,}**")
    if len(df) < 200 or pos < 5:
        st.warning("Dataset terlalu kecil / positif terlalu sedikit untuk evaluasi.")
        st.stop()

    X = df[CAT_COLS + NUM_COLS]
    y = df["label_berulang"].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(test_size), random_state=int(random_state), stratify=y
    )

    pre = build_preprocessor()
    results = []
    cms = {}

    meta = {
        "jenis_pendekatan": str(sel["jenis_pendekatan"]),
        "modeling_id": str(sel["modeling_id"]),
        "window_days": int(sel["window_days"]),
        "time_col": str(sel["time_col"]),
        "include_noise": bool(sel["include_noise"]),
        "eligible_rule": str(sel["eligible_rule"]),
        "split_name": str(sel["split_name"]),
        "test_size": float(test_size),
        "random_state": int(random_state),
    }

    def eval_save(model_pipeline: Pipeline, name: str):
        model_pipeline.fit(X_train, y_train)
        y_pred = model_pipeline.predict(X_test)

        p = float(precision_score(y_test, y_pred, pos_label=1, zero_division=0))
        r = float(recall_score(y_test, y_pred, pos_label=1, zero_division=0))
        f1v = float(f1_score(y_test, y_pred, pos_label=1, zero_division=0))

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]

        # 1) SAVE MODEL (file)
        model_path = MODEL_DIR / f"{meta['modeling_id']}_{name}.joblib"
        joblib.dump(model_pipeline, model_path)

        # 2) UPSERT METRICS (DB)  ✅ already cast to native types
        upsert_metrics(engine, meta, name, p, r, f1v, tn, fp, fn, tp)

        results.append(
            {
                "Model": name,
                "Precision (Berulang)": round(p, 4),
                "Recall (Berulang)": round(r, 4),
                "F1-Score (Berulang)": round(f1v, 4),
                "TP": tp,
                "FN": fn,
                "FP": fp,
                "TN": tn,
                "Model_File": str(model_path),
            }
        )
        cms[name] = cm

    with st.spinner("Training & evaluasi..."):
        if use_rf:
            rf = Pipeline(
                [
                    ("prep", pre),
                    ("clf", RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced_subsample",
                        random_state=int(random_state),
                        n_jobs=-1,
                    )),
                ]
            )
            eval_save(rf, "Random Forest")

        if use_xgb and XGB_OK:
            pos_tr = int((y_train == 1).sum())
            neg_tr = int((y_train == 0).sum())
            scale = float(neg_tr / pos_tr) if pos_tr > 0 else 1.0

            xgb = Pipeline(
                [
                    ("prep", pre),
                    ("clf", XGBClassifier(
                        n_estimators=300,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        scale_pos_weight=scale,
                        random_state=int(random_state),
                        n_jobs=-1,
                    )),
                ]
            )
            eval_save(xgb, "XGBoost")

        if use_svm:
            svm = Pipeline(
                [
                    ("prep", pre),
                    ("clf", LinearSVC(class_weight="balanced")),
                ]
            )
            eval_save(svm, "SVM")

    st.success("✅ Selesai. Model dan metrik telah disimpan.")
    st.subheader("Ringkasan Hasil Evaluasi")
    st.dataframe(pd.DataFrame(results), use_container_width=True)

    st.subheader("Confusion Matrix per Model")
    cols = st.columns(len(cms))
    for i, (name, cm) in enumerate(cms.items()):
        with cols[i]:
            st.write(f"**{name}**")
            st.dataframe(
                pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
                use_container_width=True,
            )

    st.subheader("Verifikasi Data Tersimpan di DB")
    df_saved = read_df(
        engine,
        f"""
        SELECT run_time, model_name, precision_pos, recall_pos, f1_pos, tp, fn, fp, tn
        FROM {T_EVAL}
        WHERE jenis_pendekatan = :jenis
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
          AND COALESCE(split_name,'') = :split_name
          AND test_size = :test_size
          AND random_state = :random_state
        ORDER BY run_time DESC, model_name ASC
        """,
        {
            "jenis": meta["jenis_pendekatan"],
            "modeling_id": meta["modeling_id"],
            "window_days": meta["window_days"],
            "time_col": meta["time_col"],
            "split_name": meta["split_name"],
            "test_size": meta["test_size"],
            "random_state": meta["random_state"],
        },
    )
    st.dataframe(df_saved, use_container_width=True)
