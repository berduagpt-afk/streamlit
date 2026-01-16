# pages/prediction/prediction_model_evaluation.py
# ============================================================
# 🔮 Prediksi Insiden Berulang — Training, Evaluasi, & AUTO-SAVE
# Sumber : lasis_djp.dataset_supervised
# Model  : Random Forest, SVM, XGBoost (opsional)
# Output : lasis_djp.prediction_evaluation_results
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

# XGBoost opsional
try:
    from xgboost import XGBClassifier  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False


# ============================================================
# 🔧 PARAMETER DATABASE
# ============================================================
SCHEMA = "lasis_djp"
T_LABEL = f"{SCHEMA}.dataset_supervised"
T_EVAL = f"{SCHEMA}.prediction_evaluation_results"


# ============================================================
# 🔐 Guard Login (samakan dengan app Anda)
# ============================================================
if not st.session_state.get("logged_in", True):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ============================================================
# 🔌 DB Connection (IDENTIK dengan halaman labeling)
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


def exec_sql(engine, sql: str, params: Optional[dict] = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})


def assert_dataset_table_exists(engine) -> None:
    sql = """
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = :schema AND table_name = 'dataset_supervised'
    LIMIT 1
    """
    df = read_df(engine, sql, {"schema": SCHEMA})
    if df.empty:
        st.error(f"Tabel sumber tidak ditemukan: `{T_LABEL}`. Periksa koneksi DB (st.secrets).")
        st.stop()


def ensure_eval_table(engine) -> None:
    ddl = f"""
    CREATE TABLE IF NOT EXISTS {T_EVAL}
    (
        eval_id uuid DEFAULT gen_random_uuid(),
        run_time timestamptz NOT NULL DEFAULT now(),

        -- identity of dataset
        jenis_pendekatan text NOT NULL,
        modeling_id uuid NOT NULL,
        window_days integer NOT NULL,
        time_col text NOT NULL DEFAULT '',
        include_noise boolean,
        eligible_rule text,
        split_name text,

        -- evaluation setup
        model_name text NOT NULL,
        test_size double precision NOT NULL,
        random_state integer NOT NULL,

        -- metrics (positive class = insiden berulang)
        precision_pos double precision NOT NULL,
        recall_pos double precision NOT NULL,
        f1_pos double precision NOT NULL,

        -- confusion matrix
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


# ============================================================
# 📥 Load Dataset
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
      COUNT(*) AS n_rows,
      SUM(CASE WHEN label_berulang = 1 THEN 1 ELSE 0 END) AS n_pos
    FROM {T_LABEL}
    GROUP BY 1,2,3,4,5,6,7
    ORDER BY last_run_time DESC
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
    q = f"""
    SELECT
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
      tgl_submit,
      event_time
    FROM {T_LABEL}
    WHERE jenis_pendekatan = :jenis_pendekatan
      AND modeling_id = CAST(:modeling_id AS uuid)
      AND window_days = :window_days
      AND time_col = :time_col
      AND COALESCE(include_noise,false) = :include_noise
      AND COALESCE(eligible_rule,'') = :eligible_rule
      AND COALESCE(split_name,'') = :split_name
    ORDER BY COALESCE(tgl_submit, event_time)
    LIMIT :limit
    """
    df = pd.read_sql(
        text(q),
        _engine,
        params=dict(
            jenis_pendekatan=jenis_pendekatan,
            modeling_id=modeling_id,
            window_days=window_days,
            time_col=time_col,
            include_noise=include_noise,
            eligible_rule=eligible_rule,
            split_name=split_name,
            limit=limit,
        ),
    )

    df["label_berulang"] = pd.to_numeric(df["label_berulang"], errors="coerce").fillna(0).astype(int)
    for c in ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    for c in ["site", "assignee", "modul", "sub_modul"]:
        df[c] = df[c].fillna("UNKNOWN").astype(str)

    return df


# ============================================================
# 🧠 Modeling & Save
# ============================================================
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


def build_preprocessor(cat_cols: List[str], num_cols: List[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        remainder="drop",
    )


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


def upsert_evaluation(engine, meta: dict, rows: List[EvalRow]) -> int:
    sql = f"""
    INSERT INTO {T_EVAL} (
        jenis_pendekatan,
        modeling_id,
        window_days,
        time_col,
        include_noise,
        eligible_rule,
        split_name,
        model_name,
        test_size,
        random_state,
        precision_pos,
        recall_pos,
        f1_pos,
        tp,
        fn,
        fp,
        tn
    )
    VALUES (
        :jenis_pendekatan,
        CAST(:modeling_id AS uuid),
        :window_days,
        :time_col,
        :include_noise,
        :eligible_rule,
        :split_name,
        :model_name,
        :test_size,
        :random_state,
        :precision_pos,
        :recall_pos,
        :f1_pos,
        :tp,
        :fn,
        :fp,
        :tn
    )
    ON CONFLICT (
        jenis_pendekatan,
        modeling_id,
        window_days,
        time_col,
        split_name,
        model_name,
        test_size,
        random_state
    )
    DO UPDATE SET
        run_time      = now(),
        precision_pos = EXCLUDED.precision_pos,
        recall_pos    = EXCLUDED.recall_pos,
        f1_pos        = EXCLUDED.f1_pos,
        tp            = EXCLUDED.tp,
        fn            = EXCLUDED.fn,
        fp            = EXCLUDED.fp,
        tn            = EXCLUDED.tn;
    """

    payload = []
    for r in rows:
        payload.append(
            dict(
                jenis_pendekatan=meta["jenis_pendekatan"],
                modeling_id=meta["modeling_id"],
                window_days=int(meta["window_days"]),
                time_col=str(meta["time_col"]),
                include_noise=meta["include_noise"],
                eligible_rule=meta["eligible_rule"],
                split_name=str(meta["split_name"]),
                model_name=r.model,
                test_size=float(meta["test_size"]),
                random_state=int(meta["random_state"]),
                precision_pos=float(r.precision_pos),
                recall_pos=float(r.recall_pos),
                f1_pos=float(r.f1_pos),
                tp=int(r.tp),
                fn=int(r.fn),
                fp=int(r.fp),
                tn=int(r.tn),
            )
        )

    with engine.begin() as conn:
        res = conn.execute(text(sql), payload)
        # rowcount pada executemany bisa tidak selalu akurat di psycopg2, tapi tetap berguna sebagai indikasi
        return int(res.rowcount or 0)


# ============================================================
# 🧭 UI
# ============================================================
#st.set_page_config(page_title="Evaluasi Model Prediksi", layout="wide")
st.title("🔮 Evaluasi Model Prediksi Insiden Berulang (AUTO-SAVE)")
st.caption(f"Sumber: `{T_LABEL}` • Output: `{T_EVAL}`")

engine = get_engine()
assert_dataset_table_exists(engine)
ensure_eval_table(engine)

runs = fetch_runs(engine)
if runs.empty:
    st.warning("dataset_supervised kosong.")
    st.stop()

with st.expander("Ringkasan run tersedia", expanded=False):
    st.dataframe(runs, use_container_width=True)

st.subheader("1) Pilih Dataset")
c1, c2, c3 = st.columns(3)
with c1:
    jenis = st.selectbox("jenis_pendekatan", sorted(runs["jenis_pendekatan"].unique().tolist()))
sub = runs[runs["jenis_pendekatan"] == jenis].copy()

with c2:
    modeling_id = st.selectbox("modeling_id", sub["modeling_id"].astype(str).unique().tolist())
sub = sub[sub["modeling_id"].astype(str) == str(modeling_id)].copy()

with c3:
    window_days = st.selectbox("window_days", sorted(sub["window_days"].unique().tolist()))
sub = sub[sub["window_days"] == int(window_days)].copy()

c4, c5, c6 = st.columns([1.2, 1, 1.2])
with c4:
    time_col = st.selectbox("time_col", sub["time_col"].astype(str).unique().tolist())
sub = sub[sub["time_col"].astype(str) == str(time_col)].copy()

with c5:
    include_noise = st.selectbox("include_noise", [False, True], index=0)
sub = sub[sub["include_noise"].astype(bool) == bool(include_noise)].copy()

with c6:
    eligible_rule = st.selectbox("eligible_rule", sub["eligible_rule"].astype(str).unique().tolist())
sub = sub[sub["eligible_rule"].astype(str) == str(eligible_rule)].copy()

split_name = st.selectbox("split_name", sub["split_name"].astype(str).unique().tolist())
limit = st.number_input("Limit data", min_value=1000, max_value=500000, value=100000, step=10000)

df = load_dataset(
    engine,
    jenis_pendekatan=jenis,
    modeling_id=str(modeling_id),
    window_days=int(window_days),
    time_col=str(time_col),
    include_noise=bool(include_noise),
    eligible_rule=str(eligible_rule),
    split_name=str(split_name),
    limit=int(limit),
)

pos = int((df["label_berulang"] == 1).sum())
st.write(f"Jumlah data: **{len(df):,}** | Label berulang=1: **{pos:,}**")

if len(df) < 200 or pos < 5:
    st.warning("Dataset terlalu kecil / positif terlalu sedikit.")
    st.stop()

st.subheader("2) Konfigurasi & Pilih Model")
k1, k2, k3 = st.columns(3)
with k1:
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
with k2:
    random_state = st.number_input("random_state", min_value=0, max_value=9999, value=42, step=1)
with k3:
    st.caption("Metrik fokus kelas minoritas (label_berulang=1).")

m1, m2, m3 = st.columns(3)
with m1:
    use_rf = st.checkbox("Random Forest", value=True)
with m2:
    use_xgb = st.checkbox("XGBoost", value=True, disabled=not XGB_OK)
    if not XGB_OK:
        st.caption("xgboost tidak terpasang → XGBoost dinonaktifkan.")
with m3:
    use_svm = st.checkbox("SVM (LinearSVC)", value=True)

run_btn = st.button("🚀 Jalankan Training + Evaluasi + SIMPAN", type="primary")

if run_btn:
    if not (use_rf or use_svm or use_xgb):
        st.error("Pilih minimal 1 model.")
        st.stop()

    # features
    cat_cols = ["site", "assignee", "modul", "sub_modul"]
    num_cols = ["gap_days", "n_member_cluster", "n_episode_cluster", "n_member_episode", "window_days"]

    X = df[cat_cols + num_cols].copy()
    y = df["label_berulang"].astype(int).values

    pre = build_preprocessor(cat_cols, num_cols)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=y,
    )

    cms: Dict[str, np.ndarray] = {}
    rows_for_db: List[EvalRow] = []

    with st.spinner("Training & evaluasi..."):
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
            rf.fit(X_train, y_train)
            row, cm = evaluate_binary(y_test, rf.predict(X_test), "Random Forest")
            rows_for_db.append(row)
            cms[row.model] = cm

        if use_xgb and XGB_OK:
            pos_train = int((y_train == 1).sum())
            neg_train = int((y_train == 0).sum())
            scale_pos_weight = float(neg_train / pos_train) if pos_train > 0 else 1.0

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
                        scale_pos_weight=scale_pos_weight,
                        random_state=int(random_state),
                        n_jobs=-1,
                    )),
                ]
            )
            xgb.fit(X_train, y_train)
            row, cm = evaluate_binary(y_test, xgb.predict(X_test), "XGBoost")
            rows_for_db.append(row)
            cms[row.model] = cm

        if use_svm:
            svm = Pipeline([("pre", pre), ("clf", LinearSVC(class_weight="balanced"))])
            svm.fit(X_train, y_train)
            row, cm = evaluate_binary(y_test, svm.predict(X_test), "SVM")
            rows_for_db.append(row)
            cms[row.model] = cm

    if not rows_for_db:
        st.error("Tidak ada hasil evaluasi yang terbentuk (rows_for_db kosong).")
        st.stop()

    # tabel untuk Bab 4.7
    out = pd.DataFrame(
        [
            {
                "Model Prediktif": r.model,
                "Precision (Berulang)": round(r.precision_pos, 4),
                "Recall (Berulang)": round(r.recall_pos, 4),
                "F1-Score (Berulang)": round(r.f1_pos, 4),
                "True Positive (TP)": r.tp,
                "False Negative (FN)": r.fn,
                "False Positive (FP)": r.fp,
            }
            for r in rows_for_db
        ]
    )

    st.success("Evaluasi selesai.")
    st.subheader("Hasil Evaluasi")
    st.dataframe(out, use_container_width=True)

    st.markdown("### Confusion Matrix per Model")
    cols = st.columns(len(cms))
    for i, (name, cm) in enumerate(cms.items()):
        with cols[i]:
            st.write(f"**{name}**")
            st.dataframe(
                pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
                use_container_width=True,
            )

    # AUTO SAVE (langsung simpan setelah evaluasi)
    meta = dict(
        jenis_pendekatan=jenis,
        modeling_id=str(modeling_id),
        window_days=int(window_days),
        time_col=str(time_col),
        include_noise=bool(include_noise),
        eligible_rule=str(eligible_rule),
        split_name=str(split_name),
        test_size=float(test_size),
        random_state=int(random_state),
    )

    with st.spinner("Menyimpan hasil evaluasi ke database..."):
        rc = upsert_evaluation(engine, meta, rows_for_db)

    st.success(f"✅ Hasil evaluasi tersimpan ke `{T_EVAL}`")
    st.caption(f"DEBUG: rowcount (indikatif) = {rc}")

    # Verifikasi hasil tersimpan
    st.subheader("Verifikasi Data Tersimpan")
    df_saved = read_df(
        engine,
        f"""
        SELECT run_time, model_name, precision_pos, recall_pos, f1_pos, tp, fn, fp, tn
        FROM {T_EVAL}
        WHERE jenis_pendekatan = :jenis_pendekatan
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
          AND COALESCE(split_name,'') = :split_name
          AND test_size = :test_size
          AND random_state = :random_state
        ORDER BY run_time DESC, model_name ASC
        """,
        {
            "jenis_pendekatan": meta["jenis_pendekatan"],
            "modeling_id": meta["modeling_id"],
            "window_days": meta["window_days"],
            "time_col": meta["time_col"],
            "split_name": meta["split_name"],
            "test_size": meta["test_size"],
            "random_state": meta["random_state"],
        },
    )
    st.dataframe(df_saved, use_container_width=True)
