from __future__ import annotations

import os
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# XGBoost opsional
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False


# ============================================================
# DB SETTINGS (JANGAN DIUBAH)
# ============================================================
def make_engine() -> Engine:
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db   = os.getenv("PGDATABASE", "incident_djp")
    user = os.getenv("PGUSER", "postgres")
    pw   = os.getenv("PGPASSWORD", "admin*123")
    url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url, pool_pre_ping=True)


# ============================================================
# KONFIG
# ============================================================
SCHEMA = "lasis_djp"

DIST_TABLE = "predict_to_semantik_distance"
LABEL_TABLE = "incident_semantic_labels"

TOPK = int(os.getenv("TOPK", "5"))
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

# jika ada missing distance, default: drop baris yang tidak punya fitur sama sekali
INCLUDE_MISSING = os.getenv("INCLUDE_MISSING", "0") == "1"

OUT_METRICS = "prediction_ml_distance_results_cv"
OUT_PREDALL = "prediction_ml_distance_predictions_all"


# ============================================================
# DDL OUTPUT TABLES
# ============================================================
def ensure_tables(engine: Engine) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{OUT_METRICS} (
        run_id BIGSERIAL PRIMARY KEY,
        run_time TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
        scope TEXT NOT NULL,
        model_name TEXT NOT NULL,
        algo TEXT NOT NULL,
        cv_folds INT NOT NULL,

        n_total BIGINT NOT NULL,

        tp BIGINT NOT NULL,
        fn BIGINT NOT NULL,
        fp BIGINT NOT NULL,
        tn BIGINT NOT NULL,

        precision_1 DOUBLE PRECISION,
        recall_1 DOUBLE PRECISION,
        f1_1 DOUBLE PRECISION,
        accuracy DOUBLE PRECISION,

        params_json JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_{OUT_METRICS}_time
    ON {SCHEMA}.{OUT_METRICS} (run_time);

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{OUT_PREDALL} (
        pred_id BIGSERIAL PRIMARY KEY,
        run_time TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
        scope TEXT NOT NULL,
        model_name TEXT NOT NULL,
        algo TEXT NOT NULL,

        incident_number TEXT NOT NULL,
        y_true INT,
        y_pred INT NOT NULL,
        proba_1 DOUBLE PRECISION,

        params_json JSONB
    );

    CREATE INDEX IF NOT EXISTS idx_{OUT_PREDALL}_inc
    ON {SCHEMA}.{OUT_PREDALL} (incident_number);

    CREATE INDEX IF NOT EXISTS idx_{OUT_PREDALL}_algo
    ON {SCHEMA}.{OUT_PREDALL} (algo);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ============================================================
# LOAD FEATURES
# ============================================================
def load_dataset(engine: Engine, topk: int) -> pd.DataFrame:
    top_cols = []
    for k in range(1, topk + 1):
        top_cols.append(f"MAX(CASE WHEN topk_rank={k} THEN cosine_sim END) AS top{k}_sim")
    top_cols_sql = ",\n            ".join(top_cols)

    if INCLUDE_MISSING:
        q = f"""
        WITH agg AS (
          SELECT
            incident_number,
            model_name,
            {top_cols_sql},
            AVG(cosine_sim) FILTER (WHERE topk_rank <= {topk}) AS mean_topk,
            STDDEV(cosine_sim) FILTER (WHERE topk_rank <= {topk}) AS std_topk
          FROM {SCHEMA}.{DIST_TABLE}
          GROUP BY 1,2
        )
        SELECT
          l.incident_number,
          COALESCE(a.model_name, 'unknown') AS model_name,
          {", ".join([f"a.top{k}_sim" for k in range(1, topk + 1)])},
          a.mean_topk,
          a.std_topk,
          (a.top1_sim - COALESCE(a.top2_sim, 0)) AS gap12,
          l.label::int AS y
        FROM {SCHEMA}.{LABEL_TABLE} l
        LEFT JOIN agg a
          ON a.incident_number = l.incident_number;
        """
    else:
        q = f"""
        WITH agg AS (
          SELECT
            incident_number,
            model_name,
            {top_cols_sql},
            AVG(cosine_sim) FILTER (WHERE topk_rank <= {topk}) AS mean_topk,
            STDDEV(cosine_sim) FILTER (WHERE topk_rank <= {topk}) AS std_topk
          FROM {SCHEMA}.{DIST_TABLE}
          GROUP BY 1,2
        )
        SELECT
          a.incident_number,
          a.model_name,
          {", ".join([f"a.top{k}_sim" for k in range(1, topk + 1)])},
          a.mean_topk,
          a.std_topk,
          (a.top1_sim - COALESCE(a.top2_sim, 0)) AS gap12,
          l.label::int AS y
        FROM agg a
        JOIN {SCHEMA}.{LABEL_TABLE} l
          ON l.incident_number = a.incident_number;
        """

    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)

    return df


# ============================================================
# MODELS
# ============================================================
def build_models(random_state: int) -> List[Tuple[str, object, Dict]]:
    models: List[Tuple[str, object, Dict]] = []

    rf = RandomForestClassifier(
        n_estimators=400,
        random_state=random_state,
        class_weight="balanced_subsample",
        n_jobs=-1
    )
    models.append(("RandomForest", rf, {"n_estimators": 400, "class_weight": "balanced_subsample"}))

    svm = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=random_state
    )
    models.append(("SVM_RBF", svm, {"kernel": "rbf", "class_weight": "balanced"}))

    if HAS_XGB:
        xgb = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss"
        )
        models.append(("XGBoost", xgb, {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05}))

    return models


def make_pipeline(algo_name: str, estimator):
    # SVM perlu scaling, RF/XGB tidak wajib
    if algo_name.startswith("SVM"):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ])
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", estimator),
    ])


# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[1], average=None, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    return {
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "precision_1": float(pr[0]),
        "recall_1": float(rc[0]),
        "f1_1": float(f1[0]),
        "accuracy": float(acc),
    }


# ============================================================
# SAVE TO DB
# ============================================================
def save_metrics(engine: Engine, row: Dict) -> None:
    sql = f"""
    INSERT INTO {SCHEMA}.{OUT_METRICS}
    (scope, model_name, algo, cv_folds, n_total,
     tp, fn, fp, tn, precision_1, recall_1, f1_1, accuracy, params_json)
    VALUES
    (:scope, :model_name, :algo, :cv_folds, :n_total,
     :tp, :fn, :fp, :tn, :precision_1, :recall_1, :f1_1, :accuracy, CAST(:params_json AS jsonb))
    """
    with engine.begin() as conn:
        conn.execute(text(sql), row)


def save_predictions_all(engine: Engine, rows: List[Dict]) -> None:
    sql = f"""
    INSERT INTO {SCHEMA}.{OUT_PREDALL}
    (scope, model_name, algo, incident_number, y_true, y_pred, proba_1, params_json)
    VALUES
    (:scope, :model_name, :algo, :incident_number, :y_true, :y_pred, :proba_1, CAST(:params_json AS jsonb))
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    engine = make_engine()
    ensure_tables(engine)

    df = load_dataset(engine, topk=TOPK)
    if df.empty:
        print("⚠️ Dataset fitur kosong.")
        return

    feature_cols = [f"top{k}_sim" for k in range(1, TOPK + 1)] + ["mean_topk", "std_topk", "gap12"]

    # Drop baris yang benar-benar tidak punya fitur (kalau include_missing)
    df["nonnull"] = df[feature_cols].notna().sum(axis=1)
    df = df[df["nonnull"] > 0].drop(columns=["nonnull"]).reset_index(drop=True)

    print(f"✅ Dataset: {len(df):,} rows | TOPK={TOPK} | CV={CV_FOLDS} folds | HAS_XGB={HAS_XGB}")

    # Evaluasi per model_name (jika ada lebih dari 1)
    model_names = sorted(df["model_name"].unique().tolist())

    summary = []

    for mname in model_names:
        dfi = df[df["model_name"] == mname].reset_index(drop=True)
        if len(dfi) < CV_FOLDS * 2:
            print(f"⚠️ Skip model_name={mname} karena data terlalu sedikit: {len(dfi)}")
            continue

        X = dfi[feature_cols].astype(float).values
        y = dfi["y"].astype(int).values
        inc = dfi["incident_number"].astype(str).tolist()

        # Pastikan stratified possible
        if len(np.unique(y)) < 2:
            print(f"⚠️ Skip model_name={mname} karena label hanya 1 kelas.")
            continue

        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for algo_name, estimator, params in build_models(RANDOM_STATE):
            pipe = make_pipeline(algo_name, estimator)

            # Prediksi out-of-fold untuk SEMUA data -> confusion matrix total = n_total
            y_pred = cross_val_predict(pipe, X, y, cv=skf, method="predict")

            # probability (kalau ada)
            try:
                proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
            except Exception:
                proba = np.array([None] * len(y), dtype=object)

            met = compute_metrics(y, y_pred)

            params_json = json.dumps({
                "TOPK": TOPK,
                "features": feature_cols,
                "cv_folds": CV_FOLDS,
                "random_state": RANDOM_STATE,
                "algo_params": params,
                "HAS_XGB": HAS_XGB,
                "INCLUDE_MISSING": INCLUDE_MISSING,
            })

            save_metrics(engine, {
                "scope": "ALL100_CV",
                "model_name": mname,
                "algo": algo_name,
                "cv_folds": CV_FOLDS,
                "n_total": int(len(y)),
                **met,
                "params_json": params_json
            })

            # Simpan prediksi untuk semua data (OOF predictions)
            rows_pred = []
            for i in range(len(y)):
                rows_pred.append({
                    "scope": "ALL100_CV",
                    "model_name": mname,
                    "algo": algo_name,
                    "incident_number": inc[i],
                    "y_true": int(y[i]),
                    "y_pred": int(y_pred[i]),
                    "proba_1": None if proba[i] is None else float(proba[i]),
                    "params_json": params_json
                })
            save_predictions_all(engine, rows_pred)

            summary.append({
                "algo": algo_name,
                "model_name": mname,
                "n_total": len(y),
                "precision_1": met["precision_1"],
                "recall_1": met["recall_1"],
                "f1_1": met["f1_1"],
                "tp": met["tp"],
                "fn": met["fn"],
                "fp": met["fp"],
                "tn": met["tn"],
                "accuracy": met["accuracy"],
            })

            print(f"✅ CV saved: {algo_name} | model={mname} | n={len(y)} | F1(1)={met['f1_1']:.4f}")

    if not summary:
        print("⚠️ Tidak ada hasil.")
        return

    df_sum = pd.DataFrame(summary).sort_values(["f1_1", "recall_1", "precision_1"], ascending=False)
    print("\n=== Perbandingan Model (CV, total = n_total) ===")
    print(df_sum[[
        "algo", "model_name", "n_total",
        "precision_1", "recall_1", "f1_1",
        "tp", "fn", "fp", "tn", "accuracy"
    ]].to_string(index=False))

    print(f"\n✅ Metrics: {SCHEMA}.{OUT_METRICS}")
    print(f"✅ Predictions (all rows): {SCHEMA}.{OUT_PREDALL}")

    if not HAS_XGB:
        print("ℹ️ XGBoost tidak terpasang, jadi hanya RF & SVM yang dievaluasi.")


if __name__ == "__main__":
    main()
