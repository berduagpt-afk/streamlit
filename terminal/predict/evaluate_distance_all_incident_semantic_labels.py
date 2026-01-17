from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


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

DIST_TABLE = "predict_to_semantik_distance"   # berisi cosine_sim + topk_rank
LABEL_TABLE = "incident_semantic_labels"      # ground truth label (0/1)

OUT_TABLE = "eval_distance_all_labels"        # simpan hasil sweep

DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.80"))
TH_START = float(os.getenv("TH_START", "0.60"))
TH_END   = float(os.getenv("TH_END", "0.90"))
TH_STEP  = float(os.getenv("TH_STEP", "0.02"))

# True  -> include semua label rows, missing cosine_sim dianggap pred=0
# False -> hanya evaluasi yang punya distance (inner join)
INCLUDE_MISSING_AS_NEGATIVE = os.getenv("INCLUDE_MISSING_AS_NEGATIVE", "1") == "1"


# ============================================================
# OUTPUT TABLE
# ============================================================
def ensure_output_table(engine: Engine) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{OUT_TABLE} (
        eval_id BIGSERIAL PRIMARY KEY,
        scope TEXT NOT NULL,
        threshold DOUBLE PRECISION NOT NULL,
        n_total BIGINT NOT NULL,
        n_missing_distance BIGINT NOT NULL,
        tp BIGINT NOT NULL,
        fn BIGINT NOT NULL,
        fp BIGINT NOT NULL,
        tn BIGINT NOT NULL,
        precision DOUBLE PRECISION,
        recall DOUBLE PRECISION,
        f1_score DOUBLE PRECISION,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_scope_threshold
    ON {SCHEMA}.{OUT_TABLE} (scope, threshold);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


# ============================================================
# VIEWS (DROP dulu biar aman)
# ============================================================
def create_views(engine: Engine) -> None:
    """
    FIX untuk error:
    - PostgreSQL tidak izinkan CREATE OR REPLACE VIEW jika mengurangi kolom.
    - Solusi final: DROP VIEW (CASCADE) lalu CREATE ulang.
    """
    with engine.begin() as conn:
        conn.execute(text(f'DROP VIEW IF EXISTS {SCHEMA}.v_eval_all_labels CASCADE;'))
        conn.execute(text(f'DROP VIEW IF EXISTS {SCHEMA}.v_top1_distance CASCADE;'))

        conn.execute(text(f"""
        CREATE VIEW {SCHEMA}.v_top1_distance AS
        SELECT
            incident_number,
            model_name,
            cosine_sim
        FROM {SCHEMA}.{DIST_TABLE}
        WHERE topk_rank = 1;
        """))

        if INCLUDE_MISSING_AS_NEGATIVE:
            conn.execute(text(f"""
            CREATE VIEW {SCHEMA}.v_eval_all_labels AS
            SELECT
                l.incident_number,
                l.label::int AS actual_label,
                d.model_name,
                d.cosine_sim
            FROM {SCHEMA}.{LABEL_TABLE} l
            LEFT JOIN {SCHEMA}.v_top1_distance d
              ON d.incident_number = l.incident_number;
            """))
        else:
            conn.execute(text(f"""
            CREATE VIEW {SCHEMA}.v_eval_all_labels AS
            SELECT
                l.incident_number,
                l.label::int AS actual_label,
                d.model_name,
                d.cosine_sim
            FROM {SCHEMA}.{LABEL_TABLE} l
            JOIN {SCHEMA}.v_top1_distance d
              ON d.incident_number = l.incident_number;
            """))


def fetch_eval(engine: Engine) -> pd.DataFrame:
    q = f"""
    SELECT incident_number, actual_label, model_name, cosine_sim
    FROM {SCHEMA}.v_eval_all_labels
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)

    df["actual_label"] = df["actual_label"].astype(int)
    return df


# ============================================================
# METRICS
# ============================================================
@dataclass
class Confusion:
    tp: int
    fn: int
    fp: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom else float("nan")

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom else float("nan")

    @property
    def f1(self) -> float:
        denom = 2 * self.tp + self.fp + self.fn
        return float((2 * self.tp) / denom) if denom else float("nan")


def confusion_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Confusion:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    return Confusion(tp=tp, fn=fn, fp=fp, tn=tn)


def compute_for_threshold(df: pd.DataFrame, th: float) -> Tuple[int, int, Confusion]:
    y_true = df["actual_label"].to_numpy(dtype=int)

    sims = df["cosine_sim"].to_numpy()
    # missing cosine_sim -> pred 0
    sims = np.where(pd.isna(sims), -1.0, sims.astype(float))

    y_pred = (sims >= th).astype(int)
    cm = confusion_from_arrays(y_true, y_pred)

    n_total = len(df)
    n_missing = int(pd.isna(df["cosine_sim"]).sum())
    return n_total, n_missing, cm


# ============================================================
# SAVE RESULTS
# ============================================================
def save_results(engine: Engine, rows: List[Dict]) -> None:
    sql = f"""
    INSERT INTO {SCHEMA}.{OUT_TABLE}
    (scope, threshold, n_total, n_missing_distance, tp, fn, fp, tn, precision, recall, f1_score)
    VALUES
    (:scope, :threshold, :n_total, :n_missing_distance, :tp, :fn, :fp, :tn, :precision, :recall, :f1_score)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    engine = make_engine()

    ensure_output_table(engine)
    create_views(engine)

    df = fetch_eval(engine)
    if df.empty:
        print("⚠️ Data evaluasi kosong. Pastikan incident_semantic_labels berisi data.")
        return

    scope = "ALL_INCIDENT_SEMANTIC_LABELS"
    n_missing = int(df["cosine_sim"].isna().sum())

    print(f"✅ Loaded {len(df):,} rows from {SCHEMA}.{LABEL_TABLE}")
    print(f"✅ Missing distance (cosine_sim NULL): {n_missing:,} (treated as pred=0 if enabled)")

    # Distribusi label
    print("\n=== Distribusi Actual Label ===")
    print(df["actual_label"].value_counts(dropna=False).to_string())

    # Single threshold
    n_total, n_missing, cm = compute_for_threshold(df, DEFAULT_THRESHOLD)
    print("\n=== Evaluasi Rule-based (Single Threshold) ===")
    print(f"Threshold: {DEFAULT_THRESHOLD}")
    print(f"n_total: {n_total:,} | missing_distance: {n_missing:,}")
    print(f"TP={cm.tp} FN={cm.fn} FP={cm.fp} TN={cm.tn}")
    print(f"Precision={cm.precision:.4f} Recall={cm.recall:.4f} F1={cm.f1:.4f}")

    # Sweep threshold
    thresholds = np.arange(TH_START, TH_END + 1e-9, TH_STEP)
    rows_out: List[Dict] = []

    for th in thresholds:
        n_total, n_missing, cm = compute_for_threshold(df, float(th))
        rows_out.append(
            {
                "scope": scope,
                "threshold": float(th),
                "n_total": int(n_total),
                "n_missing_distance": int(n_missing),
                "tp": cm.tp,
                "fn": cm.fn,
                "fp": cm.fp,
                "tn": cm.tn,
                "precision": cm.precision if not np.isnan(cm.precision) else None,
                "recall": cm.recall if not np.isnan(cm.recall) else None,
                "f1_score": cm.f1 if not np.isnan(cm.f1) else None,
            }
        )

    save_results(engine, rows_out)

    df_out = (
        pd.DataFrame(rows_out)
        .dropna(subset=["f1_score"])
        .sort_values("f1_score", ascending=False)
        .head(10)
    )

    print("\n=== Top Threshold by F1 ===")
    print(df_out[["threshold", "tp", "fn", "fp", "tn", "precision", "recall", "f1_score", "n_missing_distance"]].to_string(index=False))

    print(f"\n✅ Hasil sweep disimpan di {SCHEMA}.{OUT_TABLE}")


if __name__ == "__main__":
    main()
