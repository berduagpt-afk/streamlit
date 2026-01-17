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

DIST_TABLE = "predict_to_semantik_distance"   # hasil top-k distance
GT_TABLE   = "dataset_supervised"             # ground truth (label_berulang)
SPLIT_NAME = "test"                           # evaluasi hanya test

OUT_TABLE  = "eval_distance_thresholds"       # simpan hasil sweep

# default threshold untuk “Langkah 2–5” (single-run)
DEFAULT_THRESHOLD = float(os.getenv("DEFAULT_THRESHOLD", "0.80"))

# sweep threshold (Langkah 6)
TH_START = float(os.getenv("TH_START", "0.60"))
TH_END   = float(os.getenv("TH_END", "0.90"))
TH_STEP  = float(os.getenv("TH_STEP", "0.02"))

# kalau dataset_supervised punya lebih dari satu baris per incident_number (mis. multi-run),
# set ini untuk pilih satu saja (dedup) pakai MAX(label_berulang) -> aman
DEDUP_GT = os.getenv("DEDUP_GT", "1") == "1"


# ============================================================
# SQL HELPERS
# ============================================================
def ensure_output_table(engine: Engine) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {SCHEMA};

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{OUT_TABLE} (
        eval_id BIGSERIAL PRIMARY KEY,
        split_name TEXT NOT NULL,
        threshold DOUBLE PRECISION NOT NULL,
        n_test BIGINT NOT NULL,
        tp BIGINT NOT NULL,
        fn BIGINT NOT NULL,
        fp BIGINT NOT NULL,
        tn BIGINT NOT NULL,
        precision DOUBLE PRECISION,
        recall DOUBLE PRECISION,
        f1_score DOUBLE PRECISION,
        created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_split_threshold
    ON {SCHEMA}.{OUT_TABLE} (split_name, threshold);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


def create_views(engine: Engine) -> None:
    """
    Langkah 1–3:
    - v_top1_distance: ambil top1 cosine_sim per incident_number + model_name
    - v_eval_base: join ke dataset_supervised split=test -> dataset evaluasi dasar
    """
    # View top1
    v_top1 = f"""
    CREATE OR REPLACE VIEW {SCHEMA}.v_top1_distance AS
    SELECT
        d.incident_number,
        d.model_name,
        d.target_cluster_id,
        d.cosine_sim
    FROM {SCHEMA}.{DIST_TABLE} d
    WHERE d.topk_rank = 1;
    """

    # Ground truth: dedup jika perlu
    if DEDUP_GT:
        gt_sub = f"""
        SELECT incident_number,
               MAX(label_berulang)::int AS label_berulang
        FROM {SCHEMA}.{GT_TABLE}
        WHERE split_name = :split_name
        GROUP BY 1
        """
    else:
        gt_sub = f"""
        SELECT incident_number,
               label_berulang::int AS label_berulang
        FROM {SCHEMA}.{GT_TABLE}
        WHERE split_name = :split_name
        """

    v_eval_base = f"""
    CREATE OR REPLACE VIEW {SCHEMA}.v_eval_base AS
    WITH gt AS (
        {gt_sub}
    )
    SELECT
        gt.incident_number,
        gt.label_berulang AS actual_label,
        t.model_name,
        t.cosine_sim,
        t.target_cluster_id
    FROM gt
    JOIN {SCHEMA}.v_top1_distance t
      ON t.incident_number = gt.incident_number;
    """

    with engine.begin() as conn:
        conn.execute(text(v_top1))
        conn.execute(text(v_eval_base), {"split_name": SPLIT_NAME})


def fetch_eval_base(engine: Engine) -> pd.DataFrame:
    q = f"""
    SELECT incident_number, actual_label, model_name, cosine_sim
    FROM {SCHEMA}.v_eval_base
    WHERE cosine_sim IS NOT NULL
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(q), conn)

    # validasi label
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


def compute_for_threshold(df: pd.DataFrame, th: float) -> Tuple[int, Confusion]:
    y_true = df["actual_label"].to_numpy(dtype=int)
    y_pred = (df["cosine_sim"].to_numpy(dtype=float) >= th).astype(int)
    cm = confusion_from_arrays(y_true, y_pred)
    return len(df), cm


# ============================================================
# SAVE RESULTS
# ============================================================
def save_threshold_results(engine: Engine, split_name: str, rows: List[Dict]) -> None:
    sql = f"""
    INSERT INTO {SCHEMA}.{OUT_TABLE}
    (split_name, threshold, n_test, tp, fn, fp, tn, precision, recall, f1_score)
    VALUES
    (:split_name, :threshold, :n_test, :tp, :fn, :fp, :tn, :precision, :recall, :f1_score)
    """
    with engine.begin() as conn:
        conn.execute(text(sql), rows)


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    engine = make_engine()

    # Langkah 1–3: views + base dataset
    ensure_output_table(engine)
    create_views(engine)

    df = fetch_eval_base(engine)
    if df.empty:
        print("⚠️ v_eval_base kosong. Pastikan:")
        print("- predict_to_semantik_distance sudah terisi (topk_rank=1 ada)")
        print("- dataset_supervised punya split_name='test'")
        print("- incident_number overlap antara keduanya")
        return

    print(f"✅ Base evaluasi loaded: {len(df):,} baris (split={SPLIT_NAME})")
    print(f"   contoh model_name: {df['model_name'].value_counts().head(3).to_dict()}")

    # Langkah 4–5: confusion matrix + metrics untuk 1 threshold default
    n_test, cm = compute_for_threshold(df, DEFAULT_THRESHOLD)
    print("\n=== Evaluasi Rule-based (Single Threshold) ===")
    print(f"Threshold: {DEFAULT_THRESHOLD}")
    print(f"n_test: {n_test}")
    print(f"TP={cm.tp} FN={cm.fn} FP={cm.fp} TN={cm.tn}")
    print(f"Precision={cm.precision:.4f} Recall={cm.recall:.4f} F1={cm.f1:.4f}")

    # Langkah 6: sweep threshold dan simpan ke tabel
    thresholds = np.arange(TH_START, TH_END + 1e-9, TH_STEP)
    out_rows: List[Dict] = []

    for th in thresholds:
        n_test, cm = compute_for_threshold(df, float(th))
        out_rows.append(
            {
                "split_name": SPLIT_NAME,
                "threshold": float(th),
                "n_test": int(n_test),
                "tp": cm.tp,
                "fn": cm.fn,
                "fp": cm.fp,
                "tn": cm.tn,
                "precision": cm.precision if not np.isnan(cm.precision) else None,
                "recall": cm.recall if not np.isnan(cm.recall) else None,
                "f1_score": cm.f1 if not np.isnan(cm.f1) else None,
            }
        )

    save_threshold_results(engine, SPLIT_NAME, out_rows)

    # tampilkan top-5 threshold berdasarkan F1
    df_out = pd.DataFrame(out_rows)
    df_out = df_out.dropna(subset=["f1_score"]).sort_values("f1_score", ascending=False).head(10)

    print("\n=== Top Threshold by F1 (Rule-based) ===")
    print(df_out[["threshold", "tp", "fn", "fp", "tn", "precision", "recall", "f1_score"]].to_string(index=False))

    print(f"\n✅ Hasil sweep disimpan di {SCHEMA}.{OUT_TABLE}")


if __name__ == "__main__":
    main()
