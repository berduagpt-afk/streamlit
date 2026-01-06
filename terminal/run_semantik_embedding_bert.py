# terminal/run_semantik_embedding_bert.py
# ============================================================
# Offline Runner — Semantik Embedding (IndoBERT / SBERT)
# FINAL VERSION (SAFE + AUTO CREATE TABLES + TEST LIMIT DEFAULT)
#
# Source : lasis_djp.incident_semantik
# Output :
#   - lasis_djp.semantik_embedding_runs
#   - lasis_djp.semantik_embedding_vectors
#
# Default mode: proses 1000 tiket dulu (agar tidak menunggu lama lalu error).
# Setelah sukses, naikkan --limit 5000 lalu --limit 0 (full).
#
# Usage:
#   python terminal/run_semantik_embedding_bert.py --provider st
#   python terminal/run_semantik_embedding_bert.py --provider tf --batch-size 16 --max-length 256
#   python terminal/run_semantik_embedding_bert.py --provider st --limit 5000
#   python terminal/run_semantik_embedding_bert.py --provider st --limit 0
# ============================================================

from __future__ import annotations

import argparse
import sys
import time
import uuid
from datetime import datetime, timezone, timedelta
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from psycopg2.extras import Json


# ============================================================
# 🔐 DB CONFIG (HARDCODED)
# ============================================================

DB = {
    "host": "localhost",
    "port": 5432,
    "database": "incident_djp",
    "user": "postgres",
    "password": "admin*123",
}

SCHEMA = "lasis_djp"
SRC_TABLE = "incident_semantik"
RUNS_TABLE = "semantik_embedding_runs"
VECS_TABLE = "semantik_embedding_vectors"


# ============================================================
# DB ENGINE
# ============================================================

def get_engine() -> Engine:
    url = (
        f"postgresql+psycopg2://{DB['user']}:{DB['password']}"
        f"@{DB['host']}:{DB['port']}/{DB['database']}"
    )
    return create_engine(url, pool_pre_ping=True, future=True)


# ============================================================
# ENSURE OUTPUT TABLES
# ============================================================

def ensure_tables(engine: Engine) -> None:
    ddl_runs = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{RUNS_TABLE} (
        run_id uuid PRIMARY KEY,
        run_time timestamp without time zone NOT NULL,
        source_table text NOT NULL,
        n_rows integer NOT NULL,
        provider text NOT NULL,
        model_name text NOT NULL,
        device text NOT NULL,
        batch_size integer NOT NULL,
        max_length integer NOT NULL,
        normalize_embeddings boolean NOT NULL,
        params_json jsonb NOT NULL,
        notes text
    );
    """

    ddl_vectors = f"""
    CREATE TABLE IF NOT EXISTS {SCHEMA}.{VECS_TABLE} (
        run_id uuid NOT NULL,
        incident_number text NOT NULL,
        tgl_submit timestamp without time zone,
        n_chars integer,
        embedding_dim integer NOT NULL,
        embedding_json jsonb NOT NULL,
        PRIMARY KEY (run_id, incident_number)
    );
    """

    ddl_indexes = f"""
    CREATE INDEX IF NOT EXISTS idx_{VECS_TABLE}_run_id
        ON {SCHEMA}.{VECS_TABLE}(run_id);
    CREATE INDEX IF NOT EXISTS idx_{VECS_TABLE}_tgl_submit
        ON {SCHEMA}.{VECS_TABLE}(tgl_submit);
    """

    with engine.begin() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA};"))
        conn.execute(text(ddl_runs))
        conn.execute(text(ddl_vectors))
        conn.execute(text(ddl_indexes))


# ============================================================
# CLI ARGUMENTS
# ============================================================

def parse_args():
    ap = argparse.ArgumentParser("Offline Semantik Embedding (FINAL)")
    ap.add_argument(
        "--provider",
        choices=["st", "tf"],
        default="st",
        help="st=sentence-transformers | tf=transformers IndoBERT",
    )
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Jumlah tiket yang diproses. 1000 untuk test cepat. 0 = full data.",
    )
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--notes", default="")
    return ap.parse_args()


# ============================================================
# EMBEDDING BACKENDS
# ============================================================

def embed_sentence_transformers(texts: List[str], device: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    model = SentenceTransformer(model_name, device=device)

    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(vecs, dtype=np.float32)


def embed_indobert(texts: List[str], device: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "indobenchmark/indobert-base-p1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    def mean_pooling(out, mask):
        tok = out.last_hidden_state
        mask = mask.unsqueeze(-1).expand(tok.size()).float()
        return (tok * mask).sum(1) / mask.sum(1)

    all_vecs = []
    total = len(texts)

    for i in range(0, total, batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)

        vec = mean_pooling(out, enc["attention_mask"])
        vec = vec.cpu().numpy().astype(np.float32)

        # L2 normalize (cosine-friendly)
        norm = np.linalg.norm(vec, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-9, None)
        vec = vec / norm

        all_vecs.append(vec)
        print(f"[EMBED] {min(i + batch_size, total):,}/{total:,}")

    return np.vstack(all_vecs)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    engine = get_engine()

    print("[INFO] Ensuring output tables ...")
    ensure_tables(engine)

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------
    limit_sql = f"LIMIT {int(args.limit)}" if int(args.limit) > 0 else ""
    sql = f"""
        SELECT incident_number, text_semantic, tgl_semantik
        FROM {SCHEMA}.{SRC_TABLE}
        WHERE text_semantic IS NOT NULL AND text_semantic <> ''
        {limit_sql}
    """

    df = pd.read_sql(text(sql), engine)
    if df.empty:
        print("[ERROR] Tidak ada data untuk diproses.")
        return

    df["text_semantic"] = df["text_semantic"].astype(str)
    texts = df["text_semantic"].tolist()

    print(f"[INFO] Loaded {len(texts):,} tickets (limit={args.limit})")

    # --------------------------------------------------------
    # EMBEDDING
    # --------------------------------------------------------
    t0 = time.time()

    if args.provider == "st":
        vectors = embed_sentence_transformers(texts, device=args.device, batch_size=args.batch_size)
        model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    else:
        vectors = embed_indobert(texts, device=args.device, batch_size=args.batch_size, max_length=args.max_length)
        model_name = "indobenchmark/indobert-base-p1"

    dim = int(vectors.shape[1])

    # --------------------------------------------------------
    # SAVE RUN METADATA
    # --------------------------------------------------------
    run_id = uuid.uuid4()  # IMPORTANT: UUID object (no ::uuid)
    WIB = timezone(timedelta(hours=7))
    run_time = datetime.now(WIB).replace(tzinfo=None)

    notes = args.notes.strip() if args.notes else ""
    if not notes:
        notes = f"test run (limit={args.limit})"

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {SCHEMA}.{RUNS_TABLE}
                (run_id, run_time, source_table, n_rows,
                 provider, model_name, device,
                 batch_size, max_length, normalize_embeddings,
                 params_json, notes)
                VALUES
                (:run_id, :run_time, :source_table, :n_rows,
                 :provider, :model_name, :device,
                 :batch_size, :max_length, :normalize_embeddings,
                 :params_json, :notes)
            """),
            {
                "run_id": run_id,
                "run_time": run_time,
                "source_table": f"{SCHEMA}.{SRC_TABLE}",
                "n_rows": int(len(df)),
                "provider": args.provider,
                "model_name": model_name,
                "device": args.device,
                "batch_size": int(args.batch_size),
                "max_length": int(args.max_length),
                "normalize_embeddings": True,
                "params_json": Json(vars(args)),
                "notes": notes,
            },
        )

    # --------------------------------------------------------
    # SAVE VECTORS (executemany)
    # --------------------------------------------------------
    rows = []
    for i, row in df.iterrows():
        rows.append(
            {
                "run_id": run_id,
                "incident_number": str(row["incident_number"]),
                "tgl_submit": row["tgl_semantik"],
                "n_chars": int(len(row["text_semantic"])),
                "embedding_dim": dim,
                "embedding_json": Json(vectors[i].tolist()),
            }
        )

    with engine.begin() as conn:
        conn.execute(
            text(f"""
                INSERT INTO {SCHEMA}.{VECS_TABLE}
                (run_id, incident_number, tgl_submit, n_chars, embedding_dim, embedding_json)
                VALUES
                (:run_id, :incident_number, :tgl_submit, :n_chars, :embedding_dim, :embedding_json)
                ON CONFLICT (run_id, incident_number) DO NOTHING
            """),
            rows,
        )

    elapsed = time.time() - t0
    print(f"[DONE] run_id={run_id} | rows={len(df):,} | dim={dim} | time={elapsed:.1f}s")
    print("[NEXT] Cek DB:")
    print(f"  SELECT COUNT(*) FROM {SCHEMA}.{VECS_TABLE} WHERE run_id = '{run_id}';")
    print(f"  SELECT * FROM {SCHEMA}.{RUNS_TABLE} ORDER BY run_time DESC LIMIT 5;")


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOP] Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
