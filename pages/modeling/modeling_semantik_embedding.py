# pages/data_preparation/text_processing/semantic_embedding_builder.py
# ============================================================
# Semantic Embedding Builder — Data Preparation (Pendekatan Semantik)
#
# Input:
# - lasis_djp.incident_semantic (kolom: text_semantic)
#
# Output:
# - lasis_djp.incident_semantic_embedding_runs  (metadata run)
# - lasis_djp.incident_semantic_embeddings      (embedding per tiket)
#
# Fitur:
# - Pilih backend embedding:
#   1) sentence-transformers (recommended)
#   2) transformers + torch (fallback)
# - Filter data (tanggal, modul) + opsi skip yang sudah ter-embed (per model)
# - Batch embedding + simpan ke Postgres (float8[])
# - Preview kandidat + statistik embedding
#
# Patch FINAL (clean):
# ✅ Perbaiki error SQLAlchemy "syntax error near :run_id::uuid" -> NO CAST di SQL
# ✅ params_json disimpan via psycopg2.extras.Json (aman jsonb)
# ✅ Hindari UnhashableParamError: arg Engine di @st.cache_data pakai _engine
# ✅ Upsert cepat pakai psycopg2 execute_values
# ✅ Guard dependency (ST / HF) + device selection aman
# ============================================================

from __future__ import annotations

import math
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from psycopg2.extras import Json, execute_values

# Optional deps:
try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except Exception:
    SentenceTransformer = None
    _HAS_ST = False

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    _HAS_HF = True
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModel = None
    _HAS_HF = False


# =========================
# 🔐 Guard login
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()


# =========================
# ⚙️ Konstanta
# =========================
SCHEMA_DEFAULT = "lasis_djp"

T_SEM = "incident_semantic"
T_RUNS = "incident_semantic_embedding_runs"
T_EMB = "incident_semantic_embeddings"

KEY_COL = "incident_number"
DATE_COL = "tgl_submit"
MODUL_COL = "modul"
SUBMODUL_COL = "sub_modul"
TEXT_COL = "text_semantic"


# =========================
# 🔌 Database Connection
# =========================
@st.cache_resource(show_spinner=False)
def get_engine() -> Engine:
    cfg = st.secrets["connections"]["postgres"]
    url = (
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}"
    )
    return create_engine(url, pool_pre_ping=True)


def ensure_tables(engine: Engine, schema: str) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.{T_RUNS} (
      run_id uuid PRIMARY KEY,
      run_time timestamptz NOT NULL DEFAULT now(),
      model_name text NOT NULL,
      approach text NOT NULL DEFAULT 'semantic_embedding',
      params_json jsonb NOT NULL,
      notes text
    );

    CREATE TABLE IF NOT EXISTS {schema}.{T_EMB} (
      run_id uuid NOT NULL,
      incident_number text NOT NULL,
      tgl_submit timestamptz,
      modul text,
      sub_modul text,
      model_name text NOT NULL,
      embedding float8[] NOT NULL,
      created_at timestamptz NOT NULL DEFAULT now(),
      PRIMARY KEY (run_id, incident_number)
    );

    CREATE INDEX IF NOT EXISTS idx_{T_EMB}_model ON {schema}.{T_EMB} (model_name);
    CREATE INDEX IF NOT EXISTS idx_{T_EMB}_tgl   ON {schema}.{T_EMB} (tgl_submit);
    CREATE INDEX IF NOT EXISTS idx_{T_EMB}_modul ON {schema}.{T_EMB} (modul);

    CREATE INDEX IF NOT EXISTS idx_{T_SEM}_tgl   ON {schema}.{T_SEM} (tgl_submit);
    CREATE INDEX IF NOT EXISTS idx_{T_SEM}_modul ON {schema}.{T_SEM} (modul);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


@st.cache_data(show_spinner=False, ttl=300)
def get_date_bounds(_engine: Engine, schema: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    q = f"""
    SELECT MIN({DATE_COL}) AS min_dt, MAX({DATE_COL}) AS max_dt
    FROM {schema}.{T_SEM}
    WHERE {DATE_COL} IS NOT NULL
    """
    df = pd.read_sql_query(text(q), _engine)
    if df.empty:
        return None, None
    return df.loc[0, "min_dt"], df.loc[0, "max_dt"]


@st.cache_data(show_spinner=False, ttl=300)
def get_distinct_modul(_engine: Engine, schema: str) -> List[str]:
    q = f"""
    SELECT DISTINCT {MODUL_COL} AS modul
    FROM {schema}.{T_SEM}
    WHERE {MODUL_COL} IS NOT NULL AND {MODUL_COL} <> ''
    ORDER BY {MODUL_COL}
    """
    df = pd.read_sql_query(text(q), _engine)
    return df["modul"].dropna().astype(str).tolist() if not df.empty else []


def _date_end_exclusive(dr: Optional[Tuple[Any, Any]]) -> Optional[str]:
    if not dr:
        return None
    return str(pd.to_datetime(dr[1]) + pd.Timedelta(days=1))[:10]


def build_candidates_query(
    schema: str,
    date_start: Optional[str],
    date_end_excl: Optional[str],
    modul_list: List[str],
    limit: int,
    skip_already_embedded: bool,
    model_name: str,
) -> Tuple[str, Dict[str, Any]]:
    where = [f"s.{TEXT_COL} IS NOT NULL", f"s.{TEXT_COL} <> ''", f"s.{KEY_COL} IS NOT NULL"]
    params: Dict[str, Any] = {}

    if date_start:
        where.append(f"s.{DATE_COL} >= :date_start")
        params["date_start"] = date_start
    if date_end_excl:
        where.append(f"s.{DATE_COL} < :date_end")
        params["date_end"] = date_end_excl
    if modul_list:
        where.append(f"s.{MODUL_COL} = ANY(:modul_list)")
        params["modul_list"] = modul_list

    join = ""
    if skip_already_embedded:
        join = f"""
        LEFT JOIN (
          SELECT DISTINCT incident_number
          FROM {schema}.{T_EMB}
          WHERE model_name = :model_name
        ) e ON e.incident_number = s.incident_number
        """
        where.append("e.incident_number IS NULL")
        params["model_name"] = model_name

    sql = f"""
    SELECT
      s.{KEY_COL}::text AS {KEY_COL},
      s.{DATE_COL} AS {DATE_COL},
      s.{MODUL_COL} AS {MODUL_COL},
      s.{SUBMODUL_COL} AS {SUBMODUL_COL},
      s.{TEXT_COL} AS {TEXT_COL}
    FROM {schema}.{T_SEM} s
    {join}
    WHERE {" AND ".join(where)}
    ORDER BY s.{DATE_COL} NULLS LAST
    LIMIT :lim
    """
    params["lim"] = int(limit)
    return sql, params


# =========================
# 🧠 Embedding backends
# =========================
def _pick_device() -> str:
    if _HAS_HF and torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


@st.cache_resource(show_spinner=False)
def load_st_model(model_name: str, device: str) -> Any:
    if not _HAS_ST:
        raise RuntimeError("sentence-transformers tidak tersedia. Install: pip install sentence-transformers")
    return SentenceTransformer(model_name, device=device)


@st.cache_resource(show_spinner=False)
def load_hf_model(model_name: str, device: str) -> Tuple[Any, Any]:
    if not _HAS_HF:
        raise RuntimeError("transformers/torch tidak tersedia. Install: pip install torch transformers")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    if device != "cpu" and torch is not None and torch.cuda.is_available():
        model.to(device)
    return tokenizer, model


def _mean_pooling(last_hidden_state: Any, attention_mask: Any) -> Any:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_texts(
    texts: List[str],
    backend: str,
    model_name: str,
    device: str,
    normalize_embeddings: bool,
    max_length: int,
) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)

    if backend == "sentence-transformers":
        model = load_st_model(model_name, device=device)
        emb = model.encode(
            texts,
            batch_size=min(64, len(texts)),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
        )
        return emb.astype(np.float32)

    # HF fallback
    tokenizer, model = load_hf_model(model_name, device=device)
    with torch.no_grad():
        tok = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        if device != "cpu" and torch.cuda.is_available():
            tok = {k: v.to(device) for k, v in tok.items()}

        out = model(**tok)
        pooled = _mean_pooling(out.last_hidden_state, tok["attention_mask"])
        if normalize_embeddings:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        emb = pooled.detach().cpu().numpy().astype(np.float32)
        return emb


# =========================
# 💾 DB write (Run + Upsert Embeddings)
# =========================
def insert_run(engine: Engine, schema: str, run_id: str, model_name: str, params: Dict[str, Any], notes: str) -> None:
    # IMPORTANT: jangan gunakan CAST :run_id::uuid / :params_json::jsonb di SQLAlchemy
    q = f"""
    INSERT INTO {schema}.{T_RUNS} (run_id, run_time, model_name, params_json, notes)
    VALUES (:run_id, now(), :model_name, :params_json, :notes)
    """
    with engine.begin() as conn:
        conn.execute(
            text(q),
            {
                "run_id": run_id,                 # string UUID OK
                "model_name": model_name,
                "params_json": Json(params),      # Json wrapper -> jsonb
                "notes": notes or "",
            },
        )


def upsert_embeddings(
    engine: Engine,
    schema: str,
    run_id: str,
    model_name: str,
    df: pd.DataFrame,
    emb: np.ndarray,
) -> int:
    if df.empty:
        return 0
    if len(df) != emb.shape[0]:
        raise ValueError("Panjang df dan embedding tidak sama.")

    created_at = datetime.now(timezone.utc)

    values = []
    df2 = df.reset_index(drop=True)
    for i in range(len(df2)):
        row = df2.iloc[i]
        vec = emb[i].astype(np.float64).tolist()
        values.append((
            run_id,
            str(row[KEY_COL]),
            row.get(DATE_COL),
            row.get(MODUL_COL),
            row.get(SUBMODUL_COL),
            model_name,
            vec,
            created_at,
        ))

    sql = f"""
    INSERT INTO {schema}.{T_EMB} (
      run_id, incident_number, tgl_submit, modul, sub_modul, model_name, embedding, created_at
    )
    VALUES %s
    ON CONFLICT (run_id, incident_number) DO UPDATE SET
      tgl_submit = EXCLUDED.tgl_submit,
      modul = EXCLUDED.modul,
      sub_modul = EXCLUDED.sub_modul,
      model_name = EXCLUDED.model_name,
      embedding = EXCLUDED.embedding,
      created_at = EXCLUDED.created_at
    """

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            execute_values(cur, sql, values, page_size=min(2000, max(500, len(values))))
        raw.commit()
    finally:
        raw.close()

    return len(values)


# =========================
# 🧭 UI
# =========================
st.title("🧬 Semantic Embedding Builder")
st.caption("Mengubah `text_semantic` menjadi embedding vektor (untuk cosine similarity, threshold, dan clustering semantik).")

engine = get_engine()

with st.sidebar:
    st.header("⚙️ Pengaturan")

    schema = st.text_input("Schema", value=SCHEMA_DEFAULT)
    ensure_tables(engine, schema)

    st.divider()
    st.subheader("Model Embedding")

    default_models_st = [
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "sentence-transformers/distiluse-base-multilingual-cased-v2",
    ]
    default_models_hf = [
        "indobenchmark/indobert-base-p1",
        "xlm-roberta-base",
    ]

    backend_ui = st.selectbox(
        "Backend",
        options=["sentence-transformers", "transformers (fallback)"],
        index=0 if _HAS_ST else 1,
        help="Disarankan: sentence-transformers. Jika belum ada, gunakan transformers fallback.",
    )

    backend = "sentence-transformers" if backend_ui == "sentence-transformers" else "transformers"

    if backend == "sentence-transformers":
        model_name = st.selectbox("Model", options=default_models_st, index=0)
        if not _HAS_ST:
            st.warning("sentence-transformers belum terdeteksi. Install: pip install sentence-transformers")
    else:
        model_name = st.selectbox("Model", options=default_models_hf, index=0)
        if not _HAS_HF:
            st.warning("transformers/torch belum terdeteksi. Install: pip install torch transformers")

    normalize_embeddings = st.checkbox(
        "Normalize embeddings (L2)",
        value=True,
        help="Jika ON, cosine similarity lebih stabil karena embedding dinormalisasi.",
    )

    max_length = st.slider(
        "Max token length (untuk backend transformers)",
        min_value=64,
        max_value=512,
        value=256,
        step=32,
        help="Hanya dipakai jika backend = transformers.",
    )

    auto_device = _pick_device()
    if auto_device == "cuda":
        device = st.selectbox("Device", options=["cuda", "cpu"], index=0)
    else:
        device = st.selectbox("Device", options=["cpu"], index=0, disabled=True)

    st.divider()
    st.subheader("Filter & Batch")

    min_dt, max_dt = get_date_bounds(engine, schema)
    if min_dt is None or max_dt is None:
        st.warning(f"Tidak dapat membaca batas tanggal dari `{schema}.{T_SEM}`.")
        date_range = None
    else:
        dr = st.date_input("Rentang tanggal (tgl_submit)", value=(min_dt.date(), max_dt.date()))
        date_range = dr if isinstance(dr, tuple) and len(dr) == 2 else None

    modul_options = get_distinct_modul(engine, schema)
    selected_modul = st.multiselect("Filter modul", options=modul_options, default=[])

    limit_rows = st.number_input(
        "Limit kandidat (boleh bertahap untuk data besar)",
        min_value=100,
        max_value=500000,
        value=50000,
        step=1000,
    )

    batch_rows = st.number_input(
        "Batch size embedding",
        min_value=16,
        max_value=2048,
        value=256,
        step=16,
        help="Jika CPU, 128–512 biasanya aman. Jika GPU, bisa lebih besar.",
    )

    skip_already = st.checkbox(
        "Skip yang sudah ter-embed (berdasarkan model_name)",
        value=True,
        help="Mencegah embedding ulang untuk incident_number yang sudah ada untuk model ini.",
    )

    notes = st.text_input("Catatan run (opsional)", value="")

    st.divider()
    colA, colB = st.columns(2)
    do_preview = colA.button("👀 Preview kandidat", use_container_width=True)
    do_run = colB.button("🚀 Run embedding", type="primary", use_container_width=True)


def fetch_candidates() -> pd.DataFrame:
    sql, params = build_candidates_query(
        schema=schema,
        date_start=(str(date_range[0]) if date_range else None),
        date_end_excl=_date_end_exclusive(date_range),
        modul_list=selected_modul,
        limit=int(limit_rows),
        skip_already_embedded=bool(skip_already),
        model_name=model_name,
    )
    return pd.read_sql_query(text(sql), engine, params=params)


# =========================
# Preview
# =========================
if do_preview:
    try:
        df = fetch_candidates()
        if df.empty:
            st.info("Tidak ada kandidat (cek filter/opsi skip).")
        else:
            st.subheader("Preview Kandidat")
            st.dataframe(df.head(50), use_container_width=True, height=420)
            c1, c2, c3 = st.columns(3)
            c1.metric("Kandidat (limit)", f"{len(df):,}")
            c2.metric("Model", model_name)
            c3.metric("Backend", backend_ui)
    except Exception as e:
        st.exception(e)


# =========================
# Run Embedding
# =========================
if do_run:
    try:
        if backend == "sentence-transformers" and not _HAS_ST:
            st.error("Backend sentence-transformers dipilih, tapi package belum tersedia.")
            st.stop()
        if backend == "transformers" and not _HAS_HF:
            st.error("Backend transformers dipilih, tapi torch/transformers belum tersedia.")
            st.stop()

        df = fetch_candidates()
        if df.empty:
            st.warning("Tidak ada kandidat untuk di-embed (cek filter/opsi skip).")
            st.stop()

        run_id = str(uuid.uuid4())
        params_run = {
            "schema": schema,
            "source_table": f"{schema}.{T_SEM}",
            "model_name": model_name,
            "backend": backend,
            "normalize_embeddings": bool(normalize_embeddings),
            "device": device,
            "max_length": int(max_length),
            "filters": {
                "date_start": (str(date_range[0]) if date_range else None),
                "date_end_exclusive": _date_end_exclusive(date_range),
                "modul": selected_modul,
                "skip_already": bool(skip_already),
                "limit_rows": int(limit_rows),
            },
            "batch_rows": int(batch_rows),
        }

        insert_run(engine, schema, run_id, model_name, params_run, notes)

        st.info(f"Mulai embedding: {len(df):,} baris (run_id={run_id})")

        n = len(df)
        n_batches = math.ceil(n / int(batch_rows))
        progress = st.progress(0, text="Menyiapkan…")
        status = st.empty()

        t0 = time.time()
        written = 0

        last_emb: Optional[np.ndarray] = None

        for bi in range(n_batches):
            start = bi * int(batch_rows)
            end = min(n, (bi + 1) * int(batch_rows))

            dfb = df.iloc[start:end].copy()
            texts = dfb[TEXT_COL].astype(str).tolist()

            emb = embed_texts(
                texts=texts,
                backend=backend,
                model_name=model_name,
                device=device,
                normalize_embeddings=bool(normalize_embeddings),
                max_length=int(max_length),
            )
            last_emb = emb

            written += upsert_embeddings(engine, schema, run_id, model_name, dfb, emb)

            pct = int(((bi + 1) / n_batches) * 100)
            elapsed = time.time() - t0
            rate = (end / elapsed) if elapsed > 0 else 0.0

            progress.progress(pct, text=f"Batch {bi+1}/{n_batches} — {end:,}/{n:,} ({pct}%)")
            status.write(f"✅ Written: **{written:,}** | Rate: **{rate:,.0f} rows/s** | Elapsed: **{elapsed:,.1f}s**")

        st.success(f"Selesai embedding. Total tersimpan: {written:,} (run_id={run_id})")

        if last_emb is not None and last_emb.size > 0:
            with st.expander("📊 Statistik embedding (sample batch terakhir)", expanded=True):
                norms = np.linalg.norm(last_emb, axis=1)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Dimensi", f"{last_emb.shape[1]:,}")
                c2.metric("Norm min", f"{float(norms.min()):.4f}")
                c3.metric("Norm mean", f"{float(norms.mean()):.4f}")
                c4.metric("Norm max", f"{float(norms.max()):.4f}")

        with st.expander("🔎 Sampel hasil tersimpan (top 50)", expanded=False):
            q = f"""
            SELECT run_id, incident_number, tgl_submit, modul, sub_modul, model_name,
                   array_length(embedding, 1) AS dim, created_at
            FROM {schema}.{T_EMB}
            WHERE run_id = :run_id
            ORDER BY created_at DESC
            LIMIT 50
            """
            df_out = pd.read_sql_query(text(q), engine, params={"run_id": run_id})
            st.dataframe(df_out, use_container_width=True, height=420)

        st.divider()
        st.markdown(
            "✅ **Langkah berikutnya:** hitung **cosine similarity** antar embedding (kNN/radius graph), "
            "lalu lakukan **threshold grid** atau **HDBSCAN** untuk clustering semantik."
        )

    except Exception as e:
        st.exception(e)


with st.expander("📌 Catatan", expanded=False):
    st.markdown(
        """
- Embedding adalah representasi numerik makna; setelah ini baru dihitung **cosine similarity**.
- Opsi **Normalize embeddings (L2)** direkomendasikan untuk stabilitas cosine.
- Jika `sentence-transformers` tersedia, itu yang paling praktis untuk sentence embedding.
        """.strip()
    )
