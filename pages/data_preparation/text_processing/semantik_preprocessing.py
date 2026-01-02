# pages/data_preparation/text_processing/semantic_preprocessing.py
# ============================================================
# Semantic Text Preprocessing — Data Preparation (Pendekatan Semantik)
#
# Tujuan:
# - Menyiapkan teks "text_semantic" yang siap dipakai untuk embedding (IndoBERT/SBERT).
# - Preprocessing dibuat "ringan" (tanpa stemming/stopword agresif) agar makna tetap terjaga.
#
# Sumber (default):
# - lasis_djp.incident_normalized (kolom text_norm)
#
# Output:
# - lasis_djp.incident_semantic
#
# Fitur UI:
# - Filter rentang tanggal + modul + limit preview
# - Opsi: lowercasing, normalize whitespace, hapus stop-phrases template
# - Preview before/after
# - Proses batch + upsert ke DB (ON CONFLICT)
#
# Patch FINAL:
# ✅ Perbaiki UnhashableParamError Streamlit cache (Engine) dengan _engine pada @st.cache_data
# ✅ Guard jika kolom opsional (site/assignee/sub_modul) tidak ada -> set NULL aman
# ✅ Query count & preview aman + parameterized
# ✅ Progress lebih informatif
# ============================================================

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from psycopg2.extras import Json, execute_values


# =========================
# 🔐 Guard login
# =========================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu untuk mengakses halaman ini.")
    st.stop()


# =========================
# ⚙️ Konstanta
# =========================
DEFAULT_SCHEMA = "lasis_djp"

# default sumber
DEFAULT_SRC_TABLE = "incident_normalized"
DEFAULT_SRC_TEXT_COL = "text_norm"

# kolom yang lazim pada dataset insiden
KEY_COL = "incident_number"
DATE_COL = "tgl_submit"
MODUL_COL = "modul"
SUBMODUL_COL = "sub_modul"
SITE_COL = "site"
ASSIGNEE_COL = "assignee"

# output
OUT_TABLE = "incident_semantic"


# =========================
# 🧹 Preprocessing (Semantik - ringan)
# =========================
RE_WS = re.compile(r"\s+")
RE_CTRL = re.compile(r"[\x00-\x1F\x7F]")


def semantic_clean(
    text_in: Any,
    lowercase: bool = True,
    normalize_whitespace: bool = True,
) -> str:
    if text_in is None:
        return ""
    t = str(text_in)

    # remove control chars
    t = RE_CTRL.sub(" ", t)

    # normalize nbsp
    t = t.replace("\u00a0", " ")

    if normalize_whitespace:
        t = RE_WS.sub(" ", t).strip()
    else:
        t = t.strip()

    if lowercase:
        t = t.lower()

    return t


def apply_stop_phrases(text_in: str, phrases: List[str]) -> str:
    """Hapus frasa template/noise yang sangat umum (opsional).
    Tips: isi hanya frasa yang benar-benar template, jangan agresif.
    """
    if not phrases:
        return text_in

    t = text_in
    for p in phrases:
        p = (p or "").strip().lower()
        if not p:
            continue
        t = t.replace(p, " ")

    t = RE_WS.sub(" ", t).strip()
    return t


def build_text_semantic(
    raw_text: Any,
    lowercase: bool,
    normalize_whitespace: bool,
    stop_phrases: List[str],
) -> str:
    t = semantic_clean(raw_text, lowercase=lowercase, normalize_whitespace=normalize_whitespace)
    t = apply_stop_phrases(t, stop_phrases)
    return t


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


def ensure_output_table(engine: Engine, schema: str) -> None:
    ddl = f"""
    CREATE SCHEMA IF NOT EXISTS {schema};

    CREATE TABLE IF NOT EXISTS {schema}.{OUT_TABLE} (
      incident_number text PRIMARY KEY,
      tgl_submit timestamptz,
      site text,
      assignee text,
      modul text,
      sub_modul text,
      text_semantic text NOT NULL,
      tokens_semantic_json jsonb,
      preprocess_params jsonb NOT NULL,
      preprocess_time timestamptz NOT NULL DEFAULT now()
    );

    CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_modul
      ON {schema}.{OUT_TABLE} (modul);

    CREATE INDEX IF NOT EXISTS idx_{OUT_TABLE}_tgl
      ON {schema}.{OUT_TABLE} (tgl_submit);
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))


@st.cache_data(show_spinner=False, ttl=300)
def get_date_bounds(_engine: Engine, src_schema: str, src_table: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    q = f"""
    SELECT
      MIN({DATE_COL}) AS min_dt,
      MAX({DATE_COL}) AS max_dt
    FROM {src_schema}.{src_table}
    WHERE {DATE_COL} IS NOT NULL
    """
    df = pd.read_sql_query(text(q), _engine)
    if df.empty:
        return None, None
    return df.loc[0, "min_dt"], df.loc[0, "max_dt"]


@st.cache_data(show_spinner=False, ttl=300)
def get_distinct_modul(_engine: Engine, src_schema: str, src_table: str) -> List[str]:
    q = f"""
    SELECT DISTINCT {MODUL_COL} AS modul
    FROM {src_schema}.{src_table}
    WHERE {MODUL_COL} IS NOT NULL AND {MODUL_COL} <> ''
    ORDER BY {MODUL_COL}
    """
    df = pd.read_sql_query(text(q), _engine)
    if df.empty or "modul" not in df.columns:
        return []
    return df["modul"].dropna().astype(str).tolist()


@st.cache_data(show_spinner=False, ttl=300)
def get_table_columns(_engine: Engine, schema: str, table: str) -> List[str]:
    q = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = :schema AND table_name = :table
    ORDER BY ordinal_position
    """
    df = pd.read_sql_query(text(q), _engine, params={"schema": schema, "table": table})
    return df["column_name"].astype(str).tolist() if not df.empty else []


def build_source_query(
    src_schema: str,
    src_table: str,
    src_text_col: str,
    date_start: Optional[str],
    date_end_exclusive: Optional[str],
    modul_list: List[str],
    available_cols: List[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Query sumber untuk proses batch.
    - date_end_exclusive: YYYY-MM-DD (eksklusif)
    - Guard kolom opsional: site/assignee/sub_modul
    """
    def sel(col: str) -> str:
        return col if col in available_cols else "NULL"

    where = ["1=1", f"{KEY_COL} IS NOT NULL", f"{src_text_col} IS NOT NULL"]
    params: Dict[str, Any] = {}

    if date_start:
        where.append(f"{DATE_COL} >= :date_start")
        params["date_start"] = date_start
    if date_end_exclusive:
        where.append(f"{DATE_COL} < :date_end")
        params["date_end"] = date_end_exclusive
    if modul_list:
        where.append(f"{MODUL_COL} = ANY(:modul_list)")
        params["modul_list"] = modul_list

    sql = f"""
    SELECT
      {KEY_COL}::text AS {KEY_COL},
      {DATE_COL} AS {DATE_COL},
      {sel(SITE_COL)} AS {SITE_COL},
      {sel(ASSIGNEE_COL)} AS {ASSIGNEE_COL},
      {MODUL_COL} AS {MODUL_COL},
      {sel(SUBMODUL_COL)} AS {SUBMODUL_COL},
      {src_text_col} AS raw_text
    FROM {src_schema}.{src_table}
    WHERE {" AND ".join(where)}
    ORDER BY {DATE_COL} NULLS LAST
    """
    return sql, params


def stream_source_chunks(
    engine: Engine,
    sql: str,
    params: Dict[str, Any],
    chunksize: int,
) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_sql_query(text(sql), engine, params=params, chunksize=chunksize):
        yield chunk


def upsert_chunk(engine: Engine, schema: str, rows: List[Dict[str, Any]]) -> int:
    if not rows:
        return 0

    values = []
    for r in rows:
        values.append((
            r.get("incident_number"),
            r.get("tgl_submit"),
            r.get("site"),
            r.get("assignee"),
            r.get("modul"),
            r.get("sub_modul"),
            r.get("text_semantic"),
            r.get("tokens_semantic_json"),
            r.get("preprocess_params"),   # Json wrapper
            r.get("preprocess_time"),
        ))

    sql = f"""
    INSERT INTO {schema}.{OUT_TABLE} (
      incident_number, tgl_submit, site, assignee, modul, sub_modul,
      text_semantic, tokens_semantic_json, preprocess_params, preprocess_time
    )
    VALUES %s
    ON CONFLICT (incident_number) DO UPDATE SET
      tgl_submit = EXCLUDED.tgl_submit,
      site = EXCLUDED.site,
      assignee = EXCLUDED.assignee,
      modul = EXCLUDED.modul,
      sub_modul = EXCLUDED.sub_modul,
      text_semantic = EXCLUDED.text_semantic,
      tokens_semantic_json = EXCLUDED.tokens_semantic_json,
      preprocess_params = EXCLUDED.preprocess_params,
      preprocess_time = EXCLUDED.preprocess_time
    """

    raw = engine.raw_connection()
    try:
        with raw.cursor() as cur:
            execute_values(cur, sql, values, page_size=min(5000, max(1000, len(values))))
        raw.commit()
    finally:
        raw.close()

    return len(values)


# =========================
# 🧭 UI
# =========================
st.title("🧠 Semantic Text Preprocessing")
st.caption(
    "Menyiapkan `text_semantic` (preprocessing ringan) untuk embedding semantik (IndoBERT/SBERT). "
    "Tanpa stemming/stopword agresif agar makna tidak hilang."
)

engine = get_engine()

# sidebar settings
with st.sidebar:
    st.header("⚙️ Pengaturan")

    out_schema = st.text_input("Schema output", value=DEFAULT_SCHEMA)

    src_schema = st.text_input("Schema sumber", value=DEFAULT_SCHEMA)
    src_table = st.text_input("Tabel sumber", value=DEFAULT_SRC_TABLE)
    src_text_col = st.text_input("Kolom teks sumber", value=DEFAULT_SRC_TEXT_COL)

    st.divider()
    st.subheader("Opsi Preprocessing (Semantik)")
    lowercase = st.checkbox("Lowercase", value=True)
    normalize_whitespace = st.checkbox("Normalize whitespace", value=True)

    stop_phrases_txt = st.text_area(
        "Stop-phrases (1 baris 1 frasa) — opsional",
        value="mohon solusinya\nterima kasih\nmohon bantuan",
        height=120,
        help="Isi hanya frasa template/noise yang sangat umum. Jangan agresif.",
    )
    stop_phrases = [x.strip() for x in stop_phrases_txt.splitlines() if x.strip()]

    st.divider()
    st.subheader("Filter Data")

    # load date bounds
    try:
        min_dt, max_dt = get_date_bounds(engine, src_schema, src_table)
    except Exception:
        min_dt, max_dt = None, None

    if min_dt is None or max_dt is None:
        st.warning("Tidak dapat membaca batas tanggal dari sumber. Pastikan kolom `tgl_submit` ada.")
        date_range = None
    else:
        dr = st.date_input(
            "Rentang tanggal (tgl_submit)",
            value=(min_dt.date(), max_dt.date()),
        )
        date_range = dr if isinstance(dr, tuple) and len(dr) == 2 else None

    modul_options = []
    try:
        modul_options = get_distinct_modul(engine, src_schema, src_table)
    except Exception:
        modul_options = []
    selected_modul = st.multiselect("Filter modul", options=modul_options, default=[])

    st.divider()
    st.subheader("Batch")
    preview_limit = st.number_input("Limit preview", min_value=10, max_value=1000, value=50, step=10)
    chunksize = st.number_input("Chunksize proses", min_value=500, max_value=20000, value=5000, step=500)

    st.divider()
    colA, colB = st.columns(2)
    do_preview = colA.button("👀 Preview", use_container_width=True)
    do_run = colB.button("🚀 Run Preprocessing", type="primary", use_container_width=True)

# ensure output table
ensure_output_table(engine, out_schema)

# check columns from source (for guarding optional cols)
try:
    src_cols = get_table_columns(engine, src_schema, src_table)
except Exception:
    src_cols = []


def _date_end_exclusive_from_range(dr: Optional[Tuple[Any, Any]]) -> Optional[str]:
    if not dr:
        return None
    # dr[1] adalah date; +1 hari agar eksklusif
    return str(pd.to_datetime(dr[1]) + pd.Timedelta(days=1))[:10]


# =========================
# 🔎 Preview
# =========================
def preview_data() -> None:
    sql, params = build_source_query(
        src_schema=src_schema,
        src_table=src_table,
        src_text_col=src_text_col,
        date_start=(str(date_range[0]) if date_range else None),
        date_end_exclusive=_date_end_exclusive_from_range(date_range),
        modul_list=selected_modul,
        available_cols=src_cols,
    )

    sql_prev = f"SELECT * FROM ({sql}) s LIMIT :lim"
    params_prev = dict(params)
    params_prev["lim"] = int(preview_limit)

    df = pd.read_sql_query(text(sql_prev), engine, params=params_prev)
    if df.empty:
        st.info("Tidak ada data untuk preview (cek filter / tabel sumber).")
        return

    df["text_semantic"] = df["raw_text"].apply(
        lambda x: build_text_semantic(
            x,
            lowercase=lowercase,
            normalize_whitespace=normalize_whitespace,
            stop_phrases=stop_phrases,
        )
    )

    st.subheader("Preview Before → After")
    show_cols = [KEY_COL, DATE_COL, MODUL_COL, "raw_text", "text_semantic"]
    show = df[show_cols].copy()

    st.dataframe(show, use_container_width=True, height=420)

    n_empty_after = (show["text_semantic"].astype(str).str.len() == 0).sum()
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows preview", f"{len(show):,}")
    c2.metric("Empty after", f"{n_empty_after:,}")
    c3.metric("Stop-phrases", f"{len(stop_phrases):,}")


# =========================
# 🚀 Run Preprocessing (Batch + Upsert)
# =========================
def run_preprocessing() -> None:
    sql, params = build_source_query(
        src_schema=src_schema,
        src_table=src_table,
        src_text_col=src_text_col,
        date_start=(str(date_range[0]) if date_range else None),
        date_end_exclusive=_date_end_exclusive_from_range(date_range),
        modul_list=selected_modul,
        available_cols=src_cols,
    )

    # hitung total untuk progress
    count_sql = f"SELECT COUNT(*) AS n FROM ({sql}) s"
    dfc = pd.read_sql_query(text(count_sql), engine, params=params)
    total = int(dfc.loc[0, "n"]) if not dfc.empty else 0

    if total == 0:
        st.warning("Tidak ada data yang diproses (hasil filter kosong).")
        return

    job_id = str(uuid.uuid4())
    preprocess_params = {
        "approach": "semantic_preprocessing_light",
        "job_id": job_id,
        "source": {"schema": src_schema, "table": src_table, "text_col": src_text_col},
        "filters": {
            "date_start": (str(date_range[0]) if date_range else None),
            "date_end_exclusive": _date_end_exclusive_from_range(date_range),
            "modul": selected_modul,
        },
        "options": {
            "lowercase": lowercase,
            "normalize_whitespace": normalize_whitespace,
            "stop_phrases": stop_phrases,
        },
    }

    st.info(f"Mulai preprocessing semantik untuk {total:,} baris… (job_id={job_id})")

    progress = st.progress(0, text="Menyiapkan batch…")
    status = st.empty()

    processed = 0
    upserted = 0
    t0 = time.time()
    now_ts = datetime.now(timezone.utc)

    for i, chunk in enumerate(stream_source_chunks(engine, sql, params, chunksize=int(chunksize)), start=1):
        if chunk.empty:
            continue

        chunk["text_semantic"] = chunk["raw_text"].apply(
            lambda x: build_text_semantic(
                x,
                lowercase=lowercase,
                normalize_whitespace=normalize_whitespace,
                stop_phrases=stop_phrases,
            )
        )

        rows: List[Dict[str, Any]] = []
        for _, r in chunk.iterrows():
            rows.append({
                "incident_number": (r.get(KEY_COL) if pd.notna(r.get(KEY_COL)) else None),
                "tgl_submit": r.get(DATE_COL),
                "site": r.get(SITE_COL),
                "assignee": r.get(ASSIGNEE_COL),
                "modul": r.get(MODUL_COL),
                "sub_modul": r.get(SUBMODUL_COL),
                "text_semantic": r.get("text_semantic") or "",
                "tokens_semantic_json": None,
                "preprocess_params": Json(preprocess_params),
                "preprocess_time": now_ts,
            })

        n_up = upsert_chunk(engine, out_schema, rows)
        upserted += n_up
        processed += len(chunk)

        pct = min(1.0, processed / max(1, total))
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0.0

        progress.progress(int(pct * 100), text=f"Batch {i} — {processed:,}/{total:,} ({pct*100:.1f}%)")
        status.write(f"✅ Upserted: **{upserted:,}** | Rate: **{rate:,.0f} rows/s** | Elapsed: **{elapsed:,.1f}s**")

    st.success(f"Selesai. Total diproses: {processed:,} | Total upserted: {upserted:,}")
    st.caption(f"Output: `{out_schema}.{OUT_TABLE}` | job_id: {job_id}")

    with st.expander("Lihat sampel hasil terbaru (top 50)", expanded=True):
        q = f"""
        SELECT incident_number, tgl_submit, modul, sub_modul, text_semantic, preprocess_time
        FROM {out_schema}.{OUT_TABLE}
        ORDER BY preprocess_time DESC
        LIMIT 50
        """
        df_out = pd.read_sql_query(text(q), engine)
        st.dataframe(df_out, use_container_width=True, height=420)


# =========================
# Execute actions
# =========================
if do_preview:
    try:
        preview_data()
    except Exception as e:
        st.exception(e)

if do_run:
    try:
        run_preprocessing()
    except Exception as e:
        st.exception(e)


with st.expander("📌 Catatan Metodologis", expanded=False):
    st.markdown(
        """
- Preprocessing semantik dibuat **ringan** agar representasi makna tidak hilang.
- **Stop-phrases** dipakai hanya untuk frasa template/noise yang terlalu sering muncul.
- Parameter proses disimpan dalam `preprocess_params` agar proses **reproducible**.
        """.strip()
    )
