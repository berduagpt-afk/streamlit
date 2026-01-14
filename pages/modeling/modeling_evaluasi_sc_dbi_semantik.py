# pages/evaluation_semantik_runs_to_evaltable.py
# ============================================================
# Simpan metrik evaluasi SEMANTIK (Silhouette & DBI) dari
# lasis_djp.modeling_semantik_hdbscan_runs ke tabel gabungan.
#
# ✅ FINAL (copy-paste)
# - Tabel output: lasis_djp.modeling_evaluation_results
# - DBCV dihitung di halaman lain → di sini tidak dihitung & tidak disimpan (NULL)
# - Perilaku simpan:
#   - Jika sudah ada record untuk (jenis_pendekatan='semantik' AND modeling_id sama) → UPDATE (tidak insert baris baru)
#   - Jika belum ada → INSERT 1 baris baru
# ============================================================

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text


# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ======================================================
# ⚙️ KONSTANTA DB
# ======================================================
SCHEMA = "lasis_djp"

T_RUNS_SEM = "modeling_semantik_hdbscan_runs"
T_EVAL = "modeling_evaluation_results"


# ======================================================
# 🔌 DB CONNECTION
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
        future=True,
    )


engine = get_engine()


# ======================================================
# 🧰 Helpers
# ======================================================
def _safe_json(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        try:
            obj = json.loads(x)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    return {}


def _to_text(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _to_float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        if pd.isna(v):
            return None
        return v
    except Exception:
        return None


# ======================================================
# ✅ Ensure output table (dbcv column exists; but not used here)
# ======================================================
def ensure_eval_table(_engine) -> None:
    ddl = f"""
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    CREATE TABLE IF NOT EXISTS {SCHEMA}.{T_EVAL}
    (
        eval_id uuid NOT NULL DEFAULT gen_random_uuid(),
        run_time timestamptz NOT NULL DEFAULT now(),

        jenis_pendekatan text NOT NULL,
        job_id uuid,
        modeling_id uuid,
        embedding_run_id uuid,
        temporal_id text,

        silhouette_score double precision,
        dbi double precision,

        threshold double precision,
        notes text,
        meta_json jsonb,

        CONSTRAINT {T_EVAL}_pkey PRIMARY KEY (eval_id)
    );

    -- Pastikan kolom DBCV ada (walau halaman ini tidak mengisinya)
    ALTER TABLE {SCHEMA}.{T_EVAL}
        ADD COLUMN IF NOT EXISTS dbcv double precision;

    -- Index lama tetap boleh ada (tidak mengganggu)
    CREATE UNIQUE INDEX IF NOT EXISTS uq_{T_EVAL}_core
    ON {SCHEMA}.{T_EVAL} (
        jenis_pendekatan,
        modeling_id,
        embedding_run_id,
        temporal_id,
        threshold
    );

    CREATE INDEX IF NOT EXISTS idx_{T_EVAL}_main
    ON {SCHEMA}.{T_EVAL} (jenis_pendekatan, modeling_id, run_time DESC);
    """
    statements = [s.strip() for s in ddl.split(";") if s.strip()]
    with _engine.begin() as conn:
        for stmt in statements:
            conn.exec_driver_sql(stmt)


# ======================================================
# 📥 Load runs semantik
# ======================================================
@st.cache_data(show_spinner=False, ttl=120)
def load_runs_semantik(_engine, limit: int = 300) -> pd.DataFrame:
    q = text(
        f"""
        SELECT
            modeling_id::text AS modeling_id,
            run_time,
            embedding_run_id::text AS embedding_run_id,
            n_rows,
            n_clusters,
            n_noise,
            silhouette,
            dbi,
            params_json,
            notes
        FROM {SCHEMA}.{T_RUNS_SEM}
        ORDER BY run_time DESC
        LIMIT :lim
        """
    )
    with _engine.begin() as conn:
        return pd.read_sql(q, conn, params={"lim": int(limit)})


# ======================================================
# 💾 Save (UPDATE by modeling_id first; INSERT if not exists)
# ======================================================
def save_eval_semantik_update_by_modeling_id(
    _engine,
    row: pd.Series,
    temporal_id: str = "",
) -> str:
    """
    Perilaku:
    1) UPDATE record evaluasi semantik berdasarkan modeling_id (tanpa membuat baris baru)
    2) Kalau tidak ada record untuk modeling_id tsb → INSERT 1 baris baru

    Return: "updated" atau "inserted"
    """
    pj = _safe_json(row.get("params_json"))
    notes = _to_text(row.get("notes"))

    modeling_id_text = _to_text(row.get("modeling_id")).strip()
    embedding_run_id_text = _to_text(row.get("embedding_run_id")).strip()

    payload = {
        "jenis_pendekatan": "semantik",
        "modeling_id": modeling_id_text,
        "embedding_run_id": embedding_run_id_text if embedding_run_id_text else None,
        "temporal_id": temporal_id.strip() if temporal_id.strip() else None,
        "silhouette_score": _to_float_or_none(row.get("silhouette")),
        "dbi": _to_float_or_none(row.get("dbi")),
        "dbcv": None,             # ❌ tidak dihitung di halaman ini
        "threshold": -1.0,        # sentinel untuk semantik (kalau butuh konsistensi)
        "notes": notes,
        "meta_json": json.dumps(
            {
                "source": f"{SCHEMA}.{T_RUNS_SEM}",
                "run_time": _to_text(row.get("run_time")),
                "n_rows": row.get("n_rows"),
                "n_clusters": row.get("n_clusters"),
                "n_noise": row.get("n_noise"),
                "params_json": pj,
                "dbcv": None,
                "dbcv_note": "DBCV dihitung di halaman lain",
            },
            ensure_ascii=False,
        ),
    }

    sql_update = text(
        f"""
        UPDATE {SCHEMA}.{T_EVAL} t
        SET
            run_time = now(),
            embedding_run_id = CASE
                WHEN :embedding_run_id IS NULL OR :embedding_run_id = '' THEN t.embedding_run_id
                ELSE CAST(:embedding_run_id AS uuid)
            END,
            temporal_id = COALESCE(:temporal_id, t.temporal_id),
            threshold = COALESCE(:threshold, t.threshold),

            silhouette_score = :silhouette_score,
            dbi = :dbi,

            -- DBCV tidak diubah di halaman ini (biar tidak menimpa hasil halaman DBCV)
            -- dbcv = t.dbcv,

            notes = :notes,
            meta_json = CAST(:meta_json AS jsonb)
        WHERE t.eval_id = (
            SELECT e.eval_id
            FROM {SCHEMA}.{T_EVAL} e
            WHERE e.jenis_pendekatan = 'semantik'
              AND e.modeling_id = CAST(:modeling_id AS uuid)
            ORDER BY e.run_time DESC
            LIMIT 1
        )
        """
    )

    sql_insert = text(
        f"""
        INSERT INTO {SCHEMA}.{T_EVAL}
        (
            jenis_pendekatan, job_id, modeling_id, embedding_run_id, temporal_id,
            silhouette_score, dbi, dbcv, threshold, notes, meta_json
        )
        VALUES
        (
            :jenis_pendekatan,
            NULL,
            CAST(:modeling_id AS uuid),
            CASE WHEN :embedding_run_id IS NULL OR :embedding_run_id = '' THEN NULL ELSE CAST(:embedding_run_id AS uuid) END,
            :temporal_id,
            :silhouette_score,
            :dbi,
            NULL,
            :threshold,
            :notes,
            CAST(:meta_json AS jsonb)
        )
        """
    )

    with _engine.begin() as conn:
        res = conn.execute(sql_update, payload)
        if int(getattr(res, "rowcount", 0) or 0) > 0:
            return "updated"

        # belum ada record untuk modeling_id → INSERT
        conn.execute(sql_insert, payload)
        return "inserted"


# ======================================================
# 🧭 UI
# ======================================================
st.title("🧾 Simpan Evaluasi Semantik → Tabel Gabungan")
st.caption(
    "Mengambil **Silhouette Score** dan **DBI** dari tabel runs semantik (HDBSCAN) "
    "lalu menyimpannya ke tabel evaluasi gabungan. "
    "**DBCV tidak dihitung di halaman ini** (ada halaman khusus DBCV)."
)

ensure_eval_table(engine)

df_runs = load_runs_semantik(engine, limit=300)
if df_runs.empty:
    st.warning(f"Tidak ada data pada {SCHEMA}.{T_RUNS_SEM}.")
    st.stop()

with st.sidebar:
    st.header("⚙️ Pengaturan")
    limit_show = st.number_input("Limit tampil", min_value=50, max_value=1000, value=300, step=50)
    temporal_id = st.text_input(
        "temporal_id (opsional)",
        value="",
        help="Isi jika evaluasi ini ingin ditautkan ke hasil evaluasi temporal tertentu.",
    )

df_runs = df_runs.head(int(limit_show)).copy()

st.subheader("Daftar Run Semantik (Sumber)")
st.dataframe(
    df_runs[
        [
            "modeling_id",
            "run_time",
            "embedding_run_id",
            "n_rows",
            "n_clusters",
            "n_noise",
            "silhouette",
            "dbi",
        ]
    ],
    use_container_width=True,
)

st.divider()

col1, col2 = st.columns([1, 2])

with col1:
    if st.button("💾 Simpan/Update Semua Run yang Ditampilkan", type="primary"):
        ok_upd, ok_ins, fail = 0, 0, 0
        errors = []
        for _, r in df_runs.iterrows():
            try:
                action = save_eval_semantik_update_by_modeling_id(engine, r, temporal_id=temporal_id)
                if action == "updated":
                    ok_upd += 1
                else:
                    ok_ins += 1
            except Exception as e:
                fail += 1
                errors.append(f"{r.get('modeling_id')} / {r.get('embedding_run_id')}: {e}")

        st.success(f"Selesai. Update: {ok_upd}, Insert baru: {ok_ins}, Gagal: {fail}")
        if errors:
            with st.expander("Detail error"):
                st.write("\n".join(errors))

with col2:
    st.info(
        "Catatan:\n"
        "- Mekanisme simpan: **UPDATE dulu berdasarkan (jenis_pendekatan='semantik', modeling_id)**.\n"
        "- Jika belum ada record untuk modeling_id tsb → **INSERT 1 baris**.\n"
        "- Kolom **DBCV tidak disentuh** di halaman ini agar tidak menimpa hasil dari halaman DBCV.\n"
    )

st.divider()

st.subheader("Preview Data di Tabel Evaluasi Gabungan (semantik terbaru)")
q_prev = text(
    f"""
    SELECT
        run_time,
        jenis_pendekatan,
        job_id::text AS job_id,
        modeling_id::text AS modeling_id,
        embedding_run_id::text AS embedding_run_id,
        temporal_id,
        silhouette_score,
        dbi,
        dbcv,
        threshold,
        notes
    FROM {SCHEMA}.{T_EVAL}
    WHERE jenis_pendekatan = 'semantik'
    ORDER BY run_time DESC
    LIMIT 200
    """
)
with engine.begin() as conn:
    df_prev = pd.read_sql(q_prev, conn)

st.dataframe(df_prev, use_container_width=True)
