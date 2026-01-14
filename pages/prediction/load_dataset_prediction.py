# pages/modeling/dataset_supervised_builder.py
from __future__ import annotations

import json
import uuid
import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy import create_engine, text

# ======================================================
# 🔐 Guard Login (opsional - sesuaikan dengan sistem Anda)
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()

SCHEMA = "lasis_djp"
T_LABEL = "incident_labeling_results"
T_OUT = "dataset_supervised"

# ======================================================
# 🔌 DB Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}@{cfg['host']}:{cfg.get('port', 5432)}/{cfg['database']}",
        pool_pre_ping=True,
    )

engine = get_engine()

# ======================================================
# 🧠 Session State init (KRUSIAL agar df tidak hilang saat rerun)
# ======================================================
if "ds_df" not in st.session_state:
    st.session_state.ds_df = None
if "ds_builder_json" not in st.session_state:
    st.session_state.ds_builder_json = None
if "ds_signature" not in st.session_state:
    st.session_state.ds_signature = None  # untuk memastikan df cocok dengan pilihan saat ini

# ======================================================
# 🧱 Helpers
# ======================================================
def qdf(sql: str, params: dict | None = None) -> pd.DataFrame:
    with engine.begin() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})

def exec_sql(sql: str, params: dict | None = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(sql), params or {})

def table_exists(schema: str, table: str) -> bool:
    df = qdf(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = :schema AND table_name = :table
        ) AS ok
        """,
        {"schema": schema, "table": table},
    )
    return bool(df.iloc[0]["ok"]) if not df.empty else False

def count_rows(schema: str, table: str) -> int:
    if not table_exists(schema, table):
        return 0
    df = qdf(f'SELECT COUNT(*)::bigint AS n FROM {schema}."{table}"')
    return int(df.iloc[0]["n"]) if not df.empty else 0

def ensure_schema(schema: str):
    exec_sql(f"CREATE SCHEMA IF NOT EXISTS {schema};")

def has_gen_random_uuid() -> bool:
    try:
        exec_sql("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
    except Exception:
        pass
    df = qdf(
        """
        SELECT EXISTS (
          SELECT 1
          FROM pg_proc
          WHERE proname = 'gen_random_uuid'
        ) AS ok
        """
    )
    return bool(df.iloc[0]["ok"]) if not df.empty else False

def ensure_output_table():
    ensure_schema(SCHEMA)
    gen_ok = has_gen_random_uuid()
    uuid_default = "DEFAULT gen_random_uuid()" if gen_ok else ""

    # NOTE: ini f-string -> '{}' harus jadi '{{}}'
    exec_sql(
        f"""
        CREATE TABLE IF NOT EXISTS {SCHEMA}."{T_OUT}" (
            dataset_id uuid {uuid_default},
            run_time timestamptz NOT NULL DEFAULT now(),

            -- Identitas sumber labeling
            jenis_pendekatan text NOT NULL,
            job_id uuid,
            modeling_id uuid NOT NULL,
            window_days integer NOT NULL,
            time_col text NOT NULL DEFAULT '',
            include_noise boolean,
            eligible_rule text,

            -- Identitas tiket
            incident_number text NOT NULL,
            event_time timestamp without time zone,
            tgl_submit timestamp without time zone,

            -- Informasi klaster/temporal
            cluster_id bigint,
            temporal_cluster_no integer,
            temporal_cluster_id text,
            gap_days integer,
            n_member_cluster bigint,
            n_episode_cluster bigint,
            n_member_episode bigint,
            min_time timestamp without time zone,
            max_time timestamp without time zone,

            -- Metadata (fitur)
            site text,
            assignee text,
            modul text,
            sub_modul text,

            -- Target
            label_berulang integer NOT NULL,

            -- Opsional: teks (jika join)
            text_col_1 text,
            text_col_2 text,

            -- Split
            split_name text,
            split_cutoff timestamp without time zone,

            -- Simpan parameter builder
            builder_json jsonb NOT NULL DEFAULT '{{}}'::jsonb,

            CONSTRAINT dataset_supervised_pkey
                PRIMARY KEY (jenis_pendekatan, modeling_id, window_days, incident_number, time_col)
        );
        """
    )

def discover_join_candidates(schema: str = SCHEMA) -> pd.DataFrame:
    return qdf(
        """
        SELECT table_name
        FROM information_schema.columns
        WHERE table_schema = :schema
          AND column_name = 'incident_number'
        ORDER BY table_name
        """,
        {"schema": schema},
    )

def discover_text_columns(schema: str, table: str) -> list[str]:
    preferred = [
        "judul", "title", "summary", "ringkasan",
        "deskripsi", "description", "uraian", "detail",
        "subject", "problem", "keterangan", "notes", "catatan",
        "teks", "text"
    ]
    cols = qdf(
        """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
        ORDER BY ordinal_position
        """,
        {"schema": schema, "table": table},
    )
    text_like = cols[cols["data_type"].isin(["text", "character varying", "character"])].copy()

    ordered: list[str] = []
    for p in preferred:
        hits = text_like[text_like["column_name"].str.lower() == p.lower()]["column_name"].tolist()
        ordered += hits
    for c in text_like["column_name"].tolist():
        if c not in ordered:
            ordered.append(c)
    return ordered

def get_latest_run_time(jenis: str, modeling_id: str, window_days: int, time_col: str):
    df = qdf(
        f"""
        SELECT MAX(run_time) AS last_run_time
        FROM {SCHEMA}."{T_LABEL}"
        WHERE jenis_pendekatan = :jenis
          AND modeling_id = CAST(:modeling_id AS uuid)
          AND window_days = :window_days
          AND time_col = :time_col
        """,
        {"jenis": jenis, "modeling_id": modeling_id, "window_days": window_days, "time_col": time_col},
    )
    if df.empty:
        return None
    return df.iloc[0]["last_run_time"]

def build_dataset(params: dict) -> tuple[pd.DataFrame, datetime | None]:
    last_rt = get_latest_run_time(
        params["jenis_pendekatan"],
        params["modeling_id"],
        params["window_days"],
        params["time_col"],
    )
    if last_rt is None:
        return pd.DataFrame(), None

    base_cols = """
        L.jenis_pendekatan, L.job_id, L.modeling_id, L.window_days, L.time_col,
        L.include_noise, L.eligible_rule,
        L.incident_number, L.event_time, L.tgl_submit,
        L.cluster_id, L.temporal_cluster_no, L.temporal_cluster_id, L.gap_days,
        L.site, L.assignee, L.modul, L.sub_modul,
        L.n_member_cluster, L.n_episode_cluster, L.n_member_episode,
        L.min_time, L.max_time,
        L.label_berulang
    """

    where = """
        L.jenis_pendekatan = :jenis_pendekatan
        AND L.modeling_id = CAST(:modeling_id AS uuid)
        AND L.window_days = :window_days
        AND L.time_col = :time_col
        AND L.run_time = :run_time
    """

    if params["exclude_noise"]:
        where += " AND L.cluster_id <> -1"

    join_sql = ""
    text_select = ", NULL::text AS text_col_1, NULL::text AS text_col_2"

    if params.get("join_table") and params.get("text_col_1"):
        jt = params["join_table"]
        c1 = params["text_col_1"]
        c2 = params.get("text_col_2")

        join_sql = f"""
            LEFT JOIN {SCHEMA}."{jt}" J
                ON J.incident_number = L.incident_number
        """
        if c2 and c2 != "(none)":
            text_select = f', J."{c1}" AS text_col_1, J."{c2}" AS text_col_2'
        else:
            text_select = f', J."{c1}" AS text_col_1, NULL::text AS text_col_2'

    sql = f"""
        SELECT
            {base_cols}
            {text_select}
        FROM {SCHEMA}."{T_LABEL}" L
        {join_sql}
        WHERE {where}
        ORDER BY COALESCE(L.event_time, L.tgl_submit) NULLS LAST, L.incident_number
    """

    df = qdf(
        sql,
        {
            "jenis_pendekatan": params["jenis_pendekatan"],
            "modeling_id": params["modeling_id"],
            "window_days": params["window_days"],
            "time_col": params["time_col"],
            "run_time": last_rt,
        },
    )

    split_cutoff = params.get("split_cutoff")
    if split_cutoff:
        cutoff_dt = pd.to_datetime(split_cutoff)
        t = pd.to_datetime(df["event_time"]).fillna(pd.to_datetime(df["tgl_submit"]))
        df["split_cutoff"] = cutoff_dt
        df["split_name"] = "train"
        df.loc[t >= cutoff_dt, "split_name"] = "test"
    else:
        df["split_cutoff"] = pd.NaT
        df["split_name"] = "train"

    return df, last_rt

# ======================================================
# ✅ KRUSIAL: Sanitizer agar NaT/NaN tidak nyasar ke PostgreSQL
# ======================================================
def _to_pg_value(v: Any):
    # None
    if v is None:
        return None

    # Pandas NA/NaT/NaN
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass

    # Pandas Timestamp -> datetime python
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()

    # numpy scalar -> python scalar
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        fv = float(v)
        if math.isnan(fv):
            return None
        return fv
    if isinstance(v, (np.bool_,)):
        return bool(v)

    # dict/list -> JSON string
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False)

    return v

def _sanitize_rows(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        out.append({k: _to_pg_value(v) for k, v in r.items()})
    return out

def upsert_dataset(df: pd.DataFrame, builder_json: dict):
    ensure_output_table()

    if df is None or df.empty:
        st.warning("DataFrame kosong. Tidak ada data yang disimpan.")
        return

    builder_json_str = json.dumps(builder_json, ensure_ascii=False)

    # dataset_id selalu diisi (tanpa tergantung pgcrypto)
    if "dataset_id" not in df.columns:
        df.insert(0, "dataset_id", [str(uuid.uuid4()) for _ in range(len(df))])
    else:
        df["dataset_id"] = df["dataset_id"].astype("string")
        mask = df["dataset_id"].isna() | (df["dataset_id"] == "")
        if mask.any():
            df.loc[mask, "dataset_id"] = [str(uuid.uuid4()) for _ in range(int(mask.sum()))]

    # Pastikan kolom split ada
    if "split_cutoff" not in df.columns:
        df["split_cutoff"] = pd.NaT
    if "split_name" not in df.columns:
        df["split_name"] = "train"

    # builder_json diisi per-row
    df["builder_json"] = builder_json_str

    upsert_sql = f"""
    INSERT INTO {SCHEMA}."{T_OUT}" (
        dataset_id,
        jenis_pendekatan, job_id, modeling_id, window_days, time_col, include_noise, eligible_rule,
        incident_number, event_time, tgl_submit,
        cluster_id, temporal_cluster_no, temporal_cluster_id, gap_days,
        n_member_cluster, n_episode_cluster, n_member_episode,
        min_time, max_time,
        site, assignee, modul, sub_modul,
        label_berulang,
        text_col_1, text_col_2,
        split_name, split_cutoff,
        builder_json
    )
    VALUES (
        CAST(:dataset_id AS uuid),
        :jenis_pendekatan, :job_id, CAST(:modeling_id AS uuid), :window_days, :time_col, :include_noise, :eligible_rule,
        :incident_number, :event_time, :tgl_submit,
        :cluster_id, :temporal_cluster_no, :temporal_cluster_id, :gap_days,
        :n_member_cluster, :n_episode_cluster, :n_member_episode,
        :min_time, :max_time,
        :site, :assignee, :modul, :sub_modul,
        :label_berulang,
        :text_col_1, :text_col_2,
        :split_name, :split_cutoff,
        CAST(:builder_json AS jsonb)
    )
    ON CONFLICT (jenis_pendekatan, modeling_id, window_days, incident_number, time_col)
    DO UPDATE SET
        run_time = now(),
        job_id = EXCLUDED.job_id,
        include_noise = EXCLUDED.include_noise,
        eligible_rule = EXCLUDED.eligible_rule,
        event_time = EXCLUDED.event_time,
        tgl_submit = EXCLUDED.tgl_submit,
        cluster_id = EXCLUDED.cluster_id,
        temporal_cluster_no = EXCLUDED.temporal_cluster_no,
        temporal_cluster_id = EXCLUDED.temporal_cluster_id,
        gap_days = EXCLUDED.gap_days,
        n_member_cluster = EXCLUDED.n_member_cluster,
        n_episode_cluster = EXCLUDED.n_episode_cluster,
        n_member_episode = EXCLUDED.n_member_episode,
        min_time = EXCLUDED.min_time,
        max_time = EXCLUDED.max_time,
        site = EXCLUDED.site,
        assignee = EXCLUDED.assignee,
        modul = EXCLUDED.modul,
        sub_modul = EXCLUDED.sub_modul,
        label_berulang = EXCLUDED.label_berulang,
        text_col_1 = EXCLUDED.text_col_1,
        text_col_2 = EXCLUDED.text_col_2,
        split_name = EXCLUDED.split_name,
        split_cutoff = EXCLUDED.split_cutoff,
        builder_json = EXCLUDED.builder_json;
    """

    # ✅ KRUSIAL: sanitasi di level rows dict (NaT/NaN -> None)
    rows = df.to_dict(orient="records")
    rows = _sanitize_rows(rows)

    with engine.begin() as conn:
        conn.execute(text(upsert_sql), rows)  # batch executemany

# ======================================================
# ✅ Pastikan tabel terbentuk saat halaman dibuka
# ======================================================
ensure_output_table()

# ======================================================
# 🖥️ UI
# ======================================================
st.title("📦 Builder Dataset Supervised (dari incident_labeling_results)")
st.caption("Membentuk dataset supervised (X, y) dari hasil pelabelan insiden berulang, termasuk opsi join teks dan temporal split.")

exists = table_exists(SCHEMA, T_OUT)
nrows = count_rows(SCHEMA, T_OUT) if exists else 0
st.info(f"Status tabel output: `{SCHEMA}.{T_OUT}` => {'✅ ADA' if exists else '❌ TIDAK ADA'} | rows saat ini: **{nrows:,}**")

# ---- pilih run
runs = qdf(
    f"""
WITH latest AS (
  SELECT
    jenis_pendekatan,
    modeling_id,
    window_days,
    time_col,
    MAX(run_time) AS last_run_time
  FROM {SCHEMA}."{T_LABEL}"
  GROUP BY 1,2,3,4
)
SELECT *
FROM latest
ORDER BY last_run_time DESC
"""
)

if runs.empty:
    st.warning("Belum ada data pada incident_labeling_results.")
    st.stop()

colA, colB, colC, colD = st.columns([1.2, 1.4, 1.0, 1.0])
with colA:
    jenis = st.selectbox("jenis_pendekatan", sorted(runs["jenis_pendekatan"].unique().tolist()))
sub = runs[runs["jenis_pendekatan"] == jenis].copy()

with colB:
    modeling_opts = sub["modeling_id"].astype(str).unique().tolist()
    modeling_id = st.selectbox("modeling_id", modeling_opts)

sub2 = sub[sub["modeling_id"].astype(str) == modeling_id].copy()
with colC:
    window_days = st.selectbox("window_days", sorted(sub2["window_days"].unique().tolist()))
sub3 = sub2[sub2["window_days"] == window_days].copy()

with colD:
    time_col = st.selectbox("time_col", sorted(sub3["time_col"].unique().tolist()))

st.divider()

# ---- join teks
st.subheader("Opsi Join Teks (Opsional)")
cands = discover_join_candidates(SCHEMA)
join_table = None
text_col_1 = None
text_col_2 = None

if cands.empty:
    st.info("Tidak ditemukan tabel lain di schema lasis_djp yang memiliki kolom incident_number.")
else:
    join_table = st.selectbox(
        "Pilih tabel sumber teks (optional)",
        ["(none)"] + cands["table_name"].tolist(),
        index=0,
    )

    if join_table != "(none)":
        text_cols = discover_text_columns(SCHEMA, join_table)
        if not text_cols:
            st.warning("Tabel terpilih tidak memiliki kolom bertipe text/varchar.")
            join_table = "(none)"
        else:
            cc1, cc2 = st.columns(2)
            with cc1:
                text_col_1 = st.selectbox("Kolom teks 1", text_cols, index=0)
            with cc2:
                text_col_2 = st.selectbox("Kolom teks 2 (opsional)", ["(none)"] + text_cols, index=0)

st.divider()

# ---- filter & split
st.subheader("Filter & Split")
c1, c2, c3, c4 = st.columns([1.2, 1.2, 1.6, 1.0])
with c1:
    exclude_noise = st.checkbox("Exclude noise (cluster_id = -1)", value=True)
with c2:
    split_on = st.checkbox("Aktifkan temporal split (train/test)", value=True)
with c3:
    split_cutoff = None
    if split_on:
        split_cutoff = st.date_input("Cutoff date (>= cutoff => test)", value=None)
        if split_cutoff:
            split_cutoff = datetime(split_cutoff.year, split_cutoff.month, split_cutoff.day)
with c4:
    if st.button("🧹 Reset dataset"):
        st.session_state.ds_df = None
        st.session_state.ds_builder_json = None
        st.session_state.ds_signature = None
        st.success("Dataset di-reset dari session.")
        st.rerun()

params = dict(
    jenis_pendekatan=jenis,
    modeling_id=modeling_id,
    window_days=int(window_days),
    time_col=time_col,
    exclude_noise=exclude_noise,
    join_table=None if (join_table in [None, "(none)"]) else join_table,
    text_col_1=None if (not text_col_1) else text_col_1,
    text_col_2=None if (not text_col_2) else text_col_2,
    split_cutoff=split_cutoff if split_on else None,
)

signature = json.dumps(params, default=str, sort_keys=True)

# ---- build
btn = st.button("🔧 Bangun Dataset", type="primary")
if btn:
    with st.spinner("Menarik data dan membentuk dataset..."):
        df, last_rt = build_dataset(params)

    if df.empty:
        st.warning("Hasil build_dataset kosong. Coba kombinasi pilihan lain (jenis/model/window/time_col).")
        st.session_state.ds_df = None
        st.session_state.ds_builder_json = None
        st.session_state.ds_signature = None
    else:
        st.session_state.ds_df = df
        st.session_state.ds_signature = signature
        st.session_state.ds_builder_json = {
            "source_table": f"{SCHEMA}.{T_LABEL}",
            "jenis_pendekatan": jenis,
            "modeling_id": modeling_id,
            "window_days": int(window_days),
            "time_col": time_col,
            "exclude_noise": exclude_noise,
            "join_table": params["join_table"],
            "text_col_1": params["text_col_1"],
            "text_col_2": params["text_col_2"],
            "split_on": bool(split_on),
            "split_cutoff": split_cutoff.isoformat() if split_cutoff else None,
            "label_run_time": last_rt.isoformat() if last_rt is not None else None,
            "built_at": datetime.now().isoformat(),
        }
        st.success(f"Dataset terbentuk: {len(df):,} baris (run_time={last_rt})")

st.divider()

# ---- preview & upsert (pakai session_state)
df = st.session_state.ds_df
if df is None or df.empty:
    st.info("Belum ada dataset di session. Klik **Bangun Dataset** terlebih dahulu.")
    st.stop()

# jika user mengubah pilihan, beri peringatan
if st.session_state.ds_signature != signature:
    st.warning("Pilihan filter/run sudah berubah sejak dataset terakhir dibangun. Disarankan klik **Bangun Dataset** lagi agar konsisten.")

st.write("Preview (50 baris):")
st.dataframe(df.head(50), use_container_width=True)

st.write("Ringkasan label:")
vc = df["label_berulang"].value_counts(dropna=False).rename_axis("label").reset_index(name="n")
st.dataframe(vc, use_container_width=True)

if split_on and split_cutoff:
    st.write("Ringkasan split:")
    sc = df["split_name"].value_counts().rename_axis("split").reset_index(name="n")
    st.dataframe(sc, use_container_width=True)

csv_bytes = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download CSV", data=csv_bytes, file_name="dataset_supervised.csv", mime="text/csv")

st.subheader("Simpan ke Database")
if st.button("💾 Upsert ke lasis_djp.dataset_supervised"):
    with st.spinner("Menyimpan ke database (batch upsert)..."):
        upsert_dataset(df, st.session_state.ds_builder_json or {})
    st.success("Upsert selesai ✅")
    st.info(f"Rows sekarang di `{SCHEMA}.{T_OUT}`: **{count_rows(SCHEMA, T_OUT):,}**")

st.caption(
    "Catatan: dataset ini menyimpan fitur mentah (raw). Encoding fitur kategorikal/teks dan training model dilakukan pada halaman/skrip berikutnya."
)
