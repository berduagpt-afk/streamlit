# pages/data_preparation/feature_extraction/viewer_sintaksis_tfidf.py
from __future__ import annotations

import math
import pandas as pd
import streamlit as st
import altair as alt
from sqlalchemy import create_engine, text


# ======================================================
# 🔐 Guard Login
# ======================================================
if not st.session_state.get("logged_in", False):
    st.error("Silakan login terlebih dahulu.")
    st.stop()


# ======================================================
# ⚙️ Konstanta (TABEL BARU)
# ======================================================
SCHEMA = "lasis_djp"
T_RUNS = "incident_tfidf_runs"
T_VECTORS = "incident_tfidf_vectors"


# ======================================================
# 🔌 Database Connection
# ======================================================
@st.cache_resource(show_spinner=False)
def get_engine():
    cfg = st.secrets["connections"]["postgres"]
    return create_engine(
        f"postgresql+psycopg2://{cfg['username']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}",
        pool_pre_ping=True,
    )


engine = get_engine()


# ======================================================
# 🧠 Data access
# ======================================================
@st.cache_data(show_spinner=False, ttl=60)
def fetch_runs(limit: int = 1000) -> pd.DataFrame:
    q = text(f"""
        SELECT run_id, run_time, approach, params_json, data_range, notes, idf_json, feature_names_json
        FROM {SCHEMA}.{T_RUNS}
        ORDER BY run_time DESC
        LIMIT :limit
    """)
    with engine.connect() as conn:
        return pd.read_sql(q, conn, params={"limit": int(limit)})


@st.cache_data(show_spinner=False, ttl=60)
def fetch_distinct_modul_site(run_id: str):
    q_modul = text(f"""
        SELECT DISTINCT modul
        FROM {SCHEMA}.{T_VECTORS}
        WHERE run_id = :run_id AND modul IS NOT NULL
        ORDER BY modul
    """)
    q_site = text(f"""
        SELECT DISTINCT site
        FROM {SCHEMA}.{T_VECTORS}
        WHERE run_id = :run_id AND site IS NOT NULL
        ORDER BY site
    """)
    with engine.connect() as conn:
        dfm = pd.read_sql(q_modul, conn, params={"run_id": run_id})
        dfs = pd.read_sql(q_site, conn, params={"run_id": run_id})
    moduls = ["Semua"] + dfm["modul"].dropna().astype(str).tolist()
    sites = ["Semua"] + dfs["site"].dropna().astype(str).tolist()
    return moduls, sites


def fetch_vectors_count(run_id: str, modul: str, site: str) -> int:
    where = ["run_id = :run_id"]
    params = {"run_id": run_id}

    if modul != "Semua":
        where.append("modul = :modul")
        params["modul"] = modul

    if site != "Semua":
        where.append("site = :site")
        params["site"] = site

    q = text(f"""
        SELECT COUNT(*) AS n
        FROM {SCHEMA}.{T_VECTORS}
        WHERE {" AND ".join(where)}
    """)
    with engine.connect() as conn:
        return int(conn.execute(q, params).scalar_one())


def fetch_vectors_page(run_id: str, modul: str, site: str, page: int, page_size: int) -> pd.DataFrame:
    offset = (page - 1) * page_size

    where = ["run_id = :run_id"]
    params = {"run_id": run_id, "limit": page_size, "offset": offset}

    if modul != "Semua":
        where.append("modul = :modul")
        params["modul"] = modul

    if site != "Semua":
        where.append("site = :site")
        params["site"] = site

    q = text(f"""
        SELECT
            incident_number,
            tgl_submit,
            site,
            assignee,
            modul,
            sub_modul,

            text_sintaksis,
            tokens_sintaksis_json,

            subject_tokens_json,
            tf_json,
            tfidf_json,
            tfidf_vec_json
        FROM {SCHEMA}.{T_VECTORS}
        WHERE {" AND ".join(where)}
        ORDER BY tgl_submit DESC NULLS LAST
        LIMIT :limit OFFSET :offset
    """)
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, params=params)

    if not df.empty:
        def _tok_len(r):
            v = r.get("subject_tokens_json")
            if isinstance(v, list):
                return len(v)
            v2 = r.get("tokens_sintaksis_json")
            return len(v2) if isinstance(v2, list) else 0

        df["n_tokens"] = df.apply(_tok_len, axis=1)
        df["nnz_terms"] = df["tfidf_json"].apply(lambda x: len(x) if isinstance(x, dict) else 0)

    return df


# ======================================================
# Helpers presentasi
# ======================================================
def dict_to_top_df(d: dict, k: int, colname: str) -> pd.DataFrame:
    if not isinstance(d, dict) or not d:
        return pd.DataFrame(columns=["term", colname])
    items = [(str(t), float(w)) for t, w in d.items() if w is not None]
    items.sort(key=lambda x: x[1], reverse=True)
    return pd.DataFrame(items[:k], columns=["term", colname])


def build_tf_idf_tfidf_table(tf: dict, idf: dict, tfidf: dict, k: int = 30) -> pd.DataFrame:
    if not isinstance(tfidf, dict) or not tfidf:
        return pd.DataFrame(columns=["term", "tf", "idf", "tfidf"])
    top_terms = sorted(tfidf.items(), key=lambda kv: kv[1], reverse=True)[:k]
    rows = []
    for term, w in top_terms:
        rows.append({
            "term": str(term),
            "tf": float(tf.get(term, 0.0)) if isinstance(tf, dict) else 0.0,
            "idf": float(idf.get(term, 0.0)) if isinstance(idf, dict) else 0.0,
            "tfidf": float(w),
        })
    return pd.DataFrame(rows)


def safe_json_preview(obj, max_chars: int = 700) -> str:
    try:
        import json
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        try:
            s = str(obj)
        except Exception:
            s = "<unprintable>"
    return s if len(s) <= max_chars else s[:max_chars] + "…"


def top_k_terms_from_tfidf(tfidf: dict, k: int) -> list[str]:
    if not isinstance(tfidf, dict) or not tfidf:
        return []
    items = [(str(t), float(w)) for t, w in tfidf.items() if w is not None]
    items.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in items[:k]]


def subset_dict_by_terms(d: dict, terms: list[str]) -> dict:
    if not isinstance(d, dict) or not terms:
        return {}
    out = {}
    for t in terms:
        if t in d:
            out[t] = d[t]
    return out


def make_one_row_preview(tokens: list, tf: dict, idf_global: dict, tfidf: dict, vec, k_terms: int = 20) -> pd.DataFrame:
    top_terms = top_k_terms_from_tfidf(tfidf, k_terms)
    tf_sub = subset_dict_by_terms(tf, top_terms)
    idf_sub = subset_dict_by_terms(idf_global, top_terms)
    tfidf_sub = subset_dict_by_terms(tfidf, top_terms)

    if isinstance(vec, list):
        vec_len = len(vec)
        vec_preview = vec[:20]
    else:
        vec_len = 0 if vec is None else -1
        vec_preview = None

    row_render = {
        "subject_tokens_json": safe_json_preview(tokens),
        "TF (top terms)": safe_json_preview(tf_sub),
        "IDF (top terms)": safe_json_preview(idf_sub),
        "TF-IDF (top terms)": safe_json_preview(tfidf_sub),
        "TF-IDF Vec Len": vec_len,
        "TF-IDF Vec Preview (first 20)": safe_json_preview(vec_preview),
    }
    return pd.DataFrame([row_render])


def reconstruct_dense_from_sparse(tfidf_row: dict, feature_names: list) -> tuple[list[float] | None, int]:
    """
    Opsi C: reconstruct dense vector from tfidf_json + feature_names_json.
    Return (dense_vector or None, nnz_filled).
    """
    if not isinstance(feature_names, list) or len(feature_names) == 0:
        return None, 0
    if not isinstance(tfidf_row, dict) or len(tfidf_row) == 0:
        return [0.0] * len(feature_names), 0

    idx = {str(t): i for i, t in enumerate(feature_names)}
    dense = [0.0] * len(feature_names)
    filled = 0
    for term, w in tfidf_row.items():
        j = idx.get(str(term))
        if j is not None:
            dense[j] = float(w)
            filled += 1
    return dense, filled


# ======================================================
# 🧾 UI
# ======================================================
st.title("Viewer Feature Extraction — Sintaksis (TF-IDF)")
st.info(
    "Proses **feature extraction sintaksis (TF-IDF)** dijalankan **offline & batch** untuk efisiensi "
    "pada data skala besar. Halaman ini hanya menampilkan hasil yang tersimpan di database (read-only)."
)

df_runs = fetch_runs()
if df_runs.empty:
    st.warning(f"Belum ada data pada {SCHEMA}.{T_RUNS}.")
    st.stop()

df_runs["label"] = (
    df_runs["run_time"].astype(str)
    + " | "
    + df_runs["approach"].astype(str)
    + " | "
    + df_runs["run_id"].astype(str)
)

selected = st.selectbox("Pilih Run TF-IDF", df_runs["label"].tolist(), index=0)
run = df_runs[df_runs["label"] == selected].iloc[0]
run_id = str(run["run_id"])

idf_global = run.get("idf_json") if isinstance(run.get("idf_json"), dict) else {}
feature_names = run.get("feature_names_json") if isinstance(run.get("feature_names_json"), list) else []

with st.expander("Detail Run", expanded=False):
    st.write("**run_id**:", run_id)
    st.write("**run_time**:", run["run_time"])
    st.write("**approach**:", run["approach"])
    st.write("**params_json**:")
    st.json(run["params_json"])
    st.write("**data_range**:")
    st.json(run["data_range"] or {})
    st.write("**notes**:", run["notes"] or "-")
    st.write("**idf_json (global)**:", f"{len(idf_global):,} terms")
    st.write("**feature_names_json**:", f"{len(feature_names):,} features")

moduls, sites = fetch_distinct_modul_site(run_id)

c1, c2, c3 = st.columns(3)
f_modul = c1.selectbox("Filter Modul", moduls, index=0)
f_site = c2.selectbox("Filter Site", sites, index=0)
page_size = c3.selectbox("Page Size", [50, 100, 200, 500], index=2)

total = fetch_vectors_count(run_id, f_modul, f_site)
if total == 0:
    st.warning("Tidak ada data vectors untuk filter tersebut.")
    st.stop()

total_pages = max(1, math.ceil(total / int(page_size)))
page = st.number_input("Halaman", min_value=1, max_value=total_pages, value=1, step=1)
df = fetch_vectors_page(run_id, f_modul, f_site, int(page), int(page_size))

st.caption(f"Menampilkan {len(df)} baris dari total {total} (halaman {page}/{total_pages})")

st.subheader("Ringkasan TF-IDF Vectors")
st.dataframe(
    df[["incident_number", "tgl_submit", "site", "assignee", "modul", "sub_modul", "n_tokens", "nnz_terms"]],
    use_container_width=True,
    hide_index=True,
)

if len(df) > 1 and "nnz_terms" in df.columns:
    hist = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("nnz_terms:Q", bin=alt.Bin(maxbins=30)),
            y="count()",
            tooltip=["count()"],
        )
        .properties(height=180)
    )
    st.altair_chart(hist, use_container_width=True)

st.divider()
st.subheader("Drill-down TF-IDF per Tiket")

incident_options = df["incident_number"].dropna().astype(str).tolist()
incident = st.selectbox("Pilih incident_number (dari halaman aktif)", incident_options, index=0)

row = df[df["incident_number"].astype(str) == str(incident)].iloc[0]

tokens_view = row.get("subject_tokens_json")
if not isinstance(tokens_view, list):
    tokens_view = row.get("tokens_sintaksis_json")
if not isinstance(tokens_view, list):
    tokens_view = []

tf_row = row.get("tf_json") if isinstance(row.get("tf_json"), dict) else {}
tfidf_row = row.get("tfidf_json") if isinstance(row.get("tfidf_json"), dict) else {}
vec_row = row.get("tfidf_vec_json")  # bisa None jika --no-dense-vec

# ======================================================
# ✅ TAMBAHAN: 1-ROW DATAFRAME "SEPERTI GAMBAR" (IDF TOP TERMS)
# ======================================================
st.markdown("### Preview 1 Baris (Subject | TF | IDF | TF-IDF | TF-IDF Vec)")
k_terms = st.slider("Jumlah term untuk preview (Top TF-IDF)", min_value=5, max_value=50, value=20, step=5)

# kalau vec kosong, kita isi preview-nya nanti dari hasil rekonstruksi (untuk 1-row)
vec_for_preview = vec_row
if not isinstance(vec_for_preview, list):
    dense_tmp, _filled = reconstruct_dense_from_sparse(tfidf_row, feature_names)
    vec_for_preview = dense_tmp

one_row_df = make_one_row_preview(tokens_view, tf_row, idf_global, tfidf_row, vec_for_preview, k_terms=int(k_terms))
st.dataframe(one_row_df, use_container_width=True, hide_index=True)

# ======================================================
# Detail per tiket
# ======================================================
colA, colB = st.columns([1, 1], vertical_alignment="top")
with colA:
    st.write("**Text Sintaksis**")
    st.text_area("", value=row.get("text_sintaksis") or "-", height=200)

    with st.expander("Tokens (subject_tokens_json)", expanded=False):
        st.json(tokens_view)

    with st.expander("TF-IDF Vector (dense) — panjang & preview", expanded=False):
        if isinstance(vec_row, list):
            st.write(f"Panjang vector: **{len(vec_row):,}** (harus sama dengan feature_names_json)")
            st.json(vec_row[:50])
        else:
            # Opsi C: reconstruct dense vector
            dense, filled = reconstruct_dense_from_sparse(tfidf_row, feature_names)
            if dense is None:
                st.info("tfidf_vec_json tidak disimpan, dan feature_names_json tidak tersedia → tidak bisa rekonstruksi.")
            else:
                st.success(
                    "tfidf_vec_json tidak disimpan (run dengan --no-dense-vec). "
                    f"Dense vector direkonstruksi dari tfidf_json. n_features={len(dense):,}, nnz_terms={filled:,}."
                )
                st.json(dense[:50])

with colB:
    st.write("**Top TF-IDF Terms**")
    top_tfidf = dict_to_top_df(tfidf_row, k=20, colname="tfidf")
    if top_tfidf.empty:
        st.warning("tfidf_json kosong / tidak valid.")
    else:
        chart = (
            alt.Chart(top_tfidf)
            .mark_bar()
            .encode(
                x="tfidf:Q",
                y=alt.Y("term:N", sort="-x"),
                tooltip=["term:N", "tfidf:Q"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(top_tfidf, use_container_width=True, hide_index=True)

    st.write("**Tabel TF + IDF + TF-IDF (Top Terms)**")
    if not idf_global:
        st.warning("idf_json tidak tersedia di tabel runs untuk run ini.")
    else:
        combo = build_tf_idf_tfidf_table(tf_row, idf_global, tfidf_row, k=30)
        st.dataframe(combo, use_container_width=True, hide_index=True)

    with st.expander("Top TF Terms (opsional)", expanded=False):
        top_tf = dict_to_top_df(tf_row, k=30, colname="tf")
        if top_tf.empty:
            st.info("tf_json kosong / tidak tersedia.")
        else:
            st.dataframe(top_tf, use_container_width=True, hide_index=True)
