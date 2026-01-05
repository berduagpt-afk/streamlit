# pages/modeling_semantik_embedding_viewer.py
# ============================================================
# Viewer Hasil Embedding Semantik (SBERT / IndoBERT)
#
# Sumber:
# - lasis_djp.semantik_embedding_runs
# - lasis_djp.semantik_embedding_vectors
#
# Fitur:
# - Pilih run_id embedding
# - KPI ringkas (rows, dimensi, provider, model)
# - Preview embedding (truncate)
# - Visualisasi UMAP 2D (opsional, untuk interpretasi visual)
#
# Catatan metodologis:
# - UMAP hanya untuk visualisasi
# - Evaluasi & clustering tetap di ruang embedding asli / PCA
# ============================================================

from __future__ import annotations

import json
import numpy as np
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
# ⚙️ KONSTANTA
# ======================================================
SCHEMA = "lasis_djp"

T_RUNS = "semantik_embedding_runs"
T_VECS = "semantik_embedding_vectors"


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


# ======================================================
# 📥 LOADERS
# ======================================================
@st.cache_data(show_spinner=False)
def load_runs() -> pd.DataFrame:
    eng = get_engine()
    q = f"""
        SELECT
            run_id::text,
            run_time,
            source_table,
            n_rows,
            provider,
            model_name,
            device,
            batch_size,
            max_length,
            normalize_embeddings,
            params_json,
            notes
        FROM {SCHEMA}.{T_RUNS}
        ORDER BY run_time DESC
    """
    with eng.connect() as c:
        return pd.read_sql(text(q), c)


@st.cache_data(show_spinner=False)
def load_vectors(run_id: str, limit: int) -> pd.DataFrame:
    eng = get_engine()
    q = f"""
        SELECT
            incident_number,
            n_chars,
            embedding_dim,
            embedding_json
        FROM {SCHEMA}.{T_VECS}
        WHERE run_id = :rid
        LIMIT :limit
    """
    with eng.connect() as c:
        return pd.read_sql(
            text(q),
            c,
            params={"rid": run_id, "limit": int(limit)},
        )


# ======================================================
# 🧾 UI HEADER
# ======================================================
st.title("🧠 Hasil Embedding Semantik")
st.caption(
    "Viewer read-only untuk validasi & eksplorasi embedding semantik "
    "(Sentence-BERT / IndoBERT)."
)


# ======================================================
# 📦 LOAD RUNS
# ======================================================
df_runs = load_runs()

if df_runs.empty:
    st.warning("Belum ada embedding semantik.")
    st.stop()


# ======================================================
# 🧭 SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Parameter Viewer")

    run_id = st.selectbox(
        "Pilih Embedding Run ID",
        options=df_runs["run_id"].tolist(),
        format_func=lambda x: x[:8],
    )

    limit_rows = st.number_input(
        "Limit baris embedding",
        min_value=200,
        max_value=20_000,
        value=2_000,
        step=500,
        help="Untuk preview & UMAP. Tidak mempengaruhi data di DB.",
    )

    show_umap = st.checkbox("Tampilkan UMAP (2D)", value=True)


# ======================================================
# 📌 RUN INFO + KPI
# ======================================================
run = df_runs[df_runs["run_id"] == run_id].iloc[0]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Jumlah Tiket", f"{int(run.n_rows):,}")
c2.metric("Provider", run.provider.upper())
c3.metric("Dimensi", "—")  # diisi setelah load vector
c4.metric("Batch Size", str(run.batch_size))
c5.metric("Normalize", "ON" if run.normalize_embeddings else "OFF")

with st.expander("ℹ️ Detail Run"):
    st.write(f"**Model**: {run.model_name}")
    st.write(f"**Device**: {run.device}")
    st.write(f"**Source**: {run.source_table}")
    st.json(run.params_json)


# ======================================================
# 📥 LOAD VECTORS
# ======================================================
with st.spinner("Memuat embedding vectors ..."):
    df_vecs = load_vectors(run_id, int(limit_rows))

if df_vecs.empty:
    st.warning("Embedding vectors kosong.")
    st.stop()

embedding_dim = int(df_vecs["embedding_dim"].iloc[0])
c3.metric("Dimensi", str(embedding_dim))


# ======================================================
# 📋 PREVIEW EMBEDDING (TRUNCATED)
# ======================================================
st.subheader("📋 Preview Embedding Vector (Truncated)")

def truncate_vec(v, n=8):
    if isinstance(v, str):
        v = json.loads(v)
    return v[:n]

df_preview = df_vecs.copy()
df_preview["embedding_preview"] = df_preview["embedding_json"].apply(truncate_vec)
df_preview = df_preview.drop(columns=["embedding_json"])

st.dataframe(
    df_preview.head(10),
    use_container_width=True,
)


# ======================================================
# 🔮 UMAP VISUALIZATION (OPTIONAL)
# ======================================================
if show_umap:
    st.subheader("🔮 Visualisasi UMAP (2D)")
    st.caption(
        "UMAP hanya digunakan untuk interpretasi visual. "
        "Tidak digunakan untuk clustering atau evaluasi."
    )

    with st.spinner("Menghitung UMAP ..."):
        import umap

        X = []
        for v in df_vecs["embedding_json"].tolist():
            if isinstance(v, str):
                X.append(json.loads(v))
            else:
                X.append(v)

        X = np.asarray(X, dtype=np.float32)

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=42,
        )
        X_umap = reducer.fit_transform(X)

    df_umap = pd.DataFrame(
        {
            "x": X_umap[:, 0],
            "y": X_umap[:, 1],
        }
    )

    chart = (
        alt.Chart(df_umap)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X("x:Q", title="UMAP-1"),
            y=alt.Y("y:Q", title="UMAP-2"),
            tooltip=["x", "y"],
        )
    )

    st.altair_chart(chart, use_container_width=True)


# ======================================================
# 📌 CATATAN METODOLOGIS
# ======================================================
with st.expander("📌 Catatan Metodologis"):
    st.markdown(
        """
- Embedding dihasilkan menggunakan model Transformer (SBERT / IndoBERT).
- UMAP digunakan **hanya untuk visualisasi**, bukan dasar clustering.
- Evaluasi & clustering dilakukan di ruang embedding asli atau PCA.
- Viewer ini berfungsi sebagai **sanity check** dan bahan dokumentasi tesis.
"""
    )
