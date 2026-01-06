import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sqlalchemy import create_engine, text

# =========================
# 1) DB CONNECTION (sesuaikan)
# =========================
DB = {
    "host": "localhost",
    "port": 5432,
    "database": "incident_djp",
    "user": "postgres",
    "password": "admin*123",
}
engine = create_engine(
    f"postgresql+psycopg2://{DB['user']}:{DB['password']}@{DB['host']}:{DB['port']}/{DB['database']}",
    pool_pre_ping=True
)

SCHEMA = "lasis_djp"
T_TEMP = "modeling_sintaksis_temporal_members"

# =========================
# 2) PARAMS
# =========================
job_id = "ac260bf3-bf1e-44d9-8edf-a0b9043bcc67"
modeling_id = "76d28fd4-ab3a-446e-8b07-be1ee1e27d9e"
window_days = 14  # 7 / 14 / 30

# tampilkan N cluster saja biar terbaca (ambil cluster terbesar)
TOP_N_CLUSTERS = 30

# =========================
# 3) LOAD DATA (temporal members)
# =========================
sql = text(f"""
SELECT
  job_id, modeling_id, window_days,
  cluster_id,
  temporal_cluster_no,
  incident_number,
  tgl_submit
FROM {SCHEMA}.{T_TEMP}
WHERE job_id = CAST(:job_id AS uuid)
  AND modeling_id = CAST(:modeling_id AS uuid)
  AND window_days = :window_days
  AND tgl_submit IS NOT NULL
""")

df = pd.read_sql(
    sql, engine,
    params={"job_id": job_id, "modeling_id": modeling_id, "window_days": window_days}
)
df["tgl_submit"] = pd.to_datetime(df["tgl_submit"], errors="coerce")
df = df.dropna(subset=["tgl_submit"]).copy()

if df.empty:
    raise RuntimeError("Data temporal kosong untuk parameter yang dipilih.")

# =========================
# 4) EPISODE AGGREGATION
#    start-end per (cluster_id, temporal_cluster_no)
# =========================
ep = (
    df.groupby(["cluster_id", "temporal_cluster_no"], as_index=False)
      .agg(
          start=("tgl_submit", "min"),
          end=("tgl_submit", "max"),
          episode_size=("incident_number", "count"),
      )
)

# mid point bubble
ep["mid"] = ep["start"] + (ep["end"] - ep["start"]) / 2

# =========================
# 5) SELECT TOP N CLUSTERS (by total tickets)
# =========================
cluster_total = df.groupby("cluster_id")["incident_number"].count().sort_values(ascending=False)
keep = cluster_total.head(TOP_N_CLUSTERS).index
ep = ep[ep["cluster_id"].isin(keep)].copy()

# urut cluster: terbesar di atas (lebih enak dibaca)
cluster_order = cluster_total.loc[keep].sort_values(ascending=False).index.tolist()

# map to y positions (0..n-1)
y_map = {cid: i for i, cid in enumerate(cluster_order)}
ep["y"] = ep["cluster_id"].map(y_map)

# =========================
# 6) SCALE BUBBLE SIZE (episode_size -> marker area)
#    pakai sqrt agar stabil
# =========================
s_raw = np.sqrt(ep["episode_size"].to_numpy(dtype=float))
min_s, max_s = 60, 1200  # bisa diubah
s_scaled = (s_raw - s_raw.min()) / (s_raw.max() - s_raw.min() + 1e-9)
ep["s"] = min_s + s_scaled * (max_s - min_s)

# =========================
# 7) PLOT
# =========================
fig, ax = plt.subplots(figsize=(16, 8))

# (A) garis episode: dari start ke end
for _, r in ep.iterrows():
    ax.hlines(
        y=r["y"],
        xmin=r["start"],
        xmax=r["end"],
        linewidth=3,
        alpha=0.7
    )

# kalau start==end, garisnya jadi "titik", jadi kita tambah marker kecil
same_day = ep["start"] == ep["end"]
ax.scatter(ep.loc[same_day, "start"], ep.loc[same_day, "y"], s=30, alpha=0.8)

# (B) bubble di tengah episode: ukuran = episode_size
ax.scatter(
    ep["mid"],
    ep["y"],
    s=ep["s"],
    alpha=0.55
)

# y-axis labels = cluster_id
ax.set_yticks([y_map[c] for c in cluster_order])
ax.set_yticklabels([str(c) for c in cluster_order])

ax.set_xlabel("Tanggal Insiden (tgl_submit)")
ax.set_ylabel("cluster_id")
ax.set_title(
    f"Timeline Episode Temporal per Cluster (window_days={window_days})\n"
    "Garis = rentang episode (mulai–selesai), Bubble = jumlah tiket per episode"
)

# format tanggal
locator = mdates.AutoDateLocator()
ax.xaxis.set_major_locator(locator)
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))

ax.grid(True, axis="x", alpha=0.2)
plt.tight_layout()
plt.show()
