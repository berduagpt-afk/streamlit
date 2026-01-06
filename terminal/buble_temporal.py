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

job_id = "ac260bf3-bf1e-44d9-8edf-a0b9043bcc67"
modeling_id = "76d28fd4-ab3a-446e-8b07-be1ee1e27d9e"
window_days = 7  # 7 / 14 / 30

# =========================
# 2) LOAD DATA
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
df = pd.read_sql(sql, engine, params={"job_id": job_id, "modeling_id": modeling_id, "window_days": window_days})
df["tgl_submit"] = pd.to_datetime(df["tgl_submit"])

if df.empty:
    raise RuntimeError("Data temporal kosong untuk parameter yang dipilih.")

# =========================
# 3) AGGREGATE: episode per cluster (start-end + size)
# =========================
ep = (
    df.groupby(["cluster_id", "temporal_cluster_no"], as_index=False)
      .agg(
          start=("tgl_submit", "min"),
          end=("tgl_submit", "max"),
          size=("incident_number", "count"),
      )
)

# titik bubble di tengah episode
ep["mid"] = ep["start"] + (ep["end"] - ep["start"]) / 2

# =========================
# 4) OPTIONAL: batasi jumlah cluster biar plot tidak terlalu ramai
#    Misal tampilkan 30 cluster terbesar (berdasarkan total tiket)
# =========================
top_n = 30
cluster_sizes = df.groupby("cluster_id")["incident_number"].count().sort_values(ascending=False)
keep_clusters = cluster_sizes.head(top_n).index
ep = ep[ep["cluster_id"].isin(keep_clusters)].copy()

# urutkan cluster_id agar rapi (cluster besar di atas)
cluster_order = cluster_sizes.loc[keep_clusters].sort_values(ascending=True).index.tolist()
y_map = {cid: i for i, cid in enumerate(cluster_order)}
ep["y"] = ep["cluster_id"].map(y_map)

# =========================
# 5) SCALING ukuran bubble (size -> marker area)
# =========================
# Agar tidak terlalu besar/kecil, pakai scaling log atau sqrt
ep["bubble"] = np.sqrt(ep["size"])  # smooth
# konversi ke area marker (s)
min_s, max_s = 40, 900
b = ep["bubble"].to_numpy()
b_scaled = (b - b.min()) / (b.max() - b.min() + 1e-9)
ep["s"] = min_s + b_scaled * (max_s - min_s)

# =========================
# 6) PLOT: garis horizontal + bubble
# =========================
fig, ax = plt.subplots(figsize=(14, 7))

# garis episode (start-end) per cluster
for _, r in ep.iterrows():
    ax.hlines(
        y=r["y"],
        xmin=r["start"],
        xmax=r["end"],
        linewidth=3,
        alpha=0.7
    )

# bubble di tengah episode
ax.scatter(
    ep["mid"],
    ep["y"],
    s=ep["s"],
    alpha=0.55
)

# label y-axis = cluster_id
ax.set_yticks([y_map[c] for c in cluster_order])
ax.set_yticklabels([str(c) for c in cluster_order])

ax.set_xlabel("Tanggal")
ax.set_ylabel("cluster_id")
ax.set_title(f"Timeline Episode Temporal per Cluster (window_days={window_days})\n"
             f"Bubble size = jumlah tiket per episode")

# format tanggal
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.grid(True, axis="x", alpha=0.2)

plt.tight_layout()
plt.show()
