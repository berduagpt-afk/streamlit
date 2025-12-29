import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# 1) Koneksi database
# --------------------------------------------------
engine = create_engine(
    "postgresql+psycopg2://postgres:admin*123@localhost:5432/incident_djp"
)

# --------------------------------------------------
# 2) Ambil kolom teks dari tabel
# --------------------------------------------------
sql = """
SELECT text_sintaksis
FROM lasis_djp.incident_clean
WHERE text_sintaksis IS NOT NULL
"""

df = pd.read_sql(text(sql), engine)

texts = df["text_sintaksis"].astype(str).tolist()

# --------------------------------------------------
# 3) Fit TF-IDF (TANPA transform)
# --------------------------------------------------
vectorizer = TfidfVectorizer(
    min_df=2,       # sesuaikan dengan modeling
    max_df=0.95,
    ngram_range=(1, 2),
    lowercase=False
)

vectorizer.fit(texts)

# --------------------------------------------------
# 4) Jumlah feature
# --------------------------------------------------
n_features = len(vectorizer.get_feature_names_out())

print(f"Jumlah feature TF-IDF: {n_features:,}")
