import pandas as pd
from sqlalchemy import create_engine, text
from sklearn.feature_extraction.text import TfidfVectorizer

# --------------------------------------------------
# 1) Koneksi database
# --------------------------------------------------
engine = create_engine(
    "postgresql+psycopg2://postgres:admin*123@localhost:5432/incident_djp"
)

sql = """
SELECT modul, text_sintaksis
FROM lasis_djp.incident_clean
WHERE text_sintaksis IS NOT NULL
"""

df = pd.read_sql(text(sql), engine)

for modul, g in df.groupby("modul"):
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.95)
    vectorizer.fit(g["text_sintaksis"].astype(str).tolist())
    print(f"{modul:30s} → {len(vectorizer.get_feature_names_out()):6d} features")
