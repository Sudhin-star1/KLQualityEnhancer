import sys
sys.path.insert(0, '/Users/sudhinkarki/Desktop/Hackathon2/klqe')

# import os
# import numpy as np
# from sentence_transformers import SentenceTransformer

# EMBEDDING_DIR = "data/embeddings"
# FIELDS = ["product_name", "category", "question", "answer", "details"]
# MODEL_NAME = "all-MiniLM-L6-v2"

# def compute_and_save_embeddings(df):
#     os.makedirs(EMBEDDING_DIR, exist_ok=True)
#     model = SentenceTransformer(MODEL_NAME)

#     for field in FIELDS:
#         save_path = os.path.join(EMBEDDING_DIR, f"{field}_embeddings.npy")

#         if os.path.exists(save_path):
#             print(f"✅ Cached embeddings for `{field}` found. Skipping...")
#             continue

#         print(f"🔄 Generating embeddings for `{field}`...")
#         texts = df[field].fillna("").astype(str).tolist()
#         embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)
#         np.save(save_path, embeddings)
#         print(f"💾 Saved embeddings to {save_path}")


import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import joblib

def compute_or_load_embeddings(df, path="artifacts/embeddings.npy"):
    if os.path.exists(path):
        return np.load(path)
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (df["question"].fillna('') + " " + df["details"].fillna('')).tolist()
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    
    os.makedirs("artifacts", exist_ok=True)
    np.save(path, embeddings)
    return embeddings
