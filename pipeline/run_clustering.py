import os
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

FIELDS = ['product_name', 'category', 'question', 'answer', 'details']
EMB_DIR = "data/embeddings"
OUTPUT_PATH = "data/answer_library_with_clusters.csv"

def run_rich_clustering(df, eps=0.45, min_samples=2):
    all_embeddings = []

    for field in FIELDS:
        path = os.path.join(EMB_DIR, f"{field}_embeddings.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing embedding: {path}")
        emb = np.load(path)
        if emb.shape[0] != len(df):
            raise ValueError(f"Mismatch in rows for {field}: {emb.shape[0]} vs {len(df)}")
        all_embeddings.append(emb)

    # Combine all embeddings
    combined_embs = np.concatenate(all_embeddings, axis=1)
    cluster_ids = np.full(len(df), -1)
    next_cluster_id = 0

    for category in df["category"].dropna().unique():
        mask = df["category"] == category
        indices = np.where(mask)[0]
        subset_embs = combined_embs[mask]

        if len(subset_embs) < min_samples:
            continue

        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
        labels = dbscan.fit_predict(subset_embs)

        for i, label in enumerate(labels):
            if label != -1:
                cluster_ids[indices[i]] = next_cluster_id + label

        if (labels != -1).any():
            next_cluster_id = cluster_ids.max() + 1

    df["cluster_id"] = cluster_ids
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Saved clustered DataFrame: {OUTPUT_PATH}")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/cleaned_deduplicated_input.csv")
    run_rich_clustering(df)
