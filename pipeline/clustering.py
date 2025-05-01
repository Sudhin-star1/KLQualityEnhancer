# import pandas as pd
# import numpy as np
# import hdbscan
# import pickle
# from sklearn.preprocessing import LabelEncoder


# def cluster_embeddings(
#     df: pd.DataFrame,
#     embeddings: np.ndarray,
#     min_cluster_size: int = 2,
#     category_aware: bool = True
# ) -> pd.DataFrame:
#     """
#     Cluster the QnA entries using HDBSCAN.

#     Args:
#         df (pd.DataFrame): Knowledge base with question/answer/etc.
#         embeddings (np.ndarray): Combined question+details embeddings.
#         min_cluster_size (int): Minimum size of a cluster.
#         category_aware (bool): If True, run clustering separately for each category.

#     Returns:
#         pd.DataFrame: Input DataFrame with new `cluster_id` column added.
#     """
#     df = df.copy()
#     df["cluster_id"] = -1  # Initialize with -1 (noise)

#     if category_aware:
#         cluster_counter = 0
#         for category in df["category"].dropna().unique():
#             mask = df["category"] == category
#             sub_df = df[mask]
#             sub_embeddings = embeddings[mask]

#             clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
#             labels = clusterer.fit_predict(sub_embeddings)

#             # Offset cluster IDs to avoid overlaps
#             labels_offset = [label + cluster_counter + 1 if label != -1 else -1 for label in labels]
#             df.loc[mask, "cluster_id"] = labels_offset

#             cluster_counter += max(labels) + 1 if len(labels) > 0 else 0
#     else:
#         clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
#         labels = clusterer.fit_predict(embeddings)
#         df["cluster_id"] = labels

#     return df


# def save_duplicates(df_with_clusters: pd.DataFrame, output_path: str = "data/duplicates.pkl") -> None:
#     """
#     Save the clustered duplicates only.

#     Args:
#         df_with_clusters (pd.DataFrame): DataFrame with `cluster_id`.
#         output_path (str): Output pickle file path.
#     """
#     duplicates = df_with_clusters[df_with_clusters["cluster_id"] != -1].copy()
#     with open(output_path, "wb") as f:
#         pickle.dump(duplicates, f)

# import os
# import pandas as pd
# import numpy as np
# from sklearn.cluster import DBSCAN

# EMB_Q_PATH = "data/embeddings/question_embeddings.npy"
# EMB_D_PATH = "data/embeddings/detail_embeddings.npy"

# def run_category_aware_clustering(df, eps=0.45, min_samples=2):
#     # Ensure embeddings exist
#     if not os.path.exists(EMB_Q_PATH) or not os.path.exists(EMB_D_PATH):
#         raise FileNotFoundError("Embeddings not found. Please run embedding computation first.")

#     # Load precomputed embeddings
#     question_embs = np.load(EMB_Q_PATH)
#     detail_embs = np.load(EMB_D_PATH)

#     if question_embs.shape[0] != df.shape[0] or detail_embs.shape[0] != df.shape[0]:
#         raise ValueError("Mismatch between embedding count and number of rows in DataFrame.")

#     # Combine embeddings
#     combined_embs = np.concatenate([question_embs, detail_embs], axis=1)

#     # Initialize
#     df["cluster_id"] = -1
#     all_labels = np.full(len(df), -1)

#     for category in df["category"].unique():
#         cat_mask = df["category"] == category
#         subset_embs = combined_embs[cat_mask]

#         if len(subset_embs) >= min_samples:
#             dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
#             labels = dbscan.fit_predict(subset_embs)

#             max_id = all_labels.max()
#             labels = [l + max_id + 1 if l != -1 else -1 for l in labels]
#             all_labels[cat_mask] = labels

#     # Assign final cluster IDs
#     df["cluster_id"] = all_labels
#     df.to_csv("data/answer_library_with_clusters.csv", index=False)
#     return df


# import os
# import hashlib
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer
# import hdbscan

# # --- Setup ---
# model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight & fast
# CACHE_DIR = "data/cache"
# os.makedirs(CACHE_DIR, exist_ok=True)

# # --- Enrich input text with product and category info ---
# def enriched_text(row):
#     return f"[PRODUCT: {row['product_name']}] [CATEGORY: {row['category']}] {row['question']} {row['details']}"

# # --- Compute or load cached embeddings ---
# def get_cache_path(text: str) -> str:
#     key = hashlib.md5(text.encode()).hexdigest()
#     return os.path.join(CACHE_DIR, f"{key}.npy")

# def embed_rows(df: pd.DataFrame) -> np.ndarray:
#     embeddings = []
#     for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding rows"):
#         text = enriched_text(row)
#         cache_path = get_cache_path(text)
#         if os.path.exists(cache_path):
#             emb = np.load(cache_path)
#         else:
#             emb = model.encode(text)
#             np.save(cache_path, emb)
#         embeddings.append(emb)
#     return np.vstack(embeddings)

# # --- Run HDBSCAN clustering ---
# def run_hdbscan_clustering(df: pd.DataFrame, min_cluster_size: int = 2, min_samples: int = 1) -> pd.DataFrame:
#     embs = embed_rows(df)

#     clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric='euclidean')
#     cluster_labels = clusterer.fit_predict(embs)

#     df = df.copy()
#     df["cluster"] = cluster_labels
#     return df


import os
import pandas as pd
import numpy as np
import joblib
import hdbscan
from sklearn.preprocessing import StandardScaler

def cluster_or_load(df, embeddings, path="artifacts/cluster_labels.npy"):
    if os.path.exists(path):
        df["cluster"] = np.load(path)
        return df

    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings_scaled)

    os.makedirs("artifacts", exist_ok=True)
    np.save(path, cluster_labels)
    
    df["cluster"] = cluster_labels
    return df


# import os
# import numpy as np
# import pandas as pd
# import joblib
# import hdbscan
# from sklearn.preprocessing import StandardScaler

# def cluster_or_load(df, embeddings, path="artifacts/cluster_labels.npy"):
#     # If we already have saved labels, load and assign
#     if os.path.exists(path):
#         df["cluster_id"] = np.load(path)
#         return df

#     # Otherwise compute new clustering
#     scaler = StandardScaler()
#     X = scaler.fit_transform(embeddings)

#     clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
#     labels = clusterer.fit_predict(X)

#     os.makedirs(os.path.dirname(path), exist_ok=True)
#     np.save(path, labels)

#     # **Rename** to cluster_id for consistency downstream
#     df["cluster_id"] = labels
#     return df
