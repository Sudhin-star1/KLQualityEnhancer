# import pandas as pd

# def generate_suggestions(df):
#     df["suggestion"] = ""
#     df["target_id"] = ""  # For merge suggestion, indicate which cqid to merge with

#     # Suggestion 1: MERGE — based on cluster ID
#     cluster_groups = df[df["cluster_id"] != -1].groupby("cluster_id")

#     for cluster_id, group in cluster_groups:
#         if len(group) > 1:
#             sorted_group = group.sort_values("created_at")
#             main_id = sorted_group.iloc[0]["cqid"]

#             for _, row in sorted_group.iterrows():
#                 if row["cqid"] != main_id:
#                     df.loc[df["cqid"] == row["cqid"], "suggestion"] = "merge"
#                     df.loc[df["cqid"] == row["cqid"], "target_id"] = main_id

#     # Suggestion 2: ARCHIVE — if deleted and in a duplicate cluster
#     mask_deleted = df["deleted_at"].notna() & (df["cluster_id"] != -1)
#     df.loc[mask_deleted, "suggestion"] = "archive"

#     # Suggestion 3: REVIEW — poor or missing answers
#     bad_answers = ["", "n/a", "none", "null"]
#     df["answer"] = df["answer"].fillna("").astype(str).str.strip().str.lower()
#     mask_review = df["answer"].isin(bad_answers)
#     df.loc[mask_review, "suggestion"] = "review"

#     # Only return actionable suggestions
#     suggested_df = df[df["suggestion"] != ""].copy()
#     return suggested_df


# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np

# def generate_suggestions(df, emb_path="artifacts/embeddings.npy"):
#     embeddings = np.load(emb_path)
#     suggestions = []

#     for cluster_id in df["cluster"].unique():
#         if cluster_id == -1:  # Skip noise
#             continue

#         cluster_df = df[df["cluster"] == cluster_id]
#         cluster_indices = cluster_df.index.tolist()
#         cluster_embeddings = embeddings[cluster_indices]

#         cos_sim = cosine_similarity(cluster_embeddings)
#         for i in range(len(cluster_indices)):
#             for j in range(i + 1, len(cluster_indices)):
#                 idx1, idx2 = cluster_indices[i], cluster_indices[j]
#                 row1, row2 = df.loc[idx1], df.loc[idx2]

#                 same_product = row1["product_id"] == row2["product_id"]
#                 same_category = row1["category"] == row2["category"]
#                 similarity = cos_sim[i, j]

#                 if similarity > 0.92 and same_product:
#                     suggestions.append({
#                         "action": "merge",
#                         "cqid": row1["cqid"],
#                         "target_cqid": row2["cqid"],
#                         "cluster": cluster_id
#                     })

#     for idx, row in df.iterrows():
#         if row["is_outdated"]:
#             suggestions.append({
#                 "action": "archive",
#                 "cqid": row["cqid"],
#                 "target_cqid": None,
#                 "cluster": row["cluster"]
#             })

#     # Remove duplicates and format
#     return pd.DataFrame(suggestions).drop_duplicates()



# suggest.py
# suggest.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def is_outdated(row):
    answer = str(row.get("answer", "")).lower()
    return (
        pd.notnull(row.get("deleted_at")) or
        "no longer" in answer or "deprecated" in answer
    )

def is_vague_answer(answer):
    answer = str(answer).strip().lower()
    return answer in {"yes", "no", "maybe", "depends", "nan", ""}

def generate_suggestions(df, embeddings, 
                         merge_thresh=0.92, review_thresh=0.80):
    suggestions = []
    df = df.copy()
    df["is_outdated"] = df.apply(is_outdated, axis=1)

    # Group by the cluster column already in df
    for cluster_id, group in df[df["cluster"] != -1].groupby("cluster"):
        idxs = group.index.tolist()
        cluster_embs = embeddings[idxs]
        sim_matrix = cosine_similarity(cluster_embs)

        for i in range(len(idxs)):
            for j in range(i + 1, len(idxs)):
                r1, r2 = df.loc[idxs[i]], df.loc[idxs[j]]
                sim = sim_matrix[i, j]

                same_prod = r1["product_id"] == r2["product_id"]
                same_cat  = r1["category"] == r2["category"]
                vague     = is_vague_answer(r1["answer"]) or is_vague_answer(r2["answer"])

                if sim >= merge_thresh and same_prod and same_cat:
                    suggestions.append({
                        "action": "merge",
                        "cqid": r1["cqid"],
                        "target_cqid": r2["cqid"],
                        "cluster": cluster_id
                    })
                elif review_thresh < sim < merge_thresh and same_prod:
                    suggestions.append({
                        "action": "review",
                        "cqid": r1["cqid"],
                        "target_cqid": r2["cqid"],
                        "cluster": cluster_id
                    })
                elif vague and same_prod:
                    suggestions.append({
                        "action": "review",
                        "cqid": r1["cqid"],
                        "target_cqid": r2["cqid"],
                        "cluster": cluster_id
                    })

    # Archive suggestions
    for _, row in df.iterrows():
        if row["is_outdated"]:
            suggestions.append({
                "action": "archive",
                "cqid": row["cqid"],
                "target_cqid": "",
                "cluster": row["cluster"]
            })

    return pd.DataFrame(suggestions).drop_duplicates()


# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def is_outdated(row):
#     answer = str(row.get("answer", "")).lower()
#     return (
#         pd.notnull(row.get("deleted_at")) or
#         "no longer" in answer or "deprecated" in answer
#     )

# def is_vague_answer(answer):
#     answer = str(answer).strip().lower()
#     return answer in {"yes", "no", "maybe", "depends", "nan", ""}

# def generate_suggestions(df, embeddings, merge_thresh=0.92, review_thresh=0.80):
#     suggestions = []
#     df = df.copy()
#     df["is_outdated"] = df.apply(is_outdated, axis=1)

#     # Group by valid clusters
#     for cluster_id, group in df[df["cluster"] != -1].groupby("cluster"):
#         idxs = group.index.tolist()
#         cluster_embs = embeddings[idxs]
#         sim_matrix = cosine_similarity(cluster_embs)

#         # Check all pairwise similarities
#         pairwise_sims = [
#             sim_matrix[i, j]
#             for i in range(len(idxs))
#             for j in range(i + 1, len(idxs))
#         ]

#         avg_sim = np.mean(pairwise_sims) if pairwise_sims else 0.0
#         products = group["product_id"].nunique()
#         categories = group["category"].nunique()
#         vague_count = group["answer"].apply(is_vague_answer).sum()

#         cqids = group["cqid"].tolist()

#         # Suggest Merge if highly similar and same product+category
#         if avg_sim >= merge_thresh and products == 1 and categories == 1:
#             suggestions.append({
#                 "action": "merge",
#                 "cqids": cqids,
#                 "target_cqid": cqids[0],  # optional anchor
#                 "cluster": cluster_id
#             })

#         # Suggest Review if moderately similar but same product
#         elif avg_sim >= review_thresh and products == 1:
#             suggestions.append({
#                 "action": "review",
#                 "cqids": cqids,
#                 "target_cqid": "",
#                 "cluster": cluster_id
#             })

#         # Suggest Review if too many vague answers in same product
#         elif vague_count >= 1 and products == 1:
#             suggestions.append({
#                 "action": "review",
#                 "cqids": cqids,
#                 "target_cqid": "",
#                 "cluster": cluster_id
#             })

#     # Archive individually
#     for _, row in df.iterrows():
#         if row["is_outdated"]:
#             suggestions.append({
#                 "action": "archive",
#                 "cqids": [row["cqid"]],
#                 "target_cqid": "",
#                 "cluster": row["cluster"]
#             })

#     return pd.DataFrame(suggestions)



# # suggest.py

# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import cosine_similarity

# def is_outdated(row):
#     ans = str(row.get("answer", "")).lower()
#     return pd.notnull(row.get("deleted_at")) or any(k in ans for k in ("no longer","deprecated"))

# def is_vague(ans):
#     return str(ans).strip().lower() in {"yes","no","maybe","depends","","nan"}

# def generate_suggestions(df, embeddings, merge_thresh=0.92, review_thresh=0.80):
#     """
#     Returns one suggestion per cluster (action, list of cqids, cluster_id,…).
#     """
#     df = df.copy()
#     df["is_outdated"] = df.apply(is_outdated, axis=1)
#     suggestions = []

#     # ---------- cluster‐level merge/review ----------
#     for cluster_id, group in df[df["cluster_id"] != -1].groupby("cluster_id"):
#         cqids = group["cqid"].tolist()
#         if len(cqids) < 2:
#             continue

#         # average cosine‐similarity
#         sims = cosine_similarity(embeddings[group.index])
#         pairwise = [sims[i,j]
#                     for i in range(len(cqids)) 
#                     for j in range(i+1, len(cqids))]
#         avg_sim = float(np.mean(pairwise))

#         prods = group["product_id"].nunique()
#         cats  = group["category"].nunique()
#         vague = group["answer"].apply(is_vague).any()

#         if avg_sim >= merge_thresh and prods == 1 and cats == 1:
#             action = "merge"
#         elif avg_sim >= review_thresh and prods == 1:
#             action = "review"
#         elif vague and prods == 1:
#             action = "review"
#         else:
#             continue

#         suggestions.append({
#             "action": action,
#             "cqids": cqids,
#             "cluster_id": cluster_id
#         })

#     # ---------- individual archive ----------
#     for _, row in df[df["is_outdated"]].iterrows():
#         suggestions.append({
#             "action": "archive",
#             "cqids": [row["cqid"]],
#             "cluster_id": row["cluster_id"]
#         })

#     return pd.DataFrame(suggestions)
