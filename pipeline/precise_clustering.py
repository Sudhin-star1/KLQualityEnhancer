import sys
sys.path.insert(0, '/Users/sudhinkarki/Desktop/Hackathon2/klqe')

import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
import hdbscan

# ——————————————
# CONFIGURATION
# ——————————————
INPUT_CSV   = "data/cleaned_duplicates_product_question_details.csv"      # your pre-deduped data
OUTPUT_CSV  = "data/answer_library_with_clusters.csv"
SUGGEST_CSV = "data/suggestions.csv"

MODEL_NAME  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MIN_CLUSTER = 2          # HDBSCAN min cluster size
MERGE_THRESH = 0.85      # cosine-sim threshold for merge
CATEGORY_BOOST = 0.3     # add to distance if different category

# ——————————————
# STEP 1: LOAD & CLEAN
# ——————————————
df = pd.read_csv(INPUT_CSV)
df = df[df['question'].notna()]  # drop rows w/o question

def clean_text(s):
    return str(s).strip().lower()

texts = (df['question'].fillna('') + ' ' + df['details'].fillna('')).apply(clean_text).tolist()

# ——————————————
# STEP 2: EMBEDDINGS
# ——————————————
model = SentenceTransformer(MODEL_NAME)
embs = model.encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)

# ——————————————
# STEP 3: DISTANCE MATRIX + CATEGORY BOOST
# ——————————————
dist = cosine_distances(embs)
if 'category' in df.columns:
    cats = df['category'].fillna(" ").tolist()
    for i in range(len(cats)):
        for j in range(len(cats)):
            if cats[i] != cats[j]:
                dist[i,j] += CATEGORY_BOOST

# ——————————————
# STEP 4: HDBSCAN CLUSTERING
# ——————————————
clusterer = hdbscan.HDBSCAN(
    metric="precomputed",
    min_cluster_size=MIN_CLUSTER,
    min_samples=1,
    cluster_selection_method="eom"
)
labels = clusterer.fit_predict(dist)
df['cluster_id'] = labels

# Save clustered DataFrame
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Wrote clustered data → {OUTPUT_CSV}")

# ——————————————
# STEP 5: HIGH-PRECISION MERGE SUGGESTIONS
# ——————————————
suggestions = []
for cid in set(labels):
    if cid == -1: 
        continue
    members = df[df['cluster_id'] == cid].reset_index(drop=True)
    if len(members) < 2:
        continue

    sub_embs = embs[members.index]
    sim = cosine_similarity(sub_embs)

    # for each pair above threshold, suggest merge
    for i in range(len(members)):
        for j in range(i+1, len(members)):
            score = float(sim[i,j])
            if score >= MERGE_THRESH:
                # pick newer (or any heuristic) as target
                newer = members.iloc[i] if pd.to_datetime(members.iloc[i]['created_at']) >= pd.to_datetime(members.iloc[j]['created_at']) else members.iloc[j]
                older = members.iloc[j] if newer is members.iloc[i] else members.iloc[i]
                suggestions.append({
                    'cqid': older['cqid'],
                    'action': 'merge',
                    'target_cqid': newer['cqid'],
                    'cluster_id': cid,
                    'similarity': round(score,3)
                })

# Save suggestions
pd.DataFrame(suggestions).drop_duplicates(subset=['cqid','target_cqid']).to_csv(SUGGEST_CSV, index=False)
print(f"✅ Wrote merge suggestions → {SUGGEST_CSV}")
