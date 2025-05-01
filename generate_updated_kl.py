import pandas as pd
import os
import pickle
from datetime import datetime

def generate_updated_kl():
    # Load original KL
    kl_df = pd.read_csv("data/answer_library.csv")

    # Load accepted suggestions
    accepted_df = pd.read_csv("data/accepted_suggestions.csv")

    # Load duplicate clusters (optional merge cleanup)
    with open("data/duplicates.pkl", "rb") as f:
        duplicates_df = pickle.load(f)

    kl_df['suggestion_applied'] = ''

    merged_ids = set()

    for _, row in accepted_df.iterrows():
        qid = row['question_id']
        action = str(row['suggestion']).lower()
        accepted = row['accepted']

        if not accepted or qid not in kl_df['id'].values:
            continue

        idx = kl_df[kl_df['id'] == qid].index[0]

        if action == "archive":
            if pd.isna(kl_df.at[idx, 'deleted_at']):
                kl_df.at[idx, 'deleted_at'] = datetime.now().strftime("%Y-%m-%d")
            kl_df.at[idx, 'suggestion_applied'] = 'archive'

        elif action == "review":
            kl_df.at[idx, 'suggestion_applied'] = 'review'

        elif action == "merge":
            # Tag it and remove its duplicates
            kl_df.at[idx, 'details'] = f"[MERGE REQUIRED] {kl_df.at[idx, 'details']}"
            kl_df.at[idx, 'suggestion_applied'] = 'merge'

            # Get its cluster group and drop others from the same group
            if 'group_id' in duplicates_df.columns:
                group = duplicates_df[duplicates_df['id'] == qid]['group_id'].values
                if len(group) > 0:
                    gid = group[0]
                    group_members = duplicates_df[duplicates_df['group_id'] == gid]
                    merged_ids.update(set(group_members['id'].values) - {qid})

    # Remove merged duplicate questions
    kl_df = kl_df[~kl_df['id'].isin(merged_ids)]

    # Versioned saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"data/knowledge_library_v{timestamp}.csv"
    kl_df.to_csv(out_path, index=False)
    print(f"✅ Updated Knowledge Library Saved: {out_path}")

    return out_path  # For UI

if __name__ == "__main__":
    generate_updated_kl()
