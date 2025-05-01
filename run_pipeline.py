import sys
sys.path.insert(0, '/Users/sudhinkarki/Desktop/Hackathon2/klqe/pipeline')

# from clustering import run_category_aware_clustering
# from outdated import detect_outdated
# from suggest import generate_suggestions
# import pandas as pd
# import pickle

# def run_full_pipeline(input_path="data/answer_library.csv"):
#     df = pd.read_csv(input_path)
    
#     # Run pipeline components
#     df = run_category_aware_clustering(df)
#     outdated_df = detect_outdated(df)
#     suggestion_df = generate_suggestions(df)

#     # Merge suggestions into main df on 'id'
#     df = df.merge(suggestion_df[["id", "suggestion"]], on="id", how="left")

#     # Save everything
#     df.to_csv("data/answer_library_with_clusters.csv", index=False)
#     suggestion_df.to_csv("data/suggestions.csv", index=False)

#     with open("data/duplicates.pkl", "wb") as f:
#         pickle.dump(df[df["cluster_id"] != -1], f)
#     with open("data/outdated.pkl", "wb") as f:
#         pickle.dump(outdated_df, f)

#     print("Pipeline run complete!")

# if __name__ == "__main__":
#     run_full_pipeline()



# import pandas as pd
# import pickle
# import os

# from run_clustering import run_rich_clustering
# from compute_embeddings import compute_and_save_embeddings
# from outdated import detect_outdated
# from suggest import generate_suggestions

# def run_full_pipeline(input_path="data/cleaned_duplicates_product_question_details.csv"):
#     if not os.path.exists(input_path):
#         raise FileNotFoundError(f"Input file not found: {input_path}")

#     # Load deduplicated data
#     df = pd.read_csv(input_path)

#     # Compute embeddings and save to disk
#     print("🔄 Computing embeddings...")
#     compute_and_save_embeddings(df)

#     # Run rich clustering on all fields
#     print("🔄 Running clustering...")
#     df = run_rich_clustering(df)

    # Detect outdated entries
#     print("🔍 Detecting outdated entries...")
#     outdated_df = detect_outdated(df)

#     # Generate suggestions (merge/review/archive)
#     print("💡 Generating suggestions...")
#     suggestion_df = generate_suggestions(df)

#     # Merge suggestions into main df on 'cqid'
#     df = df.merge(suggestion_df[["cqid", "suggestion"]], on="cqid", how="left")

#     # Save outputs
#     df.to_csv("data/answer_library_with_clusters.csv", index=False)
#     suggestion_df.to_csv("data/suggestions.csv", index=False)

#     # Save useful intermediate data
#     with open("data/duplicates.pkl", "wb") as f:
#         pickle.dump(df[df["cluster_id"] != -1], f)

#     with open("data/outdated.pkl", "wb") as f:
#         pickle.dump(outdated_df, f)

#     print("✅ Full pipeline complete!")

# if __name__ == "__main__":
#     run_full_pipeline()



# from compute_embeddings import compute_or_load_embeddings
# from clustering import cluster_or_load
# from outdated import detect_outdated
# from suggest import generate_suggestions
# import pandas as pd

# if __name__ == "__main__":
#     df = pd.read_csv("data/cleaned_duplicates_product_question_details.csv")
    
#     embeddings = compute_or_load_embeddings(df)
#     df_clustered = cluster_or_load(df, embeddings)
#     df_outdated = detect_outdated(df_clustered)
#     df_suggested = generate_suggestions(df_outdated)

#     df_suggested.to_csv("data/final_suggestions.csv", index=False)
#     print("✅ Pipeline complete. Suggestions saved.")


from compute_embeddings import compute_or_load_embeddings
from clustering import cluster_or_load
from outdated import detect_outdated
from suggest import generate_suggestions
import pandas as pd

from apscheduler.schedulers.blocking import BlockingScheduler




def main():
    df = pd.read_csv("data/cleaned_duplicates_product_question_details.csv")
        
        # Compute or load embeddings
    embeddings = compute_or_load_embeddings(df)
        
        # Cluster the data and save the resulting dataframe with cluster_id
    df_clustered = cluster_or_load(df, embeddings)
    df_clustered.to_csv("data/answer_library_with_clusters.csv", index=False)
    
    # Detect outdated entries (this is just for logging or tracking outdated entries)
    df_outdated = detect_outdated(df_clustered)
    
    # Generate suggestions using the full clustered dataframe (not just outdated)
    df_suggested = generate_suggestions(df_clustered, embeddings)
    
    # Save the suggested actions to a new CSV
    df_suggested.to_csv("data/final_suggestions.csv", index=False)
    
    print("✅ Pipeline complete. Suggestions saved to data/final_suggestions.csv.")        


    
scheduler = BlockingScheduler()
scheduler.add_job(main, 'interval', hours=1000)
scheduler.start()
    