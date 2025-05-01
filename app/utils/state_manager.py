import pandas as pd
import sys
sys.path.insert(0, '/Users/sudhinkarki/Desktop/Hackathon2/klqe')

class StateManager:
    def __init__(self):
        # Load the data
        self.df = pd.read_csv("data/answer_library_with_clusters.csv")
        self.suggestions_df = pd.read_csv("data/suggestions.csv")

    def get_categories(self):
        return self.df["category"].unique().tolist()

    def get_filtered_questions(self, category_filter):
        if category_filter != "All":
            return self.df[self.df["category"] == category_filter].to_dict(orient="records")
        return self.df.to_dict(orient="records")

    def get_suggestions(self):
        return self.suggestions_df.to_dict(orient="records")

    def get_metrics(self):
        # Create simple metrics per category
        stats = {}
        for category in self.df["category"].unique():
            stats[category] = {
                "most_duplicates": self.df[self.df["category"] == category].shape[0],
                "most_outdated": self.df[self.df["category"] == category].shape[0],
            }
        return stats
