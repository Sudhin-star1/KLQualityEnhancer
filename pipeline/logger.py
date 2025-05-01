import pandas as pd
import os

LOG_PATH = "data/accepted_suggestions.csv"

def log_user_decision(question_id, action, accepted):
    entry = pd.DataFrame([{
        "question_id": question_id,
        "action": action,
        "accepted": accepted
    }])

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, entry], ignore_index=True)
    else:
        df = entry
    df.to_csv(LOG_PATH, index=False)
