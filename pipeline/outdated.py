import pandas as pd

def detect_outdated(df):
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df["deleted_at"] = pd.to_datetime(df["deleted_at"], errors="coerce")
    return df[
        (df["created_at"] < pd.Timestamp("2021-01-01")) |
        (df["deleted_at"].notnull())
    ].copy()
