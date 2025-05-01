import pandas as pd
import sys

def process_duplicates(input_file):
    # Load input
    df = pd.read_excel(input_file)

    # Ensure required columns exist
    required = ['product_id', 'question', 'answer', 'details', 'created_at', 'deleted_at']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert date columns
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df['deleted_at'] = pd.to_datetime(df['deleted_at'], errors='coerce')

    # Clean and filter
    df = df[df['question'].notna() & (df['question'].str.strip() != "")]
    df['answer'] = df['answer'].fillna('').str.strip()
    df['details'] = df['details'].fillna('').str.strip()
    df['is_outdated'] = df['deleted_at'].notna()

    # Define groupings
    configs = [
        {'group_cols': ['product_id', 'question'], 'suffix': 'product_question'},
        {'group_cols': ['product_id', 'question', 'answer'], 'suffix': 'product_question_answer'},
        {'group_cols': ['product_id', 'question', 'details'], 'suffix': 'product_question_details'},
        {'group_cols': ['product_id', 'question', 'answer', 'details'], 'suffix': 'product_question_answer_details'}
    ]

    # Process each configuration
    for conf in configs:
        print(f"Processing: {conf['suffix']}")

        # Calculate duplicate counts before deduplication
        duplicate_counts = (
            df.groupby(conf['group_cols']).size().reset_index(name='duplicate_count')
        )

        # Merge count back into original DataFrame
        temp_df = df.merge(duplicate_counts, on=conf['group_cols'], how='left')

        # Sort and deduplicate (keep most recent)
        temp_df = temp_df.sort_values('created_at', ascending=False).drop_duplicates(
            subset=conf['group_cols'], keep='first'
        )

        # Save
        filename = f"cleaned_duplicates_{conf['suffix']}.csv"
        temp_df.to_csv(filename, index=False)
        print(f"Saved {filename} ({len(temp_df)} records), "
              f"with max duplicates in group: {temp_df['duplicate_count'].max()}")

if _name_ == "_main_":
    if len(sys.argv) < 2:
        print("Usage: python script.py <input_file.xlsx>")
        exit(1)

    input_file = sys.argv[1]
    process_duplicates(input_file)
    print("✅ All product-specific duplicate files generated successfully")