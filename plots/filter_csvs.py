# safe_csv_filter.py
import pandas as pd
import os
import sys

prompt = sys.argv[1]  # 1–5 or 'all'
INPUT_DIR = "../csv_files/new_extracted"
OUTPUT_DIR = "./filtered_csvs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    if not file.endswith("_cleaned.csv"):
        continue

    path = os.path.join(INPUT_DIR, file)
    df = pd.read_csv(path)

    if prompt.lower() == "all":
        df_filtered = df
    else:
        index = int(prompt) - 1
        df_filtered = df.iloc[index::5].reset_index(drop=True)

    output_path = os.path.join(OUTPUT_DIR, file)
    df_filtered.to_csv(output_path, index=False)

print(f"Filtered CSVs saved to {OUTPUT_DIR} for prompt {prompt}")
