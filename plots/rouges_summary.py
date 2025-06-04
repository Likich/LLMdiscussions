import os
import pandas as pd

BASE_DIR = "./plots"
subdirs = [d for d in os.listdir(BASE_DIR) if d.startswith("new_extracted_") and os.path.isdir(os.path.join(BASE_DIR, d))]

print("📊 ROUGE-L Final Round Summary per Folder\n")

for subdir in sorted(subdirs):
    csv_path = os.path.join(BASE_DIR, subdir, "rouge_scores.csv")
    if not os.path.exists(csv_path):
        print(f"❌ {subdir}: Missing rouge_scores.csv")
        continue

    try:
        df = pd.read_csv(csv_path)
        if "ROUGE-L" not in df.columns:
            print(f"⚠️ {subdir}: No ROUGE-L column found.")
            continue

        final_rows = df[df["Round"] == "final"]
        if final_rows.empty:
            print(f"🤷 {subdir}: No final round rows found.")
            continue

        avg_rouge_l = final_rows["ROUGE-L"].mean()
        max_rouge_l = final_rows["ROUGE-L"].max()

        print(f"{subdir:<50}  Avg: {avg_rouge_l:.4f}   Max: {max_rouge_l:.4f}")

    except Exception as e:
        print(f"💥 {subdir}: Error reading file — {e}")
