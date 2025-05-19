import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from rouge_score import rouge_scorer

CSV_DIR = os.path.join("..", "csv_files", "new_extracted")
PLOTS_DIR = "plots"
ROUNDS = ["initial", "R1", "R2", "R3", "R4", "R5", "final"]

def compute_rouge(text1, text2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(str(text1), str(text2))
    return scores['rouge1'].fmeasure, scores['rouge2'].fmeasure, scores['rougeL'].fmeasure

def detect_models(df):
    model_names = ["Maverick", "Llama3.3", "Deepseek", "Gemma", "Mistral"]
    return [m for m in model_names if any(col.startswith(m) for col in df.columns)]

def get_column_name(model, round_label):
    if round_label == "initial":
        return model
    elif round_label == "final":
        return f"{model} Final"
    else:
        return f"{model} {round_label}"

def calculate_rouge_for_file(file_path, output_dir):
    df = pd.read_csv(file_path)
    file_base = os.path.basename(file_path).replace(".csv", "")

    models = detect_models(df)
    if len(models) < 2:
        print(f"⏭ Skipping {file_base} — only {len(models)} model(s) found.")
        return

    rouge_results = []

    for round_label in ROUNDS:
        col_names = [get_column_name(m, round_label) for m in models]
        existing_cols = [col for col in col_names if col in df.columns]

        if len(existing_cols) < 2:
            continue

        df_selected = df[["Comment"] + existing_cols].dropna()

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                model1 = models[i]
                model2 = models[j]
                col1 = get_column_name(model1, round_label)
                col2 = get_column_name(model2, round_label)

                if col1 not in df_selected.columns or col2 not in df_selected.columns:
                    continue

                texts1 = df_selected[col1].astype(str).tolist()
                texts2 = df_selected[col2].astype(str).tolist()

                scores = [compute_rouge(t1, t2) for t1, t2 in zip(texts1, texts2)]
                rouge1, rouge2, rougeL = np.mean(scores, axis=0)

                rouge_results.append({
                    "Round": round_label,
                    "Model1": model1,
                    "Model2": model2,
                    "ROUGE-1": rouge1,
                    "ROUGE-2": rouge2,
                    "ROUGE-L": rougeL
                })


    rouge_df = pd.DataFrame(rouge_results)
    os.makedirs(output_dir, exist_ok=True)
    rouge_df.to_csv(os.path.join(output_dir, "rouge_scores.csv"), index=False)
    print(f"Saved ROUGE scores for {file_base}")


    if not rouge_df.empty:
        plt.figure(figsize=(12, 6))
        for metric in ["ROUGE-1", "ROUGE-2", "ROUGE-L"]:
            for (m1, m2), group in rouge_df.groupby(["Model1", "Model2"]):
                plt.plot(group["Round"], group[metric], marker='o', label=f"{m1} vs {m2} - {metric}")

        plt.xlabel("Round")
        plt.ylabel("ROUGE Score")
        plt.title(f"ROUGE Score Convergence - {file_base}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "rouge_convergence.png"))
        plt.close()

# Run for all files
for file in os.listdir(CSV_DIR):
    if file.endswith("_cleaned.csv"):
        file_path = os.path.join(CSV_DIR, file)
        output_path = os.path.join(PLOTS_DIR, file.replace(".csv", ""))
        calculate_rouge_for_file(file_path, output_path)
