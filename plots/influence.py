import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CSV_DIR = os.path.join("..", "plots", "filtered_csvs")
PLOTS_DIR = "plots"
ROUNDS = ["1", "2", "3", "4", "5", "6", "7"]
MODEL_PREFIXES = ["Maverick", "Llama3.3", "Deepseek", "Gemma", "Mistral"]

def detect_models(df):
    return [m for m in MODEL_PREFIXES if any(col.startswith(m) for col in df.columns)]

def rename_columns(df, models):
    new_columns = [df.columns[0]]  # Keep first column (e.g., 'Comment')
    for model in models:
        matching = [col for col in df.columns if col.startswith(model)]
        # Sort to maintain order: Initial, R1, R2, ..., Final
        sorted_cols = sorted(matching, key=lambda x: (
            0 if x == model else
            7 if "Final" in x else
            int(x.split("R")[1]) + 1 if "R" in x else 0
        ))
        for i, col in enumerate(sorted_cols):
            new_columns.append(f"{model} {ROUNDS[i]}")
    df.columns = new_columns
    return df

def compute_stability(df, models, rounds):
    results = {model: [] for model in models}
    for i in range(1, len(rounds)):
        prev_r, curr_r = rounds[i - 1], rounds[i]
        for model in models:
            try:
                prev = df[f"{model} {prev_r}"].astype(str)
                curr = df[f"{model} {curr_r}"].astype(str)
                unchanged = (prev == curr).sum() / len(prev)
                results[model].append(unchanged)
            except KeyError:
                results[model].append(np.nan)
    return pd.DataFrame(results, index=rounds[1:])

def compute_change(df, models, rounds):
    results = {model: [] for model in models}
    for i in range(1, len(rounds)):
        prev_r, curr_r = rounds[i - 1], rounds[i]
        for model in models:
            try:
                prev = df[f"{model} {prev_r}"].astype(str)
                curr = df[f"{model} {curr_r}"].astype(str)
                changed = (prev != curr).sum() / len(prev)
                results[model].append(changed)
            except KeyError:
                results[model].append(np.nan)
    return pd.DataFrame(results, index=rounds[1:])

def compute_influence(df, models, rounds):
    vectorizer = TfidfVectorizer()
    results = {model: [] for model in models}
    for i in range(1, len(rounds)):
        prev_r, curr_r = rounds[i - 1], rounds[i]
        for model in models:
            try:
                prev = df[f"{model} {prev_r}"].astype(str).tolist()
                curr = df[f"{model} {curr_r}"].astype(str).tolist()
                tfidf = vectorizer.fit_transform(prev + curr)
                sim = cosine_similarity(tfidf[:len(prev)], tfidf[len(prev):])
                scores = np.diag(sim)
                results[model].append(np.mean(scores))
            except KeyError:
                results[model].append(np.nan)
    return pd.DataFrame(results, index=rounds[1:])

def save_plot(df, ylabel, title, output_path):
    plt.figure(figsize=(10, 6))
    for col in df.columns:
        plt.plot(df.index, df[col], marker='o', label=col)
    plt.xlabel("Round")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# MAIN LOOP
for file in os.listdir(CSV_DIR):
    if not file.endswith("_cleaned.csv"):
        continue

    file_path = os.path.join(CSV_DIR, file)
    df = pd.read_csv(file_path)

    models = detect_models(df)
    if len(models) < 2:
        continue

    df = rename_columns(df, models)

    file_base = file.replace(".csv", "")
    output_dir = os.path.join(PLOTS_DIR, file_base)
    os.makedirs(output_dir, exist_ok=True)

    # STABILITY
    df_stability = compute_stability(df, models, ROUNDS)
    df_stability.to_csv(os.path.join(output_dir, "model_stability.csv"))
    save_plot(df_stability, "Stability (Unchanged Codes %)", f"Stability - {file_base}", os.path.join(output_dir, "model_stability.png"))

    # CHANGE RATE
    df_change = compute_change(df, models, ROUNDS)
    df_change.to_csv(os.path.join(output_dir, "model_change_rate.csv"))
    save_plot(df_change, "Change Rate (Modified Codes %)", f"Change Rate - {file_base}", os.path.join(output_dir, "model_change_rate.png"))

    # INFLUENCE (Cosine Similarity)
    df_influence = compute_influence(df, models, ROUNDS)
    df_influence.to_csv(os.path.join(output_dir, "model_influence.csv"))
    save_plot(df_influence, "Influence Score (Cosine Similarity)", f"Influence - {file_base}", os.path.join(output_dir, "model_influence.png"))

    print(f"Plots saved for {file_base}")
