import os
import pandas as pd
import umap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer

CSV_DIR = os.path.join("..", "csv_files", "new_extracted")
PLOTS_BASE_DIR = "plots"

os.makedirs(PLOTS_BASE_DIR, exist_ok=True)

MODEL_PREFIXES = ["Maverick", "Llama3.3", "Deepseek", "Gemma", "Mistral"]


def get_model_columns(df, mode):
    """Returns a dict of model name -> list of relevant columns (1 column per model) based on mode."""
    model_cols = {}
    for model in MODEL_PREFIXES:
        if mode == "initial":
            cols = [col for col in df.columns if col.strip() == model]
        elif mode == "final":
            cols = [col for col in df.columns if col.strip() == f"{model} Final"]
        else:
            raise ValueError("mode must be 'initial' or 'final'")
        if cols:
            model_cols[model] = cols
    return model_cols


def generate_umap_plot(file_path, mode="final"):
    df = pd.read_csv(file_path)
    file_name = os.path.basename(file_path).replace(".csv", "")
    print(f"🔍 Processing: {file_name} | Mode: {mode}")

    output_dir = os.path.join(PLOTS_BASE_DIR, file_name)
    os.makedirs(output_dir, exist_ok=True)

    model_cols = get_model_columns(df, mode=mode)

    all_texts = []
    labels = []

    for model, cols in model_cols.items():
        for col in cols:
            cleaned = df[col].dropna().astype(str)
            all_texts.extend(cleaned.tolist())
            labels.extend([model] * len(cleaned))

    if not all_texts:
        print(f"Skipping {file_name} — no usable text data for mode: {mode}.")
        return

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(all_texts)

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X)

    df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2"])
    df_umap["Model"] = labels

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x="UMAP1", y="UMAP2", hue="Model", palette="tab10", alpha=0.8, data=df_umap)
    plt.title(f"UMAP ({mode.capitalize()} Codes): {file_name}")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)

    output_path = os.path.join(output_dir, f"{mode}_umap.png")
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {mode} UMAP to: {output_path}")

for file in os.listdir(CSV_DIR):
    if file.endswith("_cleaned.csv"):
        full_path = os.path.join(CSV_DIR, file)
        for mode in ["initial", "final"]:
            generate_umap_plot(full_path, mode)
