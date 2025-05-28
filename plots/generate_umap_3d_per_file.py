import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.feature_extraction.text import TfidfVectorizer
from mpl_toolkits.mplot3d import Axes3D
CSV_DIR = os.path.join("..", "plots", "filtered_csvs")
PLOTS_BASE_DIR = "plots"

os.makedirs(PLOTS_BASE_DIR, exist_ok=True)

ALL_ROUNDS = ["initial", "R1", "R2", "R3", "R4", "R5", "final"]


def get_round_columns(df, round_label):
    """Find columns matching a given round_label (e.g. 'R1', 'Final')"""
    cols = []
    for model in MODEL_PREFIXES:
        if round_label == "initial":
            match = model
        elif round_label == "final":
            match = f"{model} Final"
        else:
            match = f"{model} {round_label}"

        if match in df.columns:
            cols.append(match)
    return cols


def generate_3d_umap_plot(df, columns, title, output_path):
    df_selected = df[columns].dropna()
    if df_selected.empty:
        print(f"⚠️  Skipping {title}: no valid data.")
        return

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df_selected.values.flatten())

    reducer = umap.UMAP(n_neighbors=10, min_dist=0.1, metric='cosine', random_state=42, n_components=3)
    embedding = reducer.fit_transform(X)

    df_umap = pd.DataFrame(embedding, columns=["UMAP1", "UMAP2", "UMAP3"])
    labels = []
    for col in columns:
        cleaned = df[col].dropna().astype(str)
        labels.extend([col] * len(cleaned))

    df_umap["Model"] = labels


    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df_umap["UMAP1"], df_umap["UMAP2"], df_umap["UMAP3"],
                         c=pd.factorize(df_umap["Model"])[0], cmap="tab10", alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")
    ax.set_zlabel("UMAP Dimension 3")

    legend_labels = {i: model for i, model in enumerate(df_umap["Model"].unique())}
    ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels.values(), loc='upper right')

    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


for file in os.listdir(CSV_DIR):
    if not file.endswith("_cleaned.csv"):
        continue

    full_path = os.path.join(CSV_DIR, file)
    df = pd.read_csv(full_path)

    file_base = file.replace(".csv", "")
    output_dir = os.path.join(PLOTS_BASE_DIR, file_base)
    os.makedirs(output_dir, exist_ok=True)

    detected_models = []
    for prefix in ["Maverick", "Llama3.3", "Deepseek", "Gemma", "Mistral"]:
        if any(col.startswith(prefix) for col in df.columns):
            detected_models.append(prefix)

    if len(detected_models) < 2:
        print(f"⏭ Skipping {file} — only {len(detected_models)} model(s) found.")
        continue

    model_prefixes = detected_models
    def get_round_columns(df, round_label):
        cols = []
        for model in model_prefixes:
            if round_label == "initial":
                match = model
            elif round_label == "final":
                match = f"{model} Final"
            else:
                match = f"{model} {round_label}"
            if match in df.columns:
                cols.append(match)
        return cols

    for round_label in ALL_ROUNDS:
        cols = get_round_columns(df, round_label)
        if not cols:
            continue

        title = f"3D UMAP - {round_label.capitalize()} Codes for {file_base}"
        out_file = os.path.join(output_dir, f"{round_label.lower()}_umap_3d.png")
        generate_3d_umap_plot(df, cols, title, out_file)
