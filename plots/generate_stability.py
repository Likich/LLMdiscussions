import os
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = os.path.join("..", "plots", "filtered_csvs")
PLOTS_DIR = "plots"

ROUNDS = ["initial", "R1", "R2", "R3", "R4", "R5", "final"]
ROUND_LABELS = ["Initial", "Round1", "Round2", "Round3", "Round4", "Round5", "Final"]
MODEL_PREFIXES = ["Maverick", "Llama3.3", "Deepseek", "Gemma", "Mistral"]

def detect_models(df):
    return [m for m in MODEL_PREFIXES if any(col.startswith(m) for col in df.columns)]

def get_col_name(model, round_label):
    if round_label == "initial":
        return model
    elif round_label == "final":
        return f"{model} Final"
    else:
        return f"{model} {round_label}"

def compute_code_stability(df, models):
    results = []
    for i in range(len(ROUNDS) - 1):
        prev_label = ROUNDS[i]
        next_label = ROUNDS[i + 1]
        prev_name = ROUND_LABELS[i]
        next_name = ROUND_LABELS[i + 1]

        for model in models:
            col_prev = get_col_name(model, prev_label)
            col_next = get_col_name(model, next_label)

            if col_prev not in df.columns or col_next not in df.columns:
                continue

            df_selected = df[[col_prev, col_next]].dropna()
            if df_selected.empty:
                continue

            changed = (df_selected[col_prev] != df_selected[col_next]).sum()
            total = len(df_selected)

            results.append({
                "Round_Transition": f"{prev_name} → {next_name}",
                "Model": model,
                "Code_Change_Ratio": changed / total
            })

    return pd.DataFrame(results)

def plot_stability(df_stability, output_path, title):
    plt.figure(figsize=(12, 6))
    print("\n🧠 DEBUG: df_stability columns:", df_stability.columns.tolist())
    print("🧠 DEBUG: df_stability head:\n", df_stability.head())

    for model in df_stability["Model"].unique():
        model_data = df_stability[df_stability["Model"] == model]
        plt.plot(model_data["Round_Transition"], model_data["Code_Change_Ratio"], marker='o', label=model)

    plt.xlabel("Round Transition")
    plt.ylabel("Code Change Ratio")
    plt.title(f"Code Stability Across Rounds - {title}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# MAIN
for file in os.listdir(CSV_DIR):
    if not file.endswith("_cleaned.csv"):
        continue

    file_path = os.path.join(CSV_DIR, file)
    df = pd.read_csv(file_path)

    file_base = file.replace(".csv", "")
    output_dir = os.path.join(PLOTS_DIR, file_base)
    os.makedirs(output_dir, exist_ok=True)

    models = detect_models(df)
    if len(models) < 1:
        continue

    stability_df = compute_code_stability(df, models)
    # stability_df.to_csv(os.path.join(output_dir, "code_stability.csv"), index=False)
    plot_stability(stability_df, os.path.join(output_dir, "code_stability.png"), title=file_base)

    print(f"Stability plot saved for {file_base}")
