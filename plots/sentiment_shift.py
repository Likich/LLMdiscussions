import os
import pandas as pd
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

CSV_DIR = os.path.join("..", "plots", "filtered_csvs")
PLOTS_DIR = "plots"

ROUNDS = ["initial", "R1", "R2", "R3", "R4", "R5", "final"]
ROUND_LABELS = ["Initial", "Round1", "Round2", "Round3", "Round4", "Round5", "Final"]
MODEL_PREFIXES = ["Maverick", "Llama3.3", "Deepseek", "Gemma", "Mistral"]

analyzer = SentimentIntensityAnalyzer()

def detect_models(df):
    return [m for m in MODEL_PREFIXES if any(col.startswith(m) for col in df.columns)]

def get_col_name(model, round_label):
    if round_label == "initial":
        return model
    elif round_label == "final":
        return f"{model} Final"
    else:
        return f"{model} {round_label}"

def compute_sentiment(df, models):
    results = []
    for round_label, round_name in zip(ROUNDS, ROUND_LABELS):
        for model in models:
            col = get_col_name(model, round_label)
            if col not in df.columns:
                continue

            df_selected = df[[col]].dropna()
            if df_selected.empty:
                continue

            scores = df_selected[col].astype(str).apply(lambda x: analyzer.polarity_scores(x)['compound'])
            avg_sentiment = scores.mean()

            results.append({
                "Round": round_name,
                "Model": model,
                "Avg_Sentiment": avg_sentiment
            })
    return pd.DataFrame(results)

def plot_sentiment(df_sentiment, output_path, title):
    plt.figure(figsize=(12, 6))
    for model in df_sentiment["Model"].unique():
        model_data = df_sentiment[df_sentiment["Model"] == model]
        plt.plot(model_data["Round"], model_data["Avg_Sentiment"], marker='o', label=model)

    plt.xlabel("Round")
    plt.ylabel("Average Sentiment Score")
    plt.title(f"Sentiment Shift in Codes - {title}")
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

    sentiment_df = compute_sentiment(df, models)
    # sentiment_df.to_csv(os.path.join(output_dir, "sentiment_trends.csv"), index=False)
    plot_sentiment(sentiment_df, os.path.join(output_dir, "sentiment_shift.png"), title=file_base)

    print(f"Sentiment shift plotted for {file_base}")
