import os
import pandas as pd
import numpy as np
import re
from transformers import pipeline

# Directory with discussion CSVs
CSV_DIR = "../csv_files/new_extracted"
OUTPUT_DIR = "./other_metrics_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Define the target files (those you're clearly showing off in your screenshot)
target_files = [f for f in os.listdir(CSV_DIR)
                if f.startswith("extracted_discussion_") and f.endswith(".csv")]

MODEL_PREFIXES = ["Maverick", "Llama3.3", "Deepseek", "Gemma", "Mistral"]

# # Load transformers
# toxicity_classifier = pipeline("text-classification", model="unitary/unbiased-toxic-roberta", top_k=None)
# sentiment_analyzer = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")
# persuasiveness_classifier = pipeline("text-classification", model="cointegrated/roberta-large-cola-krishna2020")

certainty_words = set([
    "definitely", "Definitely", "must", "Must", "undoubtedly", "Undoubtedly", "always", "Always",
    "clearly", "Clearly", "certainly", "Certainly", "absolutely", "Absolutely", "without a doubt",
    "Without a doubt", "unquestionably", "Unquestionably", "incontestably", "Incontestably",
    "evidently", "Evidently", "conclusively", "Conclusively", "decisively", "Decisively",
    "unarguably", "Unarguably", "positively", "Positively", "inevitably", "Inevitably",
    "indisputably", "Indisputably", "assuredly", "Assuredly", "convincingly", "Convincingly",
    "categorically", "Categorically", "with certainty", "With certainty", "beyond doubt",
    "Beyond doubt", "beyond question", "Beyond question", "plainly", "Plainly", "no doubt",
    "No doubt", "undeniably", "Undeniably", "unfailingly", "Unfailingly", "irreversibly",
    "Irreversibly", "decidedly", "Decidedly", "irrefutably", "Irrefutably", "firmly", "Firmly",
    "as a fact", "As a fact", "proved", "Proved", "demonstrably", "Demonstrably", "unshakably",
    "Unshakably", "strongly", "Strongly"
])

hedging_words = set([
    "might", "Might", "possibly", "Possibly", "could", "Could", "likely", "Likely",
    "seems", "Seems", "apparently", "Apparently", "perhaps", "Perhaps", "maybe", "Maybe",
    "presumably", "Presumably", "potentially", "Potentially", "arguably", "Arguably",
    "allegedly", "Allegedly", "hypothetically", "Hypothetically", "ostensibly", "Ostensibly",
    "reportedly", "Reportedly", "supposedly", "Supposedly", "not necessarily", "Not necessarily",
    "relatively", "Relatively", "tentatively", "Tentatively", "uncertainly", "Uncertainly",
    "partially", "Partially", "somewhat", "Somewhat", "vaguely", "Vaguely", "in theory",
    "In theory", "according to some", "According to some", "one might argue", "One might argue",
    "allegedly", "Allegedly", "it is said", "It is said", "from what I gather", "From what I gather",
    "potentially speaking", "Potentially speaking", "I guess", "i guess", "presumed",
    "Presumed", "there is a chance", "There is a chance", "uncertain", "Uncertain",
    "it appears", "It appears", "it may be", "It may be"
])

def compute_confidence(text):
    words = re.findall(r'\b\w+\b', str(text).lower())
    certainty = sum(1 for word in words if word in certainty_words)
    hedging = sum(1 for word in words if word in hedging_words)
    return (certainty - hedging) / max(len(words), 1)

def compute_toxicity(text):
    try:
        scores = toxicity_classifier(str(text)[:512])
        return max(s["score"] for s in scores[0] if s["label"] != "neutral")
    except:
        return np.nan

# def compute_sentiment(text):
#     try:
#         result = sentiment_analyzer(str(text)[:512])[0]
#         return result["score"] if result["label"] == "LABEL_2" else -result["score"]
#     except:
#         return np.nan

# def compute_persuasiveness(text):
#     try:
#         result = persuasiveness_classifier(str(text)[:512])[0]
#         return result["score"] if result["label"] == "acceptable" else -result["score"]
#     except:
#         return np.nan

def analyze_file(file_path):
    df = pd.read_csv(file_path)
    for col in df.columns:
        if col == "Comment" or not any(prefix in col for prefix in MODEL_PREFIXES):
            continue
        df[f"{col}_confidence"] = df[col].apply(compute_confidence)
        df[f"{col}_toxicity"] = df[col].apply(compute_toxicity)
        # df[f"{col}_sentiment"] = df[col].apply(compute_sentiment)
        # df[f"{col}_persuasiveness"] = df[col].apply(compute_persuasiveness)

    filename = os.path.basename(file_path).replace(".csv", "_analyzed.csv")
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
    print(f" Done: {filename}")

# Run on each file
for filename in target_files:
    file_path = os.path.join(CSV_DIR, filename)
    print(f"Processing: {filename}")

    df = pd.read_csv(file_path)

    original_columns = list(df.columns[1:])  # skip "Comment"
    for col in original_columns:
        df[f"{col}_confidence"] = df[col].apply(compute_confidence)
        df[f"{col}_toxicity"] = df[col].apply(compute_toxicity)
        # df[f"{col}_sentiment"] = df[col].apply(compute_sentiment)
        # df[f"{col}_persuasiveness"] = df[col].apply(compute_persuasiveness)

    # Compute averages
    # rounds = df.columns[1:]
    # summary = pd.DataFrame({
    #     "Field": rounds,
    #     "Avg Confidence": [df[f"{col}_confidence"].mean() for col in rounds],
    #     "Avg Toxicity": [df[f"{col}_toxicity"].mean() for col in rounds],
    #     # "Avg Sentiment": [df[f"{col}_sentiment"].mean() for col in rounds],
    #     # "Avg Persuasiveness": [df[f"{col}_persuasiveness"].mean() for col in rounds],
    # })

    # Save outputs
    base = filename.replace(".csv", "")
    df.to_csv(os.path.join(OUTPUT_DIR, f"{base}_scored.csv"), index=False)
    # summary.to_csv(os.path.join(OUTPUT_DIR, f"{base}_summary.csv"), index=False)

    print(f"Done: {filename}")
