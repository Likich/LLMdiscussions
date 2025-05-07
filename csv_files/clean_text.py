import os
import pandas as pd

# List of fluff phrases to delete
PHRASES_TO_REMOVE = [
    "**Qualitative Code:**",
    "To refine the codes collaboratively, let's evaluate the strengths and weaknesses of each proposal:",
    "The qualitative code extracted from the response is",
    "**Refined Qualitative Code:",
    "**Final Qualitative Code:",
    "### Qualitative Code",
    "```python",
    "**Codes:**",
    "**Refined Code:",
    "The qualitative code can be further refined for clarity and conciseness. Here's a suggestion:",
    "Comment:",
    "```plaintext",
    "### Refined Coding Structure:",
    "### Refined Coding Structure for Thematic Analysis",
    "**Final Agreed Qualitative Code**",
    "**Refined Qualitative Codes for",
    "The qualitative codes extracted from the response are:",
    "The qualitative code for the comment",
    "Certainly! The qualitative code from the comment is:",
    "Certainly! Here's the qualitative code extracted from the response:",
    "### Refined Codes and Themes",
    "* **Theme:",
    "```",
    "// The main idea is:",
    "**Thematic Analysis Codes for the Citation:**",
    "**Thematic Coding Summary:**",
    "**Thematic Coding Analysis**",
    "**Qualitative Codes:**",
    "**Refined Thematic Coding Summary**",
    "The qualitative code extracted from the comment",
    "### Refined Thematic Coding Framework:",
    "### Thematic Coding Summary:",
    "#### Themes and Subthemes:",
    "### Refined Themes and Sub-themes:",
    "**Themes:**",
    "### Refined Thematic Codes for",
    "**Code:**",
    "### Refined Thematic Coding Summary",
    "Refined Codes",
    "### Thematic Summary:",
    "**Code:",
    "** for Thematic Analysis:**",
    "#  for thematic analysis",
    "## Qualitative Code:",
    "Certainly! Here is the extracted qualitative code from the response:",
    "The qualitative code extracted from the response is:",
    "Certainly! Here are the qualitative codes extracted from the response:",
    "### Refined Thematic Codes and Themes",
    "#### Codes:",
    "The qualitative code extracted from the response includes the following themes:",
    "**Main Themes:**",
    "###",
    "**Thematic Code:**",
    "**Codes:**"
]


CSV_DIRECTORY = "new_extracted"


def clean_cell(text):
    if not isinstance(text, str):
        return text
    for phrase in PHRASES_TO_REMOVE:
        text = text.replace(phrase, "")
    return text.strip()

for filename in os.listdir(CSV_DIRECTORY):
    if filename.endswith(".csv") and filename.startswith("new_extracted_LLM_codes_"):
        file_path = os.path.join(CSV_DIRECTORY, filename)
        print(f"Cleaning {filename}...")

        df = pd.read_csv(file_path)
        df_cleaned = df.applymap(clean_cell)

        cleaned_filename = filename.replace(".csv", "_cleaned.csv")
        df_cleaned.to_csv(os.path.join(CSV_DIRECTORY, cleaned_filename), index=False)

        print(f"Saved cleaned file as {cleaned_filename}")
