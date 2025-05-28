#!/bin/bash
# run_all_analysis.sh

PROMPT=${1:-all}  # Default is 'all' if not specified
FILTER_DIR="filtered_csvs"

# Step 1: Filter CSVs using Python, not awk (trust issues)
python filter_csvs.py "$PROMPT"

# Step 2: Run your analysis scripts
for script in calculate_rouge.py generate_stability.py generate_umap_3d_per_file.py generate_umap_all.py influence.py sentiment_shift.py; do
    echo "Running $script..."
    python "$script"
done

echo "All analyses done for prompt $PROMPT"
