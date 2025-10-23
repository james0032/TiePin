"""
Extract top test triples with score > threshold for TracIn analysis.

This script filters test triples by score threshold and creates output files
suitable for TracIn analysis.
"""

import pandas as pd
from pathlib import Path

# Configuration
input_csv = '20251017_trained_test_scores_ranked.csv'
score_threshold = 0.84
output_dir = Path('top_test_triples')

# Create output directory
output_dir.mkdir(exist_ok=True)

# Read the CSV file
print(f"Reading {input_csv}...")
df = pd.read_csv(input_csv)
print(f"Loaded {len(df)} total predictions")

# Filter by score threshold
df_filtered = df[df['score'] > score_threshold].copy()
print(f"\nFiltered to {len(df_filtered)} predictions with score > {score_threshold}")

# Show score statistics for filtered data
print(f"\nFiltered score statistics:")
print(f"  Min:    {df_filtered['score'].min():.4f}")
print(f"  Max:    {df_filtered['score'].max():.4f}")
print(f"  Mean:   {df_filtered['score'].mean():.4f}")
print(f"  Median: {df_filtered['score'].median():.4f}")

# Extract the columns we need
df_triples = df_filtered[['head_label', 'relation_label', 'tail_label', 'score']].copy()

# Save triples in different formats

# Format 1: Tab-separated triples (for TracIn input)
output_triples = output_dir / 'top_test_triples.txt'
with open(output_triples, 'w') as f:
    for _, row in df_triples.iterrows():
        f.write(f"{row['head_label']}\t{row['relation_label']}\t{row['tail_label']}\n")
print(f"\n✓ Saved {len(df_triples)} triples to: {output_triples}")

# Format 2: CSV with scores (for reference)
output_csv = output_dir / 'top_test_triples_with_scores.csv'
df_triples.to_csv(output_csv, index=False)
print(f"✓ Saved CSV with scores to: {output_csv}")

# Format 3: Individual triple files (for single-triple TracIn analysis)
individual_dir = output_dir / 'individual_triples'
individual_dir.mkdir(exist_ok=True)

for idx, row in df_triples.iterrows():
    # Create filename from head and tail names (sanitized)
    head_name = row['head_label'].replace(':', '_').replace('/', '_')
    tail_name = row['tail_label'].replace(':', '_').replace('/', '_')
    filename = f"triple_{idx}_{head_name}_{tail_name}.txt"

    filepath = individual_dir / filename
    with open(filepath, 'w') as f:
        f.write(f"{row['head_label']}\t{row['relation_label']}\t{row['tail_label']}\n")

print(f"✓ Saved {len(df_triples)} individual triple files to: {individual_dir}")

# Format 4: Summary table
output_summary = output_dir / 'top_test_triples_summary.txt'
with open(output_summary, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write(f"Top Test Triples (score > {score_threshold})\n")
    f.write("=" * 100 + "\n\n")
    f.write(f"Total triples: {len(df_triples)}\n")
    f.write(f"Score range: {df_triples['score'].min():.4f} - {df_triples['score'].max():.4f}\n")
    f.write(f"Mean score: {df_triples['score'].mean():.4f}\n\n")
    f.write("=" * 100 + "\n")
    f.write("Triples List:\n")
    f.write("=" * 100 + "\n\n")

    for i, (_, row) in enumerate(df_triples.iterrows(), 1):
        f.write(f"{i:3d}. Score: {row['score']:.4f}\n")
        f.write(f"     Head:     {row['head_label']}\n")
        f.write(f"     Relation: {row['relation_label']}\n")
        f.write(f"     Tail:     {row['tail_label']}\n")
        f.write("\n")

print(f"✓ Saved summary to: {output_summary}")

# Print top 10 for quick reference
print("\n" + "=" * 100)
print("Top 10 Test Triples:")
print("=" * 100)
for i, (_, row) in enumerate(df_triples.head(10).iterrows(), 1):
    print(f"\n{i}. Score: {row['score']:.4f}")
    print(f"   {row['head_label']} --[{row['relation_label']}]--> {row['tail_label']}")

print("\n" + "=" * 100)
print("Output files created in:", output_dir)
print("=" * 100)
print("\nTo run TracIn on all top triples:")
print(f"  python run_tracin.py \\")
print(f"      --test {output_triples} \\")
print(f"      --mode test \\")
print(f"      --csv-output results/top_triples_tracin.csv")
print("\nTo run TracIn on a single triple:")
print(f"  python run_tracin.py \\")
print(f"      --test {individual_dir}/triple_2_CHEBI_34911_MONDO_0004525.txt \\")
print(f"      --mode single \\")
print(f"      --csv-output results/permethrin_scabies_tracin.csv")
print("\nDone!")
