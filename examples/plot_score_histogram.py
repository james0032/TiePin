"""
Plot histogram of scores from ranked test predictions.

This script creates a histogram showing the distribution of prediction scores
from the trained ConvE model.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Input file
csv_file = '20251017_trained_test_scores_ranked.csv'

# Read the CSV file
print(f"Reading {csv_file}...")
df = pd.read_csv(csv_file)

print(f"Loaded {len(df)} predictions")
print(f"\nScore statistics:")
print(f"  Min:    {df['score'].min():.4f}")
print(f"  Max:    {df['score'].max():.4f}")
print(f"  Mean:   {df['score'].mean():.4f}")
print(f"  Median: {df['score'].median():.4f}")
print(f"  Std:    {df['score'].std():.4f}")

# Create histogram
plt.figure(figsize=(12, 7))

# Plot histogram with 50 bins
n, bins, patches = plt.hist(df['score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')

# Add title and labels
plt.title('Distribution of ConvE Model Prediction Scores', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Prediction Score', fontsize=13, fontweight='bold')
plt.ylabel('Frequency (Number of Predictions)', fontsize=13, fontweight='bold')

# Add grid for better readability
plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add statistics text box
stats_text = f'Statistics:\n' \
             f'Total: {len(df):,}\n' \
             f'Mean: {df["score"].mean():.4f}\n' \
             f'Median: {df["score"].median():.4f}\n' \
             f'Std Dev: {df["score"].std():.4f}\n' \
             f'Min: {df["score"].min():.4f}\n' \
             f'Max: {df["score"].max():.4f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=props, family='monospace')

# Add vertical line for mean
mean_score = df['score'].mean()
plt.axvline(mean_score, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_score:.4f}', alpha=0.8)

# Add vertical line for median
median_score = df['score'].median()
plt.axvline(median_score, color='green', linestyle='--', linewidth=2,
            label=f'Median: {median_score:.4f}', alpha=0.8)

# Add legend
plt.legend(loc='upper right', fontsize=11)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the figure
output_file = '20251017_trained_test_scores_histogram.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\nHistogram saved to: {output_file}")

# Also create a second plot with log scale for y-axis
plt.figure(figsize=(12, 7))

plt.hist(df['score'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
plt.yscale('log')

plt.title('Distribution of ConvE Model Prediction Scores (Log Scale)', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Prediction Score', fontsize=13, fontweight='bold')
plt.ylabel('Frequency (Number of Predictions) - Log Scale', fontsize=13, fontweight='bold')

plt.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add statistics text box
plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', bbox=props, family='monospace')

# Add vertical lines
plt.axvline(mean_score, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_score:.4f}', alpha=0.8)
plt.axvline(median_score, color='green', linestyle='--', linewidth=2,
            label=f'Median: {median_score:.4f}', alpha=0.8)

plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()

output_file_log = '20251017_trained_test_scores_histogram_log.png'
plt.savefig(output_file_log, dpi=300, bbox_inches='tight')
print(f"Histogram (log scale) saved to: {output_file_log}")

# Create percentile analysis
print(f"\nPercentile Analysis:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
for p in percentiles:
    value = np.percentile(df['score'], p)
    print(f"  {p:2d}th percentile: {value:.4f}")

print("\nDone!")
