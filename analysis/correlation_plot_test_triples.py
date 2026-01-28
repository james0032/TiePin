"""
Create correlation plot between ConvE and CompGCN scores for the 750 test triples.
Includes Pearson's r and Spearman's rho correlation coefficients.
"""

import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


def main():
    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline/ConvE_top50_predict")

    # Load files
    print("Loading files...")
    conve_path = base_path / "scores_test_ranked_750_ConvE.csv"
    compgcn_path = base_path / "test_triple_750_scores_compgcn.csv"

    conve_df = pl.read_csv(conve_path)
    compgcn_df = pl.read_csv(compgcn_path)

    print(f"  ConvE rows: {len(conve_df):,}")
    print(f"  CompGCN rows: {len(compgcn_df):,}")

    # Join on drug-disease pairs to ensure alignment
    # ConvE: head_label, tail_label, score
    # CompGCN: head, tail, sigmoid_score
    conve_pairs = conve_df.select([
        pl.col("head_label").alias("drug"),
        pl.col("tail_label").alias("disease"),
        pl.col("score").alias("conve_score")
    ])

    compgcn_pairs = compgcn_df.select([
        pl.col("head").alias("drug"),
        pl.col("tail").alias("disease"),
        pl.col("sigmoid_score").alias("compgcn_score")
    ])

    # Join
    merged = conve_pairs.join(compgcn_pairs, on=["drug", "disease"], how="inner")
    print(f"  Merged rows: {len(merged):,}")

    # Extract scores
    conve_scores = np.array(merged["conve_score"].to_list())
    compgcn_scores = np.array(merged["compgcn_score"].to_list())

    # Calculate correlations
    pearson_r, pearson_p = stats.pearsonr(conve_scores, compgcn_scores)
    spearman_rho, spearman_p = stats.spearmanr(conve_scores, compgcn_scores)

    print(f"\nCorrelation Statistics:")
    print(f"  Pearson's r:   {pearson_r:.4f} (p={pearson_p:.2e})")
    print(f"  Spearman's ρ:  {spearman_rho:.4f} (p={spearman_p:.2e})")

    # Create plot
    fig, ax = plt.subplots(figsize=(6, 5))

    # Scatter plot
    ax.scatter(conve_scores, compgcn_scores, alpha=0.6, s=30, c='steelblue', edgecolors='none')

    # Add regression line
    z = np.polyfit(conve_scores, compgcn_scores, 1)
    p = np.poly1d(z)
    x_line = np.linspace(conve_scores.min(), conve_scores.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label='Linear fit')

    # Add diagonal reference line (y=x) if scales are similar
    min_val = min(conve_scores.min(), compgcn_scores.min())
    max_val = max(conve_scores.max(), compgcn_scores.max())

    # Labels and title
    ax.set_xlabel('ConvE Score', fontsize=12)
    ax.set_ylabel('CompGCN Sigmoid Score', fontsize=12)
    ax.set_title('Correlation of Model Scores for 750 Test Triples', fontsize=14)

    # Add correlation stats as text box
    textstr = '\n'.join([
        f"Pearson's r = {pearson_r:.4f}",
        f"(p = {pearson_p:.2e})",
        f"",
        f"Spearman's ρ = {spearman_rho:.4f}",
        f"(p = {spearman_p:.2e})",
        f"",
        f"n = {len(merged):,}"
    ])

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.05, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='bottom', bbox=props)

    # Add legend
    ax.legend(loc='lower right')

    # Grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save plot
    output_path = base_path / "correlation_plot_conve_compgcn.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    # Also save as PDF for publication quality
    pdf_path = base_path / "correlation_plot_conve_compgcn.pdf"
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to: {pdf_path}")

    plt.show()

    # Print score statistics
    print("\n=== Score Statistics ===")
    print(f"\nConvE scores:")
    print(f"  Min: {conve_scores.min():.4f}")
    print(f"  Max: {conve_scores.max():.4f}")
    print(f"  Mean: {conve_scores.mean():.4f}")
    print(f"  Std: {conve_scores.std():.4f}")

    print(f"\nCompGCN scores:")
    print(f"  Min: {compgcn_scores.min():.4f}")
    print(f"  Max: {compgcn_scores.max():.4f}")
    print(f"  Mean: {compgcn_scores.mean():.4f}")
    print(f"  Std: {compgcn_scores.std():.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
