"""Plot distribution of TracIn scores for individual TracIn CSV files."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path
import sys


def plot_tracin_distribution(csv_path: Path, output_path: Path = None):
    """Plot histogram with KDE overlay of TracInScore from a TracIn CSV file."""
    df = pd.read_csv(csv_path)
    scores = df['TracInScore'].dropna().values

    # Derive test triple name from first row
    first = df.iloc[0]
    head = first.get('TestHead_label', first.get('TestHead', ''))
    rel = first.get('TestRel_label', first.get('TestRel', ''))
    tail = first.get('TestTail_label', first.get('TestTail', ''))
    triple_label = f"{head} -> {rel} -> {tail}"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=50, edgecolor='black', linewidth=0.3, alpha=0.7,
            density=True, label='Histogram')

    # KDE overlay
    kde = gaussian_kde(scores)
    x_grid = np.linspace(scores.min(), scores.max(), 500)
    ax.plot(x_grid, kde(x_grid), color='darkblue', linewidth=2, label='KDE')

    ax.set_xlabel('TracIn Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'TracIn Score Distribution\n{triple_label}', fontsize=13)
    ax.legend(fontsize=10)

    # Add summary stats
    stats_text = (
        f"n={len(scores):,}  "
        f"mean={np.mean(scores):.4f}  "
        f"median={np.median(scores):.4f}  "
        f"std={np.std(scores):.4f}\n"
        f"min={np.min(scores):.4f}  "
        f"max={np.max(scores):.4f}"
    )
    ax.text(0.98, 0.95, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path is None:
        output_path = csv_path.with_suffix('.png')

    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    base = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline/examples")

    files = [
        base / "triple_005_CHEBI_30411_MONDO_0008228_tracin_with_gt.csv",
        base / "triple_006_CHEBI_50381_MONDO_0018150_tracin_with_gt.csv",
    ]

    for csv_file in files:
        out_name = csv_file.stem + "_distribution.png"
        out_path = base / out_name
        plot_tracin_distribution(csv_file, out_path)


if __name__ == "__main__":
    main()
