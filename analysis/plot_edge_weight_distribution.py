"""Plot distribution of edge_weight for each test_triple in pagelink_important_edges.csv."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from pathlib import Path
import sys


def plot_edge_weight_distribution(df_group: pd.DataFrame, test_triple: str, output_path: Path):
    """Plot histogram with KDE overlay of edge_weight for a single test triple."""
    scores = df_group['edge_weight'].dropna().values

    # Derive label from test triple columns
    first = df_group.iloc[0]
    head = first.get('test_head', '')
    rel = first.get('test_relation', '')
    tail = first.get('test_tail', '')
    triple_label = f"{head} -> {rel} -> {tail}"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=50, edgecolor='black', linewidth=0.3, alpha=0.7,
            density=True, label='Histogram')

    # KDE overlay
    if len(scores) > 1 and np.std(scores) > 0:
        kde = gaussian_kde(scores)
        x_grid = np.linspace(scores.min(), scores.max(), 500)
        ax.plot(x_grid, kde(x_grid), color='darkblue', linewidth=2, label='KDE')

    ax.set_xlabel('Edge Weight', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Edge Weight Distribution\n{triple_label}', fontsize=13)
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

    fig.savefig(str(output_path), dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path(
            "/Users/jchung/Documents/RENCI/everycure/experiments/Influence_estimate/"
            "gnnexplain/data/08_reporting/20260130_miglustat_cobalamin_output/"
            "pagelink_important_edges.csv"
        )

    df = pd.read_csv(csv_path)
    output_dir = csv_path.parent

    # Remove duplicate edges per test_triple, keeping the highest edge_weight
    dup_cols = ['test_triple', 'edge_source', 'edge_relation', 'edge_target']
    n_before = len(df)
    df = df.sort_values('edge_weight', ascending=False).drop_duplicates(subset=dup_cols, keep='first')
    n_after = len(df)
    n_removed = n_before - n_after
    print(f"Removed {n_removed} duplicate edges ({n_before} -> {n_after})")

    print(f"\nEdges per test_triple (after dedup):")
    for test_triple, group in df.groupby('test_triple'):
        print(f"  {test_triple}: {len(group)} edges")

        # Sanitize triple name for filename
        safe_name = (
            str(test_triple)
            .replace(' ', '_')
            .replace('(', '')
            .replace(')', '')
            .replace(',', '')
            .replace(':', '_')
            .replace('/', '_')
        )
        out_path = output_dir / f"edge_weight_distribution_{safe_name}.png"
        plot_edge_weight_distribution(group, test_triple, out_path)


if __name__ == "__main__":
    main()
