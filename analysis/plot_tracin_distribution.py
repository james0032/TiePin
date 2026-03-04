#!/usr/bin/env python3
"""Plot distribution of TracIn scores for individual TracIn CSV files.

Usage:
    # Single file
    python plot_tracin_distribution.py --input triple_000_tracin.csv

    # Multiple files
    python plot_tracin_distribution.py --input triple_000_tracin.csv triple_001_tracin.csv

    # All tracin CSVs in a directory
    python plot_tracin_distribution.py --input-dir /path/to/tracin_csvs/

    # Custom output directory
    python plot_tracin_distribution.py --input triple_000_tracin.csv --output-dir /path/to/plots/
"""

import argparse
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde


def plot_tracin_distribution(csv_path: Path, output_path: Path = None):
    """Plot histogram with KDE overlay of TracInScore from a TracIn CSV file.

    Returns:
        Path to the saved PNG file.
    """
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
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Plot TracIn score distributions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input", nargs="+", metavar="CSV",
        help="One or more TracIn CSV files",
    )
    group.add_argument(
        "--input-dir", metavar="DIR",
        help="Directory containing TracIn CSV files",
    )
    parser.add_argument(
        "--pattern", default="*_tracin*.csv",
        help="Glob pattern when using --input-dir (default: *_tracin*.csv)",
    )
    parser.add_argument(
        "--output-dir", metavar="DIR", default=None,
        help="Output directory for PNGs (default: same directory as each input file)",
    )

    args = parser.parse_args()

    # Collect input files
    if args.input:
        csv_files = [Path(f) for f in args.input]
    else:
        input_dir = Path(args.input_dir)
        csv_files = sorted(input_dir.glob(args.pattern))
        if not csv_files:
            print(f"No files matching '{args.pattern}' in {args.input_dir}", file=sys.stderr)
            return 1
        print(f"Found {len(csv_files)} files matching '{args.pattern}'")

    for csv_file in csv_files:
        if not csv_file.exists():
            print(f"File not found: {csv_file}", file=sys.stderr)
            continue
        out_dir = Path(args.output_dir) if args.output_dir else csv_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = csv_file.stem + "_distribution.png"
        plot_tracin_distribution(csv_file, out_dir / out_name)

    return 0


if __name__ == "__main__":
    sys.exit(main())
