#!/usr/bin/env python3
"""Join path edges with ground truth labels from a tracin_with_gt file.

Pulls rows from the tracin_with_gt CSV that match edges in the path_edges CSV,
so that path edges get IsGroundTruth, In_path, and On_specific_path labels.

Usage:
    python join_path_edges_with_gt.py \
        --gt triple_006_tracin_with_gt.csv \
        --path-edges triple_006_combined_path_edges.csv

    # Custom output path
    python join_path_edges_with_gt.py \
        --gt triple_006_tracin_with_gt.csv \
        --path-edges triple_006_combined_path_edges.csv \
        --output merged_with_gt.csv

    # Extract unique edges from top 10 paths, output full GT rows
    python join_path_edges_with_gt.py \
        --gt triple_006_tracin_with_gt.csv \
        --path-edges triple_006_combined_path_edges.csv \
        --top-n 10
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


OUTPUT_COLS = [
    "TestHead", "TestHead_label", "TestRel", "TestRel_label",
    "TestTail", "TestTail_label",
    "TrainHead", "TrainHead_label", "TrainRel", "TrainRel_label",
    "TrainTail", "TrainTail_label",
    "TracInScore", "SelfInfluence",
    "IsGroundTruth", "In_path", "On_specific_path",
]


def join_path_edges_with_gt(
    gt_path: Path,
    path_edges_path: Path,
    output_path: Path = None,
) -> Path:
    """Left-join path edges with ground truth labels.

    Args:
        gt_path: TracIn CSV with ground truth columns
            (IsGroundTruth, In_path, On_specific_path).
        path_edges_path: Path-edges CSV from extract_tracin_paths.py
            (columns include TrainHead, TrainRel, TrainTail).
        output_path: Where to write the merged CSV.
            Defaults to <path_edges_stem>_with_gt.csv.

    Returns:
        Path to the saved merged CSV.
    """
    df_gt = pd.read_csv(gt_path)
    df_path = pd.read_csv(path_edges_path)

    print(f"Ground truth file: {len(df_gt)} rows  ({gt_path.name})")
    print(f"Path edges file:   {len(df_path)} rows  ({path_edges_path.name})")

    # Join key: the train edge columns
    join_cols = ["TrainHead", "TrainRel", "TrainTail"]

    # Ground truth columns to pull
    gt_cols = [c for c in ["IsGroundTruth", "In_path", "On_specific_path"] if c in df_gt.columns]
    if not gt_cols:
        print("Warning: no ground truth columns found in gt file", file=sys.stderr)

    # Keep only join key + gt columns from gt file, drop duplicates
    available = [c for c in join_cols + gt_cols if c in df_gt.columns]
    df_gt_subset = df_gt[available].drop_duplicates(subset=join_cols)

    # Left join: keep all path edges, add gt labels where available
    df_merged = df_path.merge(df_gt_subset, on=join_cols, how="left")

    # Report
    if gt_cols:
        n_matched = df_merged[gt_cols[0]].notna().sum()
        n_unmatched = df_merged[gt_cols[0]].isna().sum()
        print(f"\nMatched:   {n_matched}")
        print(f"Unmatched: {n_unmatched}")

        if n_matched > 0:
            for col in gt_cols:
                n_true = (df_merged[col] == 1).sum()
                print(f"  {col}=1: {n_true}")

    # Save output
    if output_path is None:
        output_path = path_edges_path.parent / (path_edges_path.stem + "_with_gt.csv")

    df_merged.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}")
    return output_path


def extract_top_n_path_edges(
    gt_path: Path,
    path_edges_path: Path,
    top_n: int,
    output_path: Path = None,
) -> Path:
    """Extract unique edges from the top N paths and pull full GT rows.

    Filters the combined path_edges CSV to only the top N paths (by path_rank),
    collects the unique (TrainHead, TrainRel, TrainTail) edge tuples, then
    pulls the matching rows from the GT file with the standard output columns.

    Args:
        gt_path: TracIn with GT CSV (has all output columns).
        path_edges_path: Combined path-edges CSV (has path_rank column).
        top_n: Number of top-ranked paths to include.
        output_path: Where to write the result.

    Returns:
        Path to the saved CSV.
    """
    df_gt = pd.read_csv(gt_path)
    df_path = pd.read_csv(path_edges_path)

    print(f"Ground truth file: {len(df_gt)} rows  ({gt_path.name})")
    print(f"Path edges file:   {len(df_path)} rows  ({path_edges_path.name})")
    print(f"Top N paths:       {top_n}")

    # Filter to top N paths
    df_top = df_path[df_path["path_rank"] <= top_n]
    n_paths = df_top["path_rank"].nunique()
    print(f"Edges in top {top_n} paths: {len(df_top)}  ({n_paths} distinct paths)")

    # Unique edges
    join_cols = ["TrainHead", "TrainRel", "TrainTail"]
    unique_edges = df_top[join_cols].drop_duplicates()
    print(f"Unique edges: {len(unique_edges)}")

    # Inner join with GT to pull full rows
    # Only keep columns that exist in the GT file
    available_cols = [c for c in OUTPUT_COLS if c in df_gt.columns]
    df_gt_dedup = df_gt[available_cols].drop_duplicates(subset=join_cols)
    df_result = unique_edges.merge(df_gt_dedup, on=join_cols, how="left")

    # Report
    gt_label_cols = [c for c in ["IsGroundTruth", "In_path", "On_specific_path"] if c in df_result.columns]
    if gt_label_cols:
        n_matched = df_result[gt_label_cols[0]].notna().sum()
        print(f"\nMatched with GT:   {n_matched} / {len(df_result)}")
        for col in gt_label_cols:
            n_true = (df_result[col] == 1).sum()
            print(f"  {col}=1: {n_true}")

    # Save
    if output_path is None:
        stem = path_edges_path.stem.replace("_path_edges", "")
        output_path = path_edges_path.parent / f"{stem}_top{top_n}_path_edges.csv"

    df_result.to_csv(output_path, index=False)
    print(f"\nSaved: {output_path}  ({len(df_result)} rows)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Join path edges with ground truth labels from tracin_with_gt file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--gt", required=True, metavar="CSV",
        help="TracIn CSV with ground truth columns (tracin_with_gt.csv)",
    )
    parser.add_argument(
        "--path-edges", required=True, metavar="CSV",
        help="Path-edges CSV from extract_tracin_paths.py",
    )
    parser.add_argument(
        "--output", metavar="CSV", default=None,
        help="Output CSV path (default: auto-generated)",
    )
    parser.add_argument(
        "--top-n", type=int, default=None, metavar="N",
        help="Extract unique edges from the top N paths only, "
             "output full GT rows with standard columns",
    )

    args = parser.parse_args()

    gt_path = Path(args.gt)
    path_edges_path = Path(args.path_edges)
    output_path = Path(args.output) if args.output else None

    if not gt_path.exists():
        print(f"Error: GT file not found: {gt_path}", file=sys.stderr)
        return 1
    if not path_edges_path.exists():
        print(f"Error: Path edges file not found: {path_edges_path}", file=sys.stderr)
        return 1

    if args.top_n is not None:
        extract_top_n_path_edges(gt_path, path_edges_path, args.top_n, output_path)
    else:
        join_path_edges_with_gt(gt_path, path_edges_path, output_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
