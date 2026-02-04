"""Join path edges with ground truth labels from tracin_with_gt file.

Pulls rows from the tracin_with_gt CSV that match edges in the path_edges CSV,
so that path edges get IsGroundTruth, In_path, and On_specific_path labels.
"""

import pandas as pd
from pathlib import Path
import sys


def main():
    examples_dir = Path(
        "/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/"
        "data/clean_baseline/examples"
    )

    if len(sys.argv) > 2:
        gt_path = Path(sys.argv[1])
        path_edges_path = Path(sys.argv[2])
    else:
        gt_path = examples_dir / "triple_006_CHEBI_50381_MONDO_0018150_tracin_with_gt.csv"
        path_edges_path = examples_dir / "triple_006_CHEBI_50381_MONDO_0018150_cancer_head_6_hop_cancer_head_filtered_avgrank_top20_path_edges.csv"

    df_gt = pd.read_csv(gt_path)
    df_path = pd.read_csv(path_edges_path)

    print(f"Ground truth file: {len(df_gt)} rows")
    print(f"Path edges file:   {len(df_path)} rows")

    # Join key: the train edge columns
    join_cols = ['TrainHead', 'TrainRel', 'TrainTail']

    # Ground truth columns to pull
    gt_cols = ['IsGroundTruth', 'In_path', 'On_specific_path']

    # Keep only join key + gt columns from gt file, drop duplicates
    df_gt_subset = df_gt[join_cols + gt_cols].drop_duplicates(subset=join_cols)

    # Left join: keep all path edges, add gt labels where available
    df_merged = df_path.merge(df_gt_subset, on=join_cols, how='left')

    # Report
    n_matched = df_merged[gt_cols[0]].notna().sum()
    n_unmatched = df_merged[gt_cols[0]].isna().sum()
    print(f"\nMatched:   {n_matched}")
    print(f"Unmatched: {n_unmatched}")

    if n_matched > 0:
        n_gt = (df_merged['IsGroundTruth'] == 1).sum()
        n_in_path = (df_merged['In_path'] == 1).sum()
        n_on_specific = (df_merged['On_specific_path'] == 1).sum()
        print(f"\nAmong matched path edges:")
        print(f"  IsGroundTruth=1:    {n_gt}")
        print(f"  In_path=1:          {n_in_path}")
        print(f"  On_specific_path=1: {n_on_specific}")

    # Save output
    out_path = path_edges_path.parent / (path_edges_path.stem + "_with_gt.csv")
    df_merged.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
