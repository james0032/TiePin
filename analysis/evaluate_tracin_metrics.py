"""
Evaluate TracIn scores for identifying In_path edges from DrugMechDB.

This script calculates Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP)
to assess how well TracIn scores rank ground truth (In_path=1) training edges
for each test triple.

Metrics:
- MRR: Focuses on the rank of the first relevant item (1/rank_of_first_relevant)
- MAP: Considers all relevant items and their positions, rewarding rankings that
       place multiple relevant items higher
"""

import argparse
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_mrr(rankings: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Parameters
    ----------
    rankings : List[int]
        List of ranks (1-indexed) of the first relevant item for each query

    Returns
    -------
    float
        Mean Reciprocal Rank
    """
    if not rankings:
        return 0.0

    reciprocal_ranks = [1.0 / rank for rank in rankings if rank > 0]
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def calculate_average_precision(relevant_positions: List[int], total_items: int) -> float:
    """
    Calculate Average Precision for a single query.

    Parameters
    ----------
    relevant_positions : List[int]
        Sorted list of positions (1-indexed) where relevant items appear
    total_items : int
        Total number of items in the ranking

    Returns
    -------
    float
        Average Precision
    """
    if not relevant_positions:
        return 0.0

    # Calculate precision at each relevant position
    precisions = []
    for i, pos in enumerate(relevant_positions):
        # Precision@k = (number of relevant items in top k) / k
        num_relevant_at_k = i + 1
        precision_at_k = num_relevant_at_k / pos
        precisions.append(precision_at_k)

    # Average precision is the mean of precisions at relevant positions
    return np.mean(precisions)


def calculate_map(all_relevant_positions: List[List[int]], all_total_items: List[int]) -> float:
    """
    Calculate Mean Average Precision.

    Parameters
    ----------
    all_relevant_positions : List[List[int]]
        List of relevant positions for each query
    all_total_items : List[int]
        List of total items for each query

    Returns
    -------
    float
        Mean Average Precision
    """
    if not all_relevant_positions:
        return 0.0

    aps = []
    for relevant_pos, total in zip(all_relevant_positions, all_total_items):
        ap = calculate_average_precision(relevant_pos, total)
        aps.append(ap)

    return np.mean(aps)


def analyze_tracin_file(csv_path: Path, in_path_column: str = 'In_path') -> Dict[str, float]:
    """
    Analyze a single TracIn CSV file and calculate MRR and MAP.

    Parameters
    ----------
    csv_path : Path
        Path to the TracIn CSV file with ground truth annotations
    in_path_column : str
        Name of the column indicating ground truth (default: 'In_path')

    Returns
    -------
    Dict[str, float]
        Dictionary containing MRR, MAP, and other statistics
    """
    logger.info(f"Processing {csv_path.name}...")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required_cols = ['TestHead', 'TestRel', 'TestTail', 'TracInScore', in_path_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        logger.warning(f"Missing columns in {csv_path.name}: {missing_cols}")
        # Try alternate column names
        if in_path_column not in df.columns:
            if 'in_path' in df.columns:
                in_path_column = 'in_path'
            elif 'IsGroundTruth' in df.columns:
                in_path_column = 'IsGroundTruth'
            else:
                raise ValueError(f"Cannot find ground truth column in {csv_path.name}")

    # Group by test triple
    test_triple_col = 'test_triple'
    df[test_triple_col] = df['TestHead'].astype(str) + '|' + df['TestRel'].astype(str) + '|' + df['TestTail'].astype(str)

    # Calculate metrics for each test triple
    mrr_ranks = []
    map_relevant_positions = []
    map_total_items = []

    total_test_triples = 0
    triples_with_ground_truth = 0
    total_ground_truth_edges = 0

    for test_triple, group in df.groupby(test_triple_col):
        total_test_triples += 1

        # Sort by TracInScore in descending order (higher score = more influential)
        sorted_group = group.sort_values('TracInScore', ascending=False).reset_index(drop=True)

        # Find positions of ground truth items (In_path == 1)
        ground_truth_mask = sorted_group[in_path_column] == 1
        ground_truth_positions = sorted_group[ground_truth_mask].index + 1  # 1-indexed positions

        num_ground_truth = len(ground_truth_positions)
        total_ground_truth_edges += num_ground_truth

        if num_ground_truth > 0:
            triples_with_ground_truth += 1

            # MRR: rank of first relevant item
            first_relevant_rank = ground_truth_positions[0]
            mrr_ranks.append(first_relevant_rank)

            # MAP: all relevant positions
            map_relevant_positions.append(list(ground_truth_positions))
            map_total_items.append(len(sorted_group))

    # Calculate metrics
    mrr = calculate_mrr(mrr_ranks)
    map_score = calculate_map(map_relevant_positions, map_total_items)

    # Calculate additional statistics
    avg_ground_truth_per_triple = total_ground_truth_edges / total_test_triples if total_test_triples > 0 else 0
    coverage = triples_with_ground_truth / total_test_triples if total_test_triples > 0 else 0

    # Calculate average rank of ground truth edges
    avg_rank = np.mean(mrr_ranks) if mrr_ranks else 0
    median_rank = np.median(mrr_ranks) if mrr_ranks else 0

    results = {
        'file': csv_path.name,
        'mrr': mrr,
        'map': map_score,
        'total_test_triples': total_test_triples,
        'triples_with_ground_truth': triples_with_ground_truth,
        'total_ground_truth_edges': total_ground_truth_edges,
        'avg_ground_truth_per_triple': avg_ground_truth_per_triple,
        'coverage': coverage,
        'avg_rank_first_relevant': avg_rank,
        'median_rank_first_relevant': median_rank
    }

    return results


def analyze_all_files(input_dir: Path, pattern: str = "*_with_gt.csv",
                      output_path: Path = None) -> pd.DataFrame:
    """
    Analyze all TracIn files matching the pattern in the input directory.

    Parameters
    ----------
    input_dir : Path
        Directory containing TracIn CSV files
    pattern : str
        Glob pattern to match files (default: "*_with_gt.csv")
    output_path : Path, optional
        Path to save the results CSV

    Returns
    -------
    pd.DataFrame
        Results table with MRR, MAP, and statistics for each file
    """
    # Find all matching files
    csv_files = list(input_dir.glob(pattern))

    if not csv_files:
        logger.warning(f"No files matching pattern '{pattern}' found in {input_dir}")
        # Also try recursive search
        csv_files = list(input_dir.rglob(pattern))
        if csv_files:
            logger.info(f"Found {len(csv_files)} files in subdirectories")

    if not csv_files:
        logger.error(f"No files found!")
        return pd.DataFrame()

    logger.info(f"Found {len(csv_files)} files to process")

    # Process each file
    results = []
    for csv_file in sorted(csv_files):
        try:
            result = analyze_tracin_file(csv_file)
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Sort by MRR (descending)
    if not results_df.empty:
        results_df = results_df.sort_values('mrr', ascending=False).reset_index(drop=True)

    # Save results
    if output_path:
        results_df.to_csv(output_path, index=False)
        logger.info(f"Results saved to {output_path}")

    return results_df


def print_summary_statistics(results_df: pd.DataFrame):
    """
    Print summary statistics of the results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results dataframe
    """
    if results_df.empty:
        logger.warning("No results to summarize")
        return

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    print(f"\nNumber of files analyzed: {len(results_df)}")

    print(f"\nMean Reciprocal Rank (MRR):")
    print(f"  Overall Mean:   {results_df['mrr'].mean():.4f}")
    print(f"  Overall Median: {results_df['mrr'].median():.4f}")
    print(f"  Std Dev:        {results_df['mrr'].std():.4f}")
    print(f"  Min:            {results_df['mrr'].min():.4f}")
    print(f"  Max:            {results_df['mrr'].max():.4f}")

    print(f"\nMean Average Precision (MAP):")
    print(f"  Overall Mean:   {results_df['map'].mean():.4f}")
    print(f"  Overall Median: {results_df['map'].median():.4f}")
    print(f"  Std Dev:        {results_df['map'].std():.4f}")
    print(f"  Min:            {results_df['map'].min():.4f}")
    print(f"  Max:            {results_df['map'].max():.4f}")

    print(f"\nGround Truth Coverage:")
    print(f"  Avg coverage:   {results_df['coverage'].mean():.2%}")
    print(f"  Avg GT edges per triple: {results_df['avg_ground_truth_per_triple'].mean():.2f}")

    print(f"\nRanking Performance:")
    print(f"  Avg rank of first relevant: {results_df['avg_rank_first_relevant'].mean():.2f}")
    print(f"  Median rank of first relevant: {results_df['median_rank_first_relevant'].median():.2f}")

    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate TracIn scores using MRR and MAP metrics'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        required=True,
        help='Directory containing *_with_gt.csv files'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*_with_gt.csv',
        help='Glob pattern to match files (default: *_with_gt.csv)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=None,
        help='Output CSV file path (default: input_dir/tracin_evaluation_results.csv)'
    )
    parser.add_argument(
        '--in-path-column',
        type=str,
        default='In_path',
        help='Name of the column indicating ground truth (default: In_path)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Set default output path
    if args.output is None:
        args.output = args.input_dir / 'tracin_evaluation_results.csv'

    # Analyze all files
    results_df = analyze_all_files(args.input_dir, args.pattern, args.output)

    if not results_df.empty:
        # Print results table
        print("\n" + "="*80)
        print("RESULTS TABLE")
        print("="*80)
        print(results_df.to_string(index=False))

        # Print summary statistics
        print_summary_statistics(results_df)

        # Print top performers
        print("\n" + "="*80)
        print("TOP 5 FILES BY MRR")
        print("="*80)
        top_5_mrr = results_df.nlargest(5, 'mrr')[['file', 'mrr', 'map', 'coverage']]
        print(top_5_mrr.to_string(index=False))

        print("\n" + "="*80)
        print("TOP 5 FILES BY MAP")
        print("="*80)
        top_5_map = results_df.nlargest(5, 'map')[['file', 'mrr', 'map', 'coverage']]
        print(top_5_map.to_string(index=False))

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
