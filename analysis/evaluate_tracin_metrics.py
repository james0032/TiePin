"""
Evaluate TracIn scores for identifying On_specific_path edges from DrugMechDB.

This script calculates Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP)
to assess how well TracIn scores rank ground truth (On_specific_path=1) training edges
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
from scipy import stats
from scipy.stats import (
    mannwhitneyu,
    ttest_ind,
    ks_2samp,
    levene,
    shapiro
)
import diptest


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


def load_prediction_scores(scores_path: Path) -> Dict[tuple, float]:
    """
    Load link prediction scores from a CSV file into a lookup dictionary.

    Parameters
    ----------
    scores_path : Path
        Path to prediction scores CSV with columns:
        head_label, relation_label, tail_label, score

    Returns
    -------
    Dict[tuple, float]
        Dictionary mapping (head_label, relation_label, tail_label) to score
    """
    logger.info(f"Loading prediction scores from {scores_path}...")
    scores_df = pd.read_csv(scores_path)

    required = ['head_label', 'relation_label', 'tail_label', 'score']
    missing = [c for c in required if c not in scores_df.columns]
    if missing:
        raise ValueError(f"Prediction scores CSV missing columns: {missing}")

    lookup = {}
    for _, row in scores_df.iterrows():
        key = (str(row['head_label']), str(row['relation_label']), str(row['tail_label']))
        lookup[key] = float(row['score'])

    logger.info(f"Loaded {len(lookup)} prediction scores")
    return lookup


def analyze_tracin_file(csv_path: Path, in_path_column: str = 'On_specific_path',
                        prediction_scores: Dict[tuple, float] = None) -> Dict[str, float]:
    """
    Analyze a single TracIn CSV file and calculate MRR and MAP.

    Parameters
    ----------
    csv_path : Path
        Path to the TracIn CSV file with ground truth annotations
    in_path_column : str
        Name of the column indicating ground truth (default: 'On_specific_path')
    prediction_scores : Dict[tuple, float], optional
        Lookup dictionary mapping (head_label, relation_label, tail_label) to
        link prediction score (e.g. from ConvE). If provided, the score is
        added to the results.

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
            if 'On_specific_path' in df.columns:
                in_path_column = 'On_specific_path'
            elif 'In_path' in df.columns:
                in_path_column = 'In_path'
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
    total_training_edges = 0

    for test_triple, group in df.groupby(test_triple_col):
        total_test_triples += 1
        total_training_edges += len(group)

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
    avg_training_edges_per_triple = total_training_edges / total_test_triples if total_test_triples > 0 else 0
    coverage = triples_with_ground_truth / total_test_triples if total_test_triples > 0 else 0

    # Calculate average rank of ground truth edges
    avg_rank = np.mean(mrr_ranks) if mrr_ranks else 0
    median_rank = np.median(mrr_ranks) if mrr_ranks else 0

    # Count outlier training edges by TracIn score
    all_scores = df['TracInScore'].dropna().values
    if len(all_scores) > 0:
        # MAD-based: edges with score > median + 3 * MAD
        median_score = np.median(all_scores)
        mad = np.median(np.abs(all_scores - median_score))
        mad_threshold = median_score + 3 * mad
        n_above_3mad = int(np.sum(all_scores > mad_threshold))

        # Std-based: edges with score > mean + 3 * std
        mean_score = np.mean(all_scores)
        std_score = np.std(all_scores, ddof=1) if len(all_scores) > 1 else 0.0
        std_threshold = mean_score + 3 * std_score
        n_above_3std = int(np.sum(all_scores > std_threshold))
        # Hartigan's Dip Test for bimodality (full distribution)
        dip_stat, dip_pval = diptest.diptest(all_scores)

        # Right-tail dip test: test only scores above the median
        # This detects a secondary hump on the high-score end
        right_tail = all_scores[all_scores > median_score]
        if len(right_tail) >= 10:
            right_dip_stat, right_dip_pval = diptest.diptest(right_tail)
        else:
            right_dip_stat = None
            right_dip_pval = None
    else:
        n_above_3mad = 0
        n_above_3std = 0
        dip_stat = None
        dip_pval = None
        right_dip_stat = None
        right_dip_pval = None

    # Extract test triple labels from first row for identification
    first_row = df.iloc[0]
    test_head_label = first_row.get('TestHead_label', first_row.get('TestHead', ''))
    test_rel_label = first_row.get('TestRel_label', first_row.get('TestRel', ''))
    test_tail_label = first_row.get('TestTail_label', first_row.get('TestTail', ''))

    results = {
        'file': csv_path.name,
        'test_head': str(test_head_label),
        'test_rel': str(test_rel_label),
        'test_tail': str(test_tail_label),
        'mrr': mrr,
        'map': map_score,
        'total_test_triples': total_test_triples,
        'triples_with_ground_truth': triples_with_ground_truth,
        'total_training_edges': total_training_edges,
        'total_ground_truth_edges': total_ground_truth_edges,
        'avg_training_edges_per_triple': avg_training_edges_per_triple,
        'avg_ground_truth_per_triple': avg_ground_truth_per_triple,
        'coverage': coverage,
        'avg_rank_first_relevant': avg_rank,
        'median_rank_first_relevant': median_rank,
        'n_above_3mad': n_above_3mad,
        'n_above_3std': n_above_3std,
        'dip_pvalue': dip_pval,
        'right_dip_pvalue': right_dip_pval
    }

    # Look up link prediction score if provided
    if prediction_scores is not None:
        key = (str(test_head_label), str(test_rel_label), str(test_tail_label))
        results['prediction_score'] = prediction_scores.get(key, None)
        if results['prediction_score'] is None:
            logger.warning(f"No prediction score found for {key} in {csv_path.name}")

    return results


def extract_on_specific_path_edges(csv_path: Path, in_path_column: str = 'On_specific_path') -> pd.DataFrame:
    """
    Extract all On_specific_path edges for each test triple from a TracIn CSV file.

    Parameters
    ----------
    csv_path : Path
        Path to the TracIn CSV file with ground truth annotations
    in_path_column : str
        Name of the column indicating ground truth (default: 'On_specific_path')

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: File, TestHead, TestRel, TestTail,
        TrainHead, TrainRel_label, TrainTail, TracInScore, Rank
    """
    df = pd.read_csv(csv_path)

    # Resolve column name with fallback
    if in_path_column not in df.columns:
        if 'On_specific_path' in df.columns:
            in_path_column = 'On_specific_path'
        elif 'In_path' in df.columns:
            in_path_column = 'In_path'
        elif 'IsGroundTruth' in df.columns:
            in_path_column = 'IsGroundTruth'
        else:
            logger.warning(f"No ground truth column found in {csv_path.name}, skipping")
            return pd.DataFrame()

    # Build test triple key
    df['test_triple'] = (df['TestHead'].astype(str) + '|' +
                         df['TestRel'].astype(str) + '|' +
                         df['TestTail'].astype(str))

    rows = []
    for test_triple, group in df.groupby('test_triple'):
        # Sort by TracInScore descending
        sorted_group = group.sort_values('TracInScore', ascending=False).reset_index(drop=True)

        # Filter to On_specific_path == 1
        on_path = sorted_group[sorted_group[in_path_column] == 1]

        if on_path.empty:
            continue

        for idx, row in on_path.iterrows():
            # Rank is the 1-based position in the sorted list
            rank = sorted_group.index.get_loc(idx) + 1 if idx in sorted_group.index else None
            train_cols = {}
            train_cols['File'] = csv_path.name
            train_cols['TestHead'] = row['TestHead']
            train_cols['TestRel'] = row['TestRel']
            train_cols['TestTail'] = row['TestTail']
            # Include training edge columns if available
            for col in ['TrainHead', 'TrainRel_label', 'TrainTail',
                        'TrainHead_name', 'TrainTail_name']:
                if col in row.index:
                    train_cols[col] = row[col]
            train_cols['TracInScore'] = row['TracInScore']
            train_cols['Rank'] = rank
            rows.append(train_cols)

    return pd.DataFrame(rows)


def analyze_all_files(input_dir: Path, pattern: str = "*_with_gt.csv",
                      output_path: Path = None,
                      include_distribution_tests: bool = False,
                      include_permutation_tests: bool = False,
                      n_permutations: int = 1000,
                      prediction_scores: Dict[tuple, float] = None) -> Dict[str, pd.DataFrame]:
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
    include_distribution_tests : bool
        Whether to include distribution tests (default: False)
    include_permutation_tests : bool
        Whether to include permutation tests (default: False)
    n_permutations : int
        Number of permutations for permutation tests (default: 1000)
    prediction_scores : Dict[tuple, float], optional
        Lookup dictionary mapping (head_label, relation_label, tail_label) to
        link prediction score

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary containing:
        - 'basic_metrics': MRR, MAP, and basic statistics
        - 'distribution_tests': Distribution test results (if included)
        - 'permutation_tests': Permutation test results (if included)
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
        return {'basic_metrics': pd.DataFrame()}

    logger.info(f"Found {len(csv_files)} files to process")

    # Process each file
    basic_results = []
    distribution_results = []
    permutation_results = []
    on_path_edges_dfs = []

    for csv_file in sorted(csv_files):
        try:
            # Basic metrics (MRR, MAP, etc.)
            result = analyze_tracin_file(csv_file, prediction_scores=prediction_scores)
            basic_results.append(result)

            # Extract On_specific_path edges for each test triple
            edges_df = extract_on_specific_path_edges(csv_file)
            if not edges_df.empty:
                on_path_edges_dfs.append(edges_df)

            # Distribution tests
            if include_distribution_tests:
                try:
                    dist_result = test_tracin_score_distribution(csv_file)
                    if dist_result:
                        distribution_results.append(dist_result)
                except Exception as e:
                    logger.error(f"Error in distribution test for {csv_file.name}: {e}")

            # Permutation tests
            if include_permutation_tests:
                try:
                    perm_result = permutation_test_metrics(
                        csv_file,
                        n_permutations=n_permutations,
                        on_path_column='On_specific_path',
                        in_path_column='In_path',
                        score_column='TracInScore'
                    )
                    if perm_result:
                        # Remove full distributions from results (too large for table)
                        perm_result_summary = {k: v for k, v in perm_result.items()
                                              if not k.endswith('_dist')}
                        permutation_results.append(perm_result_summary)
                except Exception as e:
                    logger.error(f"Error in permutation test for {csv_file.name}: {e}")

        except Exception as e:
            logger.error(f"Error processing {csv_file.name}: {e}")
            continue

    # Create results dataframes
    results_dict = {}

    # Basic metrics
    basic_df = pd.DataFrame(basic_results)
    if not basic_df.empty:
        basic_df = basic_df.sort_values('mrr', ascending=False).reset_index(drop=True)
    results_dict['basic_metrics'] = basic_df

    # On_specific_path edges
    if on_path_edges_dfs:
        on_path_edges_df = pd.concat(on_path_edges_dfs, ignore_index=True)
    else:
        on_path_edges_df = pd.DataFrame()
    results_dict['on_specific_path_edges'] = on_path_edges_df

    # Distribution tests
    if include_distribution_tests and distribution_results:
        dist_df = pd.DataFrame(distribution_results)
        results_dict['distribution_tests'] = dist_df
    else:
        results_dict['distribution_tests'] = pd.DataFrame()

    # Permutation tests
    if include_permutation_tests and permutation_results:
        perm_df = pd.DataFrame(permutation_results)
        results_dict['permutation_tests'] = perm_df
    else:
        results_dict['permutation_tests'] = pd.DataFrame()

    # Save results
    if output_path:
        # Save basic metrics
        basic_df.to_csv(output_path, index=False)
        logger.info(f"Basic metrics saved to {output_path}")

        # Save On_specific_path edges
        if not on_path_edges_df.empty:
            edges_output = output_path.parent / f"{output_path.stem}_on_specific_path_edges.csv"
            on_path_edges_df.to_csv(edges_output, index=False)
            logger.info(f"On_specific_path edges saved to {edges_output}")

        # Save distribution tests
        if include_distribution_tests and not results_dict['distribution_tests'].empty:
            dist_output = output_path.parent / f"{output_path.stem}_distribution_tests.csv"
            results_dict['distribution_tests'].to_csv(dist_output, index=False)
            logger.info(f"Distribution test results saved to {dist_output}")

        # Save permutation tests
        if include_permutation_tests and not results_dict['permutation_tests'].empty:
            perm_output = output_path.parent / f"{output_path.stem}_permutation_tests.csv"
            results_dict['permutation_tests'].to_csv(perm_output, index=False)
            logger.info(f"Permutation test results saved to {perm_output}")

    return results_dict


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


def print_distribution_summary_statistics(results_df: pd.DataFrame):
    """
    Print summary statistics for distribution test results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Distribution test results dataframe
    """
    if results_df.empty:
        logger.warning("No distribution test results to summarize")
        return

    print("\n" + "="*80)
    print("DISTRIBUTION TEST SUMMARY STATISTICS")
    print("="*80)

    print(f"\nNumber of files analyzed: {len(results_df)}")

    # Overall distribution differences
    print(f"\nTracInScore Differences (On_path - Off_path):")
    print(f"  Mean difference:")
    mean_diffs = results_df['mean_on_path'] - results_df['mean_off_path']
    print(f"    Average:  {mean_diffs.mean():.6f}")
    print(f"    Median:   {mean_diffs.median():.6f}")
    print(f"    Min:      {mean_diffs.min():.6f}")
    print(f"    Max:      {mean_diffs.max():.6f}")

    print(f"\n  Median difference:")
    median_diffs = results_df['median_on_path'] - results_df['median_off_path']
    print(f"    Average:  {median_diffs.mean():.6f}")
    print(f"    Median:   {median_diffs.median():.6f}")
    print(f"    Min:      {median_diffs.min():.6f}")
    print(f"    Max:      {median_diffs.max():.6f}")

    # Effect sizes
    print(f"\nEffect Sizes (Cohen's d):")
    print(f"  Mean:   {results_df['cohens_d'].mean():.4f}")
    print(f"  Median: {results_df['cohens_d'].median():.4f}")
    print(f"  Min:    {results_df['cohens_d'].min():.4f}")
    print(f"  Max:    {results_df['cohens_d'].max():.4f}")

    # Count interpretations
    print(f"\n  Effect size interpretations:")
    interpretation_counts = results_df['effect_size_interpretation'].value_counts()
    for interp, count in interpretation_counts.items():
        print(f"    {interp.capitalize()}: {count} ({count/len(results_df)*100:.1f}%)")

    # Test significance
    print(f"\nStatistical Significance (α=0.05):")
    print(f"  T-test significant: {(results_df['ttest_pvalue'] < 0.05).sum()} / {len(results_df)} "
          f"({(results_df['ttest_pvalue'] < 0.05).mean()*100:.1f}%)")
    print(f"  Mann-Whitney significant: {(results_df['mannwhitney_pvalue'] < 0.05).sum()} / {len(results_df)} "
          f"({(results_df['mannwhitney_pvalue'] < 0.05).mean()*100:.1f}%)")
    print(f"  KS-test significant: {(results_df['ks_pvalue'] < 0.05).sum()} / {len(results_df)} "
          f"({(results_df['ks_pvalue'] < 0.05).mean()*100:.1f}%)")

    # Normality
    print(f"\nNormality (Shapiro-Wilk, α=0.05):")
    print(f"  On_path normal: {results_df['normal_on_path'].sum()} / {len(results_df)} "
          f"({results_df['normal_on_path'].mean()*100:.1f}%)")
    print(f"  Off_path normal: {results_df['normal_off_path'].sum()} / {len(results_df)} "
          f"({results_df['normal_off_path'].mean()*100:.1f}%)")

    print("\n" + "="*80)


def print_permutation_summary_statistics(results_df: pd.DataFrame):
    """
    Print summary statistics for permutation test results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Permutation test results dataframe
    """
    if results_df.empty:
        logger.warning("No permutation test results to summarize")
        return

    print("\n" + "="*80)
    print("PERMUTATION TEST SUMMARY STATISTICS")
    print("="*80)

    print(f"\nNumber of files analyzed: {len(results_df)}")
    print(f"Permutations per file: {results_df['n_permutations'].iloc[0]}")

    # MRR significance
    print(f"\nMRR (Mean Reciprocal Rank) Significance:")
    print(f"  Files with p < 0.05: {(results_df['p_value_mrr'] < 0.05).sum()} / {len(results_df)} "
          f"({(results_df['p_value_mrr'] < 0.05).mean()*100:.1f}%)")
    print(f"  Files with p < 0.01: {(results_df['p_value_mrr'] < 0.01).sum()} / {len(results_df)} "
          f"({(results_df['p_value_mrr'] < 0.01).mean()*100:.1f}%)")
    print(f"  Mean p-value: {results_df['p_value_mrr'].mean():.4f}")
    print(f"  Median p-value: {results_df['p_value_mrr'].median():.4f}")

    # MAP significance
    print(f"\nMAP (Mean Average Precision) Significance:")
    print(f"  Files with p < 0.05: {(results_df['p_value_map'] < 0.05).sum()} / {len(results_df)} "
          f"({(results_df['p_value_map'] < 0.05).mean()*100:.1f}%)")
    print(f"  Files with p < 0.01: {(results_df['p_value_map'] < 0.01).sum()} / {len(results_df)} "
          f"({(results_df['p_value_map'] < 0.01).mean()*100:.1f}%)")
    print(f"  Mean p-value: {results_df['p_value_map'].mean():.4f}")
    print(f"  Median p-value: {results_df['p_value_map'].median():.4f}")

    # Distribution difference significance
    print(f"\nTracInScore Mean Difference Significance:")
    print(f"  Files with p < 0.05: {(results_df['p_value_mean_diff'] < 0.05).sum()} / {len(results_df)} "
          f"({(results_df['p_value_mean_diff'] < 0.05).mean()*100:.1f}%)")
    print(f"  Files with p < 0.01: {(results_df['p_value_mean_diff'] < 0.01).sum()} / {len(results_df)} "
          f"({(results_df['p_value_mean_diff'] < 0.01).mean()*100:.1f}%)")
    print(f"  Mean p-value: {results_df['p_value_mean_diff'].mean():.4f}")
    print(f"  Median p-value: {results_df['p_value_mean_diff'].median():.4f}")

    print(f"\nTracInScore Median Difference Significance:")
    print(f"  Files with p < 0.05: {(results_df['p_value_median_diff'] < 0.05).sum()} / {len(results_df)} "
          f"({(results_df['p_value_median_diff'] < 0.05).mean()*100:.1f}%)")
    print(f"  Files with p < 0.01: {(results_df['p_value_median_diff'] < 0.01).sum()} / {len(results_df)} "
          f"({(results_df['p_value_median_diff'] < 0.01).mean()*100:.1f}%)")
    print(f"  Mean p-value: {results_df['p_value_median_diff'].mean():.4f}")
    print(f"  Median p-value: {results_df['p_value_median_diff'].median():.4f}")

    # Overall significance
    print(f"\nOverall Significance Assessment:")
    # Count files where MRR AND MAP are both significant
    both_sig = ((results_df['p_value_mrr'] < 0.05) & (results_df['p_value_map'] < 0.05)).sum()
    print(f"  Files with both MRR & MAP significant (p < 0.05): {both_sig} / {len(results_df)} "
          f"({both_sig/len(results_df)*100:.1f}%)")

    # Count files where distribution tests are significant
    dist_sig = ((results_df['p_value_mean_diff'] < 0.05) | (results_df['p_value_median_diff'] < 0.05)).sum()
    print(f"  Files with distribution difference significant (p < 0.05): {dist_sig} / {len(results_df)} "
          f"({dist_sig/len(results_df)*100:.1f}%)")

    print("\n" + "="*80)


def print_all_results_summary(results_dict: Dict[str, pd.DataFrame]):
    """
    Print comprehensive summary of all analysis results.

    Parameters
    ----------
    results_dict : Dict[str, pd.DataFrame]
        Dictionary containing all results dataframes
    """
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*80)

    # Basic metrics summary
    if not results_dict['basic_metrics'].empty:
        print_summary_statistics(results_dict['basic_metrics'])

    # Distribution tests summary
    if not results_dict['distribution_tests'].empty:
        print_distribution_summary_statistics(results_dict['distribution_tests'])

    # Permutation tests summary
    if not results_dict['permutation_tests'].empty:
        print_permutation_summary_statistics(results_dict['permutation_tests'])


def test_tracin_score_distribution(csv_path: Path,
                                   on_path_column: str = 'On_specific_path',
                                   score_column: str = 'TracInScore') -> Dict[str, any]:
    """
    Test the distribution of TracInScore between On_specific_path=1 and On_specific_path=0 groups.

    Performs both parametric and non-parametric statistical tests to compare the distributions:
    - Parametric: Independent t-test (assumes normal distribution)
    - Non-parametric: Mann-Whitney U test, Kolmogorov-Smirnov test
    - Additional: Levene's test for equality of variances, Shapiro-Wilk test for normality

    Parameters
    ----------
    csv_path : Path
        Path to the TracIn CSV file with On_specific_path annotations
    on_path_column : str
        Name of the column indicating path membership (default: 'On_specific_path')
    score_column : str
        Name of the column containing TracIn scores (default: 'TracInScore')

    Returns
    -------
    Dict[str, any]
        Dictionary containing test statistics, p-values, and descriptive statistics
    """
    logger.info(f"Testing TracInScore distribution for {csv_path.name}...")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required_cols = [on_path_column, score_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing_cols}")

    # Split data into two groups
    on_path = df[df[on_path_column] == 1][score_column].dropna()
    off_path = df[df[on_path_column] == 0][score_column].dropna()

    if len(on_path) == 0 or len(off_path) == 0:
        logger.warning(f"Insufficient data: on_path={len(on_path)}, off_path={len(off_path)}")
        return None

    # Descriptive statistics
    results = {
        'file': csv_path.name,
        'n_on_path': len(on_path),
        'n_off_path': len(off_path),
        'mean_on_path': np.mean(on_path),
        'mean_off_path': np.mean(off_path),
        'median_on_path': np.median(on_path),
        'median_off_path': np.median(off_path),
        'std_on_path': np.std(on_path, ddof=1),
        'std_off_path': np.std(off_path, ddof=1),
        'q25_on_path': np.percentile(on_path, 25),
        'q25_off_path': np.percentile(off_path, 25),
        'q75_on_path': np.percentile(on_path, 75),
        'q75_off_path': np.percentile(off_path, 75),
    }

    # Test for normality (Shapiro-Wilk test)
    # Only run on sample if dataset is too large (max 5000 for Shapiro-Wilk)
    sample_size = 5000
    on_path_sample = on_path.sample(min(len(on_path), sample_size), random_state=42)
    off_path_sample = off_path.sample(min(len(off_path), sample_size), random_state=42)

    shapiro_on_path = shapiro(on_path_sample)
    shapiro_off_path = shapiro(off_path_sample)

    results['shapiro_stat_on_path'] = shapiro_on_path.statistic
    results['shapiro_pvalue_on_path'] = shapiro_on_path.pvalue
    results['shapiro_stat_off_path'] = shapiro_off_path.statistic
    results['shapiro_pvalue_off_path'] = shapiro_off_path.pvalue
    results['normal_on_path'] = shapiro_on_path.pvalue > 0.05
    results['normal_off_path'] = shapiro_off_path.pvalue > 0.05

    # Test for equality of variances (Levene's test)
    levene_result = levene(on_path, off_path)
    results['levene_stat'] = levene_result.statistic
    results['levene_pvalue'] = levene_result.pvalue
    results['equal_variance'] = levene_result.pvalue > 0.05

    # Parametric test: Independent t-test
    # Use Welch's t-test (equal_var=False) if variances are unequal
    ttest_result = ttest_ind(on_path, off_path, equal_var=results['equal_variance'])
    results['ttest_statistic'] = ttest_result.statistic
    results['ttest_pvalue'] = ttest_result.pvalue

    # Non-parametric test: Mann-Whitney U test (Wilcoxon rank-sum test)
    # Tests if one distribution is stochastically greater than the other
    mw_result = mannwhitneyu(on_path, off_path, alternative='two-sided')
    results['mannwhitney_statistic'] = mw_result.statistic
    results['mannwhitney_pvalue'] = mw_result.pvalue

    # Non-parametric test: Kolmogorov-Smirnov test
    # Tests if two samples come from the same distribution
    ks_result = ks_2samp(on_path, off_path)
    results['ks_statistic'] = ks_result.statistic
    results['ks_pvalue'] = ks_result.pvalue

    # Effect size: Cohen's d
    pooled_std = np.sqrt(((len(on_path) - 1) * results['std_on_path']**2 +
                          (len(off_path) - 1) * results['std_off_path']**2) /
                         (len(on_path) + len(off_path) - 2))
    cohens_d = (results['mean_on_path'] - results['mean_off_path']) / pooled_std
    results['cohens_d'] = cohens_d

    # Effect size interpretation
    if abs(cohens_d) < 0.2:
        effect_size_interpretation = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_size_interpretation = "small"
    elif abs(cohens_d) < 0.8:
        effect_size_interpretation = "medium"
    else:
        effect_size_interpretation = "large"
    results['effect_size_interpretation'] = effect_size_interpretation

    return results


def print_distribution_test_results(results: Dict[str, any]):
    """
    Pretty print the distribution test results.

    Parameters
    ----------
    results : Dict[str, any]
        Results dictionary from test_tracin_score_distribution
    """
    if results is None:
        logger.warning("No results to print")
        return

    print("\n" + "="*80)
    print(f"TRACIN SCORE DISTRIBUTION TEST: {results['file']}")
    print("="*80)

    print("\nDescriptive Statistics:")
    print(f"  On_specific_path = 1: n={results['n_on_path']}")
    print(f"    Mean:   {results['mean_on_path']:.6f}")
    print(f"    Median: {results['median_on_path']:.6f}")
    print(f"    Std:    {results['std_on_path']:.6f}")
    print(f"    Q1-Q3:  {results['q25_on_path']:.6f} - {results['q75_on_path']:.6f}")

    print(f"\n  On_specific_path = 0: n={results['n_off_path']}")
    print(f"    Mean:   {results['mean_off_path']:.6f}")
    print(f"    Median: {results['median_off_path']:.6f}")
    print(f"    Std:    {results['std_off_path']:.6f}")
    print(f"    Q1-Q3:  {results['q25_off_path']:.6f} - {results['q75_off_path']:.6f}")

    print(f"\n  Difference (On - Off):")
    print(f"    Mean:   {results['mean_on_path'] - results['mean_off_path']:.6f}")
    print(f"    Median: {results['median_on_path'] - results['median_off_path']:.6f}")

    print("\nNormality Tests (Shapiro-Wilk):")
    print(f"  On_specific_path = 1: W={results['shapiro_stat_on_path']:.4f}, "
          f"p={results['shapiro_pvalue_on_path']:.4e}, "
          f"Normal={'Yes' if results['normal_on_path'] else 'No'}")
    print(f"  On_specific_path = 0: W={results['shapiro_stat_off_path']:.4f}, "
          f"p={results['shapiro_pvalue_off_path']:.4e}, "
          f"Normal={'Yes' if results['normal_off_path'] else 'No'}")

    print("\nEquality of Variances (Levene's Test):")
    print(f"  Statistic: {results['levene_stat']:.4f}")
    print(f"  p-value:   {results['levene_pvalue']:.4e}")
    print(f"  Equal variance: {'Yes' if results['equal_variance'] else 'No'}")

    print("\nParametric Test (Independent t-test):")
    print(f"  t-statistic: {results['ttest_statistic']:.4f}")
    print(f"  p-value:     {results['ttest_pvalue']:.4e}")
    print(f"  Significant: {'Yes' if results['ttest_pvalue'] < 0.05 else 'No'} (α=0.05)")

    print("\nNon-Parametric Tests:")
    print(f"  Mann-Whitney U Test:")
    print(f"    U-statistic: {results['mannwhitney_statistic']:.4f}")
    print(f"    p-value:     {results['mannwhitney_pvalue']:.4e}")
    print(f"    Significant: {'Yes' if results['mannwhitney_pvalue'] < 0.05 else 'No'} (α=0.05)")

    print(f"\n  Kolmogorov-Smirnov Test:")
    print(f"    KS-statistic: {results['ks_statistic']:.4f}")
    print(f"    p-value:      {results['ks_pvalue']:.4e}")
    print(f"    Significant:  {'Yes' if results['ks_pvalue'] < 0.05 else 'No'} (α=0.05)")

    print("\nEffect Size:")
    print(f"  Cohen's d: {results['cohens_d']:.4f}")
    print(f"  Interpretation: {results['effect_size_interpretation']}")

    print("\n" + "="*80)


def permutation_test_metrics(csv_path: Path,
                            n_permutations: int = 1000,
                            on_path_column: str = 'On_specific_path',
                            in_path_column: str = 'In_path',
                            score_column: str = 'TracInScore',
                            random_state: int = 42) -> Dict[str, any]:
    """
    Apply permutation testing to evaluate if metrics (MRR, MAP, TracInScore distribution)
    are significantly different from random chance.

    The permutation test randomly shuffles the On_specific_path labels and recalculates
    the metrics to build a null distribution. This tests if the observed metrics are
    significantly better than what would be expected by chance.

    Parameters
    ----------
    csv_path : Path
        Path to the TracIn CSV file
    n_permutations : int
        Number of permutations to perform (default: 1000)
    on_path_column : str
        Name of the column indicating path membership (default: 'On_specific_path')
    in_path_column : str
        Name of the column for ground truth (default: 'In_path')
    score_column : str
        Name of the column containing TracIn scores (default: 'TracInScore')
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns
    -------
    Dict[str, any]
        Dictionary containing observed metrics, null distributions, and p-values
    """
    logger.info(f"Running permutation test for {csv_path.name} with {n_permutations} permutations...")

    # Read CSV
    df = pd.read_csv(csv_path)

    # Verify required columns exist
    required_cols = [on_path_column, in_path_column, score_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in {csv_path.name}: {missing_cols}")

    # Calculate observed metrics
    logger.info("Calculating observed metrics...")
    observed_metrics = _calculate_permutation_metrics(df, on_path_column, in_path_column, score_column)

    # Run permutations
    logger.info(f"Running {n_permutations} permutations...")
    np.random.seed(random_state)

    null_mrr = []
    null_map = []
    null_mean_diff = []
    null_median_diff = []
    null_mannwhitney = []
    null_ks = []

    for i in range(n_permutations):
        if (i + 1) % 100 == 0:
            logger.info(f"  Completed {i + 1}/{n_permutations} permutations")

        # Shuffle On_specific_path labels
        df_permuted = df.copy()
        df_permuted[on_path_column] = np.random.permutation(df_permuted[on_path_column].values)

        # Calculate metrics on permuted data
        perm_metrics = _calculate_permutation_metrics(df_permuted, on_path_column, in_path_column, score_column)

        null_mrr.append(perm_metrics['mrr'])
        null_map.append(perm_metrics['map'])
        null_mean_diff.append(perm_metrics['mean_diff'])
        null_median_diff.append(perm_metrics['median_diff'])
        null_mannwhitney.append(perm_metrics['mannwhitney_stat'])
        null_ks.append(perm_metrics['ks_stat'])

    # Convert to arrays
    null_mrr = np.array(null_mrr)
    null_map = np.array(null_map)
    null_mean_diff = np.array(null_mean_diff)
    null_median_diff = np.array(null_median_diff)
    null_mannwhitney = np.array(null_mannwhitney)
    null_ks = np.array(null_ks)

    # Calculate p-values (two-tailed)
    # For MRR and MAP: p-value = proportion of permutations >= observed
    p_mrr = np.mean(null_mrr >= observed_metrics['mrr'])
    p_map = np.mean(null_map >= observed_metrics['map'])

    # For distribution differences: two-tailed test
    p_mean_diff = np.mean(np.abs(null_mean_diff) >= np.abs(observed_metrics['mean_diff']))
    p_median_diff = np.mean(np.abs(null_median_diff) >= np.abs(observed_metrics['median_diff']))

    # For test statistics
    p_mannwhitney = np.mean(null_mannwhitney >= observed_metrics['mannwhitney_stat'])
    p_ks = np.mean(null_ks >= observed_metrics['ks_stat'])

    results = {
        'file': csv_path.name,
        'n_permutations': n_permutations,

        # Observed metrics
        'observed_mrr': observed_metrics['mrr'],
        'observed_map': observed_metrics['map'],
        'observed_mean_diff': observed_metrics['mean_diff'],
        'observed_median_diff': observed_metrics['median_diff'],
        'observed_mannwhitney': observed_metrics['mannwhitney_stat'],
        'observed_ks': observed_metrics['ks_stat'],

        # Null distributions (summary statistics)
        'null_mrr_mean': np.mean(null_mrr),
        'null_mrr_std': np.std(null_mrr),
        'null_mrr_q95': np.percentile(null_mrr, 95),

        'null_map_mean': np.mean(null_map),
        'null_map_std': np.std(null_map),
        'null_map_q95': np.percentile(null_map, 95),

        'null_mean_diff_mean': np.mean(null_mean_diff),
        'null_mean_diff_std': np.std(null_mean_diff),
        'null_mean_diff_q95': np.percentile(np.abs(null_mean_diff), 95),

        'null_median_diff_mean': np.mean(null_median_diff),
        'null_median_diff_std': np.std(null_median_diff),
        'null_median_diff_q95': np.percentile(np.abs(null_median_diff), 95),

        # P-values
        'p_value_mrr': p_mrr,
        'p_value_map': p_map,
        'p_value_mean_diff': p_mean_diff,
        'p_value_median_diff': p_median_diff,
        'p_value_mannwhitney': p_mannwhitney,
        'p_value_ks': p_ks,

        # Full null distributions (for plotting)
        'null_mrr_dist': null_mrr,
        'null_map_dist': null_map,
        'null_mean_diff_dist': null_mean_diff,
        'null_median_diff_dist': null_median_diff,
    }

    return results


def _calculate_permutation_metrics(df: pd.DataFrame,
                                   on_path_column: str,
                                   in_path_column: str,
                                   score_column: str) -> Dict[str, float]:
    """
    Helper function to calculate metrics for permutation testing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with TracIn data
    on_path_column : str
        Name of the column indicating path membership
    in_path_column : str
        Name of the column for ground truth
    score_column : str
        Name of the column containing TracIn scores

    Returns
    -------
    Dict[str, float]
        Dictionary containing MRR, MAP, and distribution metrics
    """
    # Calculate MRR and MAP using the On_specific_path as ground truth for ranking evaluation
    test_triple_col = 'test_triple'
    df[test_triple_col] = (df['TestHead'].astype(str) + '|' +
                           df['TestRel'].astype(str) + '|' +
                           df['TestTail'].astype(str))

    mrr_ranks = []
    map_relevant_positions = []
    map_total_items = []

    for test_triple, group in df.groupby(test_triple_col):
        # Sort by TracInScore in descending order
        sorted_group = group.sort_values(score_column, ascending=False).reset_index(drop=True)

        # Find positions of On_specific_path == 1 items
        on_path_mask = sorted_group[on_path_column] == 1
        on_path_positions = sorted_group[on_path_mask].index + 1  # 1-indexed

        if len(on_path_positions) > 0:
            # MRR: rank of first relevant item
            mrr_ranks.append(on_path_positions[0])

            # MAP: all relevant positions
            map_relevant_positions.append(list(on_path_positions))
            map_total_items.append(len(sorted_group))

    # Calculate metrics
    mrr = calculate_mrr(mrr_ranks) if mrr_ranks else 0.0
    map_score = calculate_map(map_relevant_positions, map_total_items) if map_relevant_positions else 0.0

    # Calculate distribution differences
    on_path_scores = df[df[on_path_column] == 1][score_column].dropna()
    off_path_scores = df[df[on_path_column] == 0][score_column].dropna()

    if len(on_path_scores) > 0 and len(off_path_scores) > 0:
        mean_diff = np.mean(on_path_scores) - np.mean(off_path_scores)
        median_diff = np.median(on_path_scores) - np.median(off_path_scores)

        # Calculate test statistics
        mw_result = mannwhitneyu(on_path_scores, off_path_scores, alternative='two-sided')
        ks_result = ks_2samp(on_path_scores, off_path_scores)

        mannwhitney_stat = mw_result.statistic
        ks_stat = ks_result.statistic
    else:
        mean_diff = 0.0
        median_diff = 0.0
        mannwhitney_stat = 0.0
        ks_stat = 0.0

    return {
        'mrr': mrr,
        'map': map_score,
        'mean_diff': mean_diff,
        'median_diff': median_diff,
        'mannwhitney_stat': mannwhitney_stat,
        'ks_stat': ks_stat,
    }


def print_permutation_test_results(results: Dict[str, any]):
    """
    Pretty print the permutation test results.

    Parameters
    ----------
    results : Dict[str, any]
        Results dictionary from permutation_test_metrics
    """
    if results is None:
        logger.warning("No results to print")
        return

    print("\n" + "="*80)
    print(f"PERMUTATION TEST RESULTS: {results['file']}")
    print(f"Number of permutations: {results['n_permutations']}")
    print("="*80)

    print("\n1. MRR (Mean Reciprocal Rank):")
    print(f"   Observed MRR:         {results['observed_mrr']:.6f}")
    print(f"   Null mean ± std:      {results['null_mrr_mean']:.6f} ± {results['null_mrr_std']:.6f}")
    print(f"   Null 95th percentile: {results['null_mrr_q95']:.6f}")
    print(f"   p-value:              {results['p_value_mrr']:.4f}")
    print(f"   Significant:          {'Yes' if results['p_value_mrr'] < 0.05 else 'No'} (α=0.05)")

    print("\n2. MAP (Mean Average Precision):")
    print(f"   Observed MAP:         {results['observed_map']:.6f}")
    print(f"   Null mean ± std:      {results['null_map_mean']:.6f} ± {results['null_map_std']:.6f}")
    print(f"   Null 95th percentile: {results['null_map_q95']:.6f}")
    print(f"   p-value:              {results['p_value_map']:.4f}")
    print(f"   Significant:          {'Yes' if results['p_value_map'] < 0.05 else 'No'} (α=0.05)")

    print("\n3. TracInScore Mean Difference (On_path - Off_path):")
    print(f"   Observed difference:  {results['observed_mean_diff']:.6f}")
    print(f"   Null mean ± std:      {results['null_mean_diff_mean']:.6f} ± {results['null_mean_diff_std']:.6f}")
    print(f"   Null |diff| 95th %:   {results['null_mean_diff_q95']:.6f}")
    print(f"   p-value:              {results['p_value_mean_diff']:.4f}")
    print(f"   Significant:          {'Yes' if results['p_value_mean_diff'] < 0.05 else 'No'} (α=0.05)")

    print("\n4. TracInScore Median Difference (On_path - Off_path):")
    print(f"   Observed difference:  {results['observed_median_diff']:.6f}")
    print(f"   Null mean ± std:      {results['null_median_diff_mean']:.6f} ± {results['null_median_diff_std']:.6f}")
    print(f"   Null |diff| 95th %:   {results['null_median_diff_q95']:.6f}")
    print(f"   p-value:              {results['p_value_median_diff']:.4f}")
    print(f"   Significant:          {'Yes' if results['p_value_median_diff'] < 0.05 else 'No'} (α=0.05)")

    print("\n5. Mann-Whitney U Test Statistic:")
    print(f"   Observed statistic:   {results['observed_mannwhitney']:.4f}")
    print(f"   p-value:              {results['p_value_mannwhitney']:.4f}")
    print(f"   Significant:          {'Yes' if results['p_value_mannwhitney'] < 0.05 else 'No'} (α=0.05)")

    print("\n6. Kolmogorov-Smirnov Test Statistic:")
    print(f"   Observed statistic:   {results['observed_ks']:.4f}")
    print(f"   p-value:              {results['p_value_ks']:.4f}")
    print(f"   Significant:          {'Yes' if results['p_value_ks'] < 0.05 else 'No'} (α=0.05)")

    print("\n" + "="*80)
    print("\nInterpretation:")
    print("- p-value < 0.05: The observed metric is significantly different from random chance")
    print("- p-value ≥ 0.05: The observed metric is not significantly different from random")
    print("="*80)


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
        default='On_specific_path',
        help='Name of the column indicating ground truth (default: On_specific_path)'
    )
    parser.add_argument(
        '--distribution-tests',
        action='store_true',
        help='Include distribution tests (parametric and non-parametric)'
    )
    parser.add_argument(
        '--permutation-tests',
        action='store_true',
        help='Include permutation tests for significance evaluation'
    )
    parser.add_argument(
        '--n-permutations',
        type=int,
        default=1000,
        help='Number of permutations for permutation tests (default: 1000)'
    )
    parser.add_argument(
        '--prediction-scores',
        type=Path,
        default=None,
        help='Path to link prediction scores CSV (e.g. scores_test.csv) with columns: '
             'head_label, relation_label, tail_label, score. '
             'Used to add a prediction_score column to the evaluation results.'
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

    # Load prediction scores if provided
    pred_scores = None
    if args.prediction_scores is not None:
        pred_scores = load_prediction_scores(args.prediction_scores)

    # Analyze all files
    results_dict = analyze_all_files(
        args.input_dir,
        args.pattern,
        args.output,
        include_distribution_tests=args.distribution_tests,
        include_permutation_tests=args.permutation_tests,
        n_permutations=args.n_permutations,
        prediction_scores=pred_scores
    )

    # Print basic metrics results
    if not results_dict['basic_metrics'].empty:
        print("\n" + "="*80)
        print("BASIC METRICS TABLE")
        print("="*80)
        print(results_dict['basic_metrics'].to_string(index=False))

        # Print summary statistics
        print_summary_statistics(results_dict['basic_metrics'])

    # Print On_specific_path edges summary
    if not results_dict['on_specific_path_edges'].empty:
        edges_df = results_dict['on_specific_path_edges']
        print("\n" + "="*80)
        print("ON_SPECIFIC_PATH EDGES SUMMARY")
        print("="*80)
        n_files = edges_df['File'].nunique()
        n_triples = edges_df.groupby(['File', 'TestHead', 'TestRel', 'TestTail']).ngroups
        print(f"  Total On_specific_path edges: {len(edges_df)}")
        print(f"  Across {n_files} files, {n_triples} test triples")
        print(f"  Avg edges per test triple: {len(edges_df) / n_triples:.2f}")
        print(f"  Avg rank of On_specific_path edges: {edges_df['Rank'].mean():.2f}")
        print(f"  Median rank: {edges_df['Rank'].median():.1f}")
        edges_output = args.output.parent / f"{args.output.stem}_on_specific_path_edges.csv"
        print(f"  Saved to: {edges_output}")

        # Print top performers
        print("\n" + "="*80)
        print("TOP 5 FILES BY MRR")
        print("="*80)
        top_5_mrr = results_dict['basic_metrics'].nlargest(5, 'mrr')[['file', 'mrr', 'map', 'coverage']]
        print(top_5_mrr.to_string(index=False))

        print("\n" + "="*80)
        print("TOP 5 FILES BY MAP")
        print("="*80)
        top_5_map = results_dict['basic_metrics'].nlargest(5, 'map')[['file', 'mrr', 'map', 'coverage']]
        print(top_5_map.to_string(index=False))

    # Print distribution test results
    if not results_dict['distribution_tests'].empty:
        print("\n" + "="*80)
        print("DISTRIBUTION TEST RESULTS")
        print("="*80)
        print("\nKey columns from distribution tests:")
        dist_key_cols = ['file', 'mean_on_path', 'mean_off_path', 'cohens_d',
                        'ttest_pvalue', 'mannwhitney_pvalue', 'ks_pvalue']
        available_cols = [col for col in dist_key_cols if col in results_dict['distribution_tests'].columns]
        print(results_dict['distribution_tests'][available_cols].to_string(index=False))

        print_distribution_summary_statistics(results_dict['distribution_tests'])

    # Print permutation test results
    if not results_dict['permutation_tests'].empty:
        print("\n" + "="*80)
        print("PERMUTATION TEST RESULTS")
        print("="*80)
        print("\nKey columns from permutation tests:")
        perm_key_cols = ['file', 'observed_mrr', 'observed_map', 'observed_mean_diff',
                        'p_value_mrr', 'p_value_map', 'p_value_mean_diff']
        available_cols = [col for col in perm_key_cols if col in results_dict['permutation_tests'].columns]
        print(results_dict['permutation_tests'][available_cols].to_string(index=False))

        print_permutation_summary_statistics(results_dict['permutation_tests'])

    logger.info("Analysis complete!")


if __name__ == "__main__":
    main()
