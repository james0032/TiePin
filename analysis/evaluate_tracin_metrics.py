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
from scipy import stats
from scipy.stats import (
    mannwhitneyu,
    ttest_ind,
    ks_2samp,
    levene,
    shapiro
)


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
