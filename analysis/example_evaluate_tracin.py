#!/usr/bin/env python3
"""
Example script demonstrating the usage of TracIn metric evaluation functions.

This script shows how to:
1. Test the distribution of TracInScore between On_specific_path groups
2. Apply permutation testing to evaluate significance of metrics

Usage:
    python example_evaluate_tracin.py
"""

from pathlib import Path
from evaluate_tracin_metrics import (
    test_tracin_score_distribution,
    print_distribution_test_results,
    permutation_test_metrics,
    print_permutation_test_results
)

def main():
    # Path to example CSV file
    csv_file = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/CGGD_alltreat/triple_049_CHEBI_68595_MONDO_0005354_tracin_with_gt.csv")

    print("="*80)
    print("TRACIN METRICS EVALUATION EXAMPLE")
    print("="*80)
    print(f"\nAnalyzing file: {csv_file.name}")

    # 1. Test TracInScore distribution between groups
    print("\n" + "="*80)
    print("STEP 1: Testing TracInScore distribution between groups")
    print("="*80)

    distribution_results = test_tracin_score_distribution(
        csv_path=csv_file,
        on_path_column='On_specific_path',
        score_column='TracInScore'
    )

    if distribution_results:
        print_distribution_test_results(distribution_results)

    # 2. Apply permutation testing
    print("\n" + "="*80)
    print("STEP 2: Running permutation tests")
    print("="*80)
    print("\nNote: This will take a few minutes for 1000 permutations...")

    permutation_results = permutation_test_metrics(
        csv_path=csv_file,
        n_permutations=1000,
        on_path_column='On_specific_path',
        in_path_column='In_path',
        score_column='TracInScore',
        random_state=42
    )

    if permutation_results:
        print_permutation_test_results(permutation_results)

    # Optional: Save results to CSV for further analysis
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)

    if distribution_results:
        import pandas as pd

        # Save distribution test results
        dist_df = pd.DataFrame([distribution_results])
        dist_output = csv_file.parent / f"{csv_file.stem}_distribution_test.csv"
        dist_df.to_csv(dist_output, index=False)
        print(f"\nDistribution test results saved to: {dist_output}")

        # Save permutation test results (summary)
        perm_summary = {k: v for k, v in permutation_results.items()
                       if not k.endswith('_dist')}  # Exclude full distributions
        perm_df = pd.DataFrame([perm_summary])
        perm_output = csv_file.parent / f"{csv_file.stem}_permutation_test.csv"
        perm_df.to_csv(perm_output, index=False)
        print(f"Permutation test results saved to: {perm_output}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
