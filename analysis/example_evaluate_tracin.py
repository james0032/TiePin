#!/usr/bin/env python3
"""
Example script demonstrating the usage of TracIn metric evaluation functions.

This script shows how to:
1. Test the distribution of TracInScore between On_specific_path groups (single file)
2. Apply permutation testing to evaluate significance of metrics (single file)
3. Batch analyze multiple files with all tests

Usage:
    # Single file analysis
    python example_evaluate_tracin.py --single-file

    # Batch analysis of all files in directory
    python example_evaluate_tracin.py --batch

    # Batch analysis with all tests (WARNING: slow for many files)
    python example_evaluate_tracin.py --batch --all-tests
"""

import argparse
from pathlib import Path
from evaluate_tracin_metrics import (
    test_tracin_score_distribution,
    print_distribution_test_results,
    permutation_test_metrics,
    print_permutation_test_results,
    analyze_all_files,
    print_all_results_summary
)


def single_file_example():
    """Example: Analyze a single file with detailed tests."""
    # Path to example CSV file
    csv_file = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/CGGD_alltreat/triple_049_CHEBI_68595_MONDO_0005354_tracin_with_gt.csv")

    print("="*80)
    print("SINGLE FILE TRACIN METRICS EVALUATION")
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


def batch_analysis_example(include_all_tests=False):
    """Example: Batch analyze multiple files."""
    # Path to directory containing CSV files
    input_dir = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/CGGD_alltreat")
    output_path = input_dir / "tracin_evaluation_results.csv"

    print("="*80)
    print("BATCH TRACIN METRICS EVALUATION")
    print("="*80)
    print(f"\nInput directory: {input_dir}")
    print(f"Output path: {output_path}")

    if include_all_tests:
        print("\nWARNING: Running distribution and permutation tests on all files.")
        print("This may take a considerable amount of time depending on the number of files.")
        print("Each permutation test runs 1000 permutations per file.\n")

    # Analyze all files
    results_dict = analyze_all_files(
        input_dir=input_dir,
        pattern="*_with_gt.csv",
        output_path=output_path,
        include_distribution_tests=include_all_tests,
        include_permutation_tests=include_all_tests,
        n_permutations=1000
    )

    # Print comprehensive summary
    print_all_results_summary(results_dict)

    print("\n" + "="*80)
    print("BATCH ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Basic metrics: {output_path}")

    if include_all_tests:
        if not results_dict['distribution_tests'].empty:
            dist_output = output_path.parent / f"{output_path.stem}_distribution_tests.csv"
            print(f"  Distribution tests: {dist_output}")

        if not results_dict['permutation_tests'].empty:
            perm_output = output_path.parent / f"{output_path.stem}_permutation_tests.csv"
            print(f"  Permutation tests: {perm_output}")


def main():
    parser = argparse.ArgumentParser(
        description='Example usage of TracIn metric evaluation functions'
    )
    parser.add_argument(
        '--single-file',
        action='store_true',
        help='Run single file analysis example'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Run batch analysis example'
    )
    parser.add_argument(
        '--all-tests',
        action='store_true',
        help='Include distribution and permutation tests in batch analysis (WARNING: slow)'
    )

    args = parser.parse_args()

    if args.single_file:
        single_file_example()
    elif args.batch:
        batch_analysis_example(include_all_tests=args.all_tests)
    else:
        print("Please specify either --single-file or --batch mode.")
        print("Use --help for more information.")


if __name__ == "__main__":
    main()
