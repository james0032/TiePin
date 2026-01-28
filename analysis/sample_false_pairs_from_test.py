"""
Sample is_treat=False pairs for drugs and diseases that appear in the 384 test triples.

For each drug in the 384 test pairs: randomly select 15 is_treat=False pairs
For each disease in the 384 test pairs: randomly select 15 is_treat=False pairs

Also combines the filtered score matrices with the 750 test triples and removes duplicates.

This creates a balanced dataset for analysis with diverse coverage of drugs and diseases.
"""

import polars as pl
from pathlib import Path
import random

SAMPLE_SIZE = 15


def combine_compgcn_with_test(
    filtered_path: str,
    test_path: str
) -> pl.DataFrame:
    """
    Combine CompGCN filtered file with test triples, remove duplicates.

    CompGCN filtered columns: head_id, head, relation, tail_id, tail, raw_score, sigmoid_score, is_treat
    Test triple columns: head_id, head, relation, tail_id, tail, raw_score, sigmoid_score
    """
    print("\n=== Combining CompGCN with test triples ===")
    filtered = pl.read_csv(filtered_path)
    test = pl.read_csv(test_path)

    print(f"  Filtered rows: {len(filtered):,}")
    print(f"  Test triple rows: {len(test):,}")

    # Add is_treat=True column to test triples (they are all treats)
    test = test.with_columns(pl.lit(True).alias("is_treat"))

    # Combine
    combined = pl.concat([filtered, test])

    # Remove duplicates based on (head, tail) pair, keeping first occurrence
    combined = combined.unique(subset=["head", "tail"], keep="first")

    print(f"  Combined (deduplicated) rows: {len(combined):,}")
    return combined


def combine_conve_with_test(
    filtered_path: str,
    test_path: str
) -> pl.DataFrame:
    """
    Combine ConvE filtered file with test triples, remove duplicates.

    ConvE filtered columns: head_id, head_label, relation_id, relation_label, tail_id, tail_label, score, is_treat
    Test triple columns: rank, head_id, head_label, head_name, relation_id, relation_label, tail_id, tail_label, tail_name, score
    """
    print("\n=== Combining ConvE with test triples ===")
    filtered = pl.read_csv(filtered_path)
    test = pl.read_csv(test_path)

    print(f"  Filtered rows: {len(filtered):,}")
    print(f"  Test triple rows: {len(test):,}")

    # Select and rename test columns to match filtered format, add is_treat=True
    test_reformatted = test.select([
        pl.col("head_id"),
        pl.col("head_label"),
        pl.col("relation_id"),
        pl.col("relation_label"),
        pl.col("tail_id"),
        pl.col("tail_label"),
        pl.col("score"),
    ]).with_columns(pl.lit(True).alias("is_treat"))

    # Combine
    combined = pl.concat([filtered, test_reformatted])

    # Remove duplicates based on (head_label, tail_label) pair, keeping first occurrence
    combined = combined.unique(subset=["head_label", "tail_label"], keep="first")

    print(f"  Combined (deduplicated) rows: {len(combined):,}")
    return combined


def sample_false_pairs(
    compgcn: pl.DataFrame,
    conve: pl.DataFrame,
    test: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Sample is_treat=False pairs for drugs and diseases in the overlapping test pairs.
    """
    print("\n=== Sampling false pairs ===")

    # Get is_treat=True pairs from CompGCN
    compgcn_treats = compgcn.filter(pl.col("is_treat") == True)
    compgcn_treat_pairs = set(
        zip(compgcn_treats["head"].to_list(), compgcn_treats["tail"].to_list())
    )

    # Get test pairs
    test_pairs = set(
        zip(test["head_label"].to_list(), test["tail_label"].to_list())
    )

    # Find the overlapping pairs
    overlap_pairs = compgcn_treat_pairs & test_pairs
    print(f"is_treat pairs also in test triples: {len(overlap_pairs):,}")

    # Get unique drugs and diseases from the overlapping pairs
    overlap_drugs = set(p[0] for p in overlap_pairs)
    overlap_diseases = set(p[1] for p in overlap_pairs)
    print(f"  Unique drugs in overlap: {len(overlap_drugs):,}")
    print(f"  Unique diseases in overlap: {len(overlap_diseases):,}")

    # Get is_treat=False pairs from CompGCN
    compgcn_false = compgcn.filter(pl.col("is_treat") == False)
    print(f"\nis_treat=False rows in CompGCN: {len(compgcn_false):,}")

    # Sample 15 false pairs for each drug
    print(f"\nSampling {SAMPLE_SIZE} false pairs for each drug...")
    drug_sampled_pairs = set()
    for drug in overlap_drugs:
        drug_false = compgcn_false.filter(pl.col("head") == drug)
        pairs = list(zip(drug_false["head"].to_list(), drug_false["tail"].to_list()))
        sample_size = min(SAMPLE_SIZE, len(pairs))
        if sample_size > 0:
            sampled = random.sample(pairs, sample_size)
            drug_sampled_pairs.update(sampled)

    print(f"  Pairs sampled by drug: {len(drug_sampled_pairs):,}")

    # Sample 15 false pairs for each disease
    print(f"\nSampling {SAMPLE_SIZE} false pairs for each disease...")
    disease_sampled_pairs = set()
    for disease in overlap_diseases:
        disease_false = compgcn_false.filter(pl.col("tail") == disease)
        pairs = list(zip(disease_false["head"].to_list(), disease_false["tail"].to_list()))
        sample_size = min(SAMPLE_SIZE, len(pairs))
        if sample_size > 0:
            sampled = random.sample(pairs, sample_size)
            disease_sampled_pairs.update(sampled)

    print(f"  Pairs sampled by disease: {len(disease_sampled_pairs):,}")

    # Combine all sampled pairs (union to avoid duplicates)
    all_sampled_pairs = drug_sampled_pairs | disease_sampled_pairs
    print(f"\nTotal unique sampled false pairs: {len(all_sampled_pairs):,}")

    # Create DataFrame with sampled pairs for filtering
    sampled_df = pl.DataFrame({
        "drug": [p[0] for p in all_sampled_pairs],
        "disease": [p[1] for p in all_sampled_pairs]
    })

    # Also include the overlap pairs (is_treat=True)
    overlap_df = pl.DataFrame({
        "drug": [p[0] for p in overlap_pairs],
        "disease": [p[1] for p in overlap_pairs]
    })

    # Combine overlap + sampled pairs
    all_pairs_df = pl.concat([overlap_df, sampled_df]).unique()

    # Filter CompGCN to get selected pairs
    compgcn_sampled = compgcn.join(
        all_pairs_df,
        left_on=["head", "tail"],
        right_on=["drug", "disease"],
        how="inner"
    ).sort(["head", "tail"])

    # Filter ConvE to get the same pairs
    conve_sampled = conve.join(
        all_pairs_df,
        left_on=["head_label", "tail_label"],
        right_on=["drug", "disease"],
        how="inner"
    ).sort(["head_label", "tail_label"])

    print(f"\n=== Final Sampled Dataset ===")
    print(f"  CompGCN sampled rows: {len(compgcn_sampled):,}")
    print(f"  ConvE sampled rows: {len(conve_sampled):,}")
    print(f"  is_treat=True: {compgcn_sampled.filter(pl.col('is_treat')).height:,}")
    print(f"  is_treat=False: {compgcn_sampled.filter(~pl.col('is_treat')).height:,}")

    return compgcn_sampled, conve_sampled


def main():
    random.seed(42)  # For reproducibility

    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline")

    # Input paths
    compgcn_filtered_path = base_path / "scores_matrix_CompGCN_filtered.txt"
    conve_filtered_path = base_path / "scores_matrix_ConvE_filtered.txt"
    compgcn_test_path = base_path / "ConvE_top50_predict/test_triple_750_scores_compgcn.csv"
    conve_test_path = base_path / "ConvE_top50_predict/scores_test_ranked_750_ConvE.csv"

    # Step 1: Combine filtered files with test triples
    compgcn_combined = combine_compgcn_with_test(str(compgcn_filtered_path), str(compgcn_test_path))
    conve_combined = combine_conve_with_test(str(conve_filtered_path), str(conve_test_path))

    # Load test triples for overlap calculation
    test = pl.read_csv(str(conve_test_path))

    # Step 2: Sample false pairs
    compgcn_sampled, conve_sampled = sample_false_pairs(compgcn_combined, conve_combined, test)

    # Step 3: Save outputs
    print("\n=== Saving outputs ===")

    # Save combined (with test triples merged)
    compgcn_combined_path = base_path / "scores_matrix_CompGCN_combined.txt"
    compgcn_combined.write_csv(str(compgcn_combined_path))
    print(f"  CompGCN combined saved to: {compgcn_combined_path}")

    conve_combined_path = base_path / "scores_matrix_ConvE_combined.txt"
    conve_combined.write_csv(str(conve_combined_path))
    print(f"  ConvE combined saved to: {conve_combined_path}")

    # Save sampled subsets
    compgcn_sampled_path = base_path / "scores_matrix_CompGCN_sampled_test_overlap.txt"
    compgcn_sampled.write_csv(str(compgcn_sampled_path))
    print(f"  CompGCN sampled saved to: {compgcn_sampled_path}")

    conve_sampled_path = base_path / "scores_matrix_ConvE_sampled_test_overlap.txt"
    conve_sampled.write_csv(str(conve_sampled_path))
    print(f"  ConvE sampled saved to: {conve_sampled_path}")

    # Statistics
    print("\n=== Statistics ===")
    compgcn_false_sampled = compgcn_sampled.filter(pl.col("is_treat") == False)
    print(f"Drugs with sampled false pairs: {compgcn_false_sampled['head'].n_unique():,}")
    print(f"Diseases with sampled false pairs: {compgcn_false_sampled['tail'].n_unique():,}")

    drug_counts = compgcn_false_sampled.group_by("head").agg(pl.count().alias("count"))
    disease_counts = compgcn_false_sampled.group_by("tail").agg(pl.count().alias("count"))

    print(f"\nFalse pairs per drug - min: {drug_counts['count'].min()}, max: {drug_counts['count'].max()}, mean: {drug_counts['count'].mean():.1f}")
    print(f"False pairs per disease - min: {disease_counts['count'].min()}, max: {disease_counts['count'].max()}, mean: {disease_counts['count'].mean():.1f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
