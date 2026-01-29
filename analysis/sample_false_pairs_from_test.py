"""
Sample is_treat=False pairs from the 750 test triples.

Workflow:
1. Combine filtered score matrices with the 750 test triples (remove duplicates)
2. Separate into is_treat=True and is_treat=False
3. For is_treat=True: only keep pairs that exist in test_triple_750_scores_compgcn.csv
4. For is_treat=False: get unique disease list from test_triple_750_scores_compgcn.csv,
   then for each disease sample 10 is_treat=False triples
5. Combine positives + sampled negatives and save
"""

import polars as pl
from pathlib import Path
import random

SAMPLE_SIZE = 20


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


def sample_and_filter(
    compgcn: pl.DataFrame,
    conve: pl.DataFrame,
    test_compgcn: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    1. is_treat=True: keep only pairs in test_triple_750_scores_compgcn.csv
    2. is_treat=False: for each unique disease in the 750 test triples,
       sample 10 is_treat=False triples
    """
    # Build test pairs lookup
    test_pairs_df = test_compgcn.select([
        pl.col("head").alias("drug"),
        pl.col("tail").alias("disease"),
    ]).unique()

    test_diseases = test_compgcn["tail"].unique().to_list()

    print("\n=== Filtering positives to 750 test triples ===")
    print(f"  Unique test pairs: {len(test_pairs_df):,}")
    print(f"  Unique test diseases: {len(test_diseases):,}")

    # --- Separate positives and negatives ---
    compgcn_pos = compgcn.filter(pl.col("is_treat") == True)
    compgcn_neg = compgcn.filter(pl.col("is_treat") == False)
    print(f"\n  CompGCN is_treat=True: {len(compgcn_pos):,}")
    print(f"  CompGCN is_treat=False: {len(compgcn_neg):,}")

    # --- Step 1: Filter positives to 750 test triples ---
    compgcn_pos_filtered = compgcn_pos.join(
        test_pairs_df,
        left_on=["head", "tail"],
        right_on=["drug", "disease"],
        how="inner"
    )
    print(f"  CompGCN is_treat=True after filter: {len(compgcn_pos_filtered):,}")

    # --- Step 2: For each unique disease in test triples, sample 10 negatives ---
    print(f"\nSampling {SAMPLE_SIZE} is_treat=False triples for each test disease...")
    all_sampled_pairs = set()
    for disease in test_diseases:
        disease_neg = compgcn_neg.filter(pl.col("tail") == disease)
        pairs = list(zip(disease_neg["head"].to_list(), disease_neg["tail"].to_list()))
        sample_size = min(SAMPLE_SIZE, len(pairs))
        if sample_size > 0:
            sampled = random.sample(pairs, sample_size)
            all_sampled_pairs.update(sampled)

    print(f"  Total unique sampled negative pairs: {len(all_sampled_pairs):,}")

    # --- Build final pair list ---
    pos_pairs = list(zip(compgcn_pos_filtered["head"].to_list(), compgcn_pos_filtered["tail"].to_list()))
    pos_df = pl.DataFrame({"drug": [p[0] for p in pos_pairs], "disease": [p[1] for p in pos_pairs]})
    neg_df = pl.DataFrame({"drug": [p[0] for p in all_sampled_pairs], "disease": [p[1] for p in all_sampled_pairs]})
    all_pairs_df = pl.concat([pos_df, neg_df]).unique()

    # --- Filter both CompGCN and ConvE to selected pairs ---
    compgcn_sampled = compgcn.join(
        all_pairs_df,
        left_on=["head", "tail"],
        right_on=["drug", "disease"],
        how="inner"
    ).sort(["head", "tail"])

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
    compgcn_filtered_path = base_path / "scores_matrix_CompGCN_with_is_treat.txt"
    conve_filtered_path = base_path / "scores_matrix_ConvE_with_is_treat.txt"
    compgcn_test_path = base_path / "ConvE_top50_predict/test_triple_750_scores_compgcn.csv"
    conve_test_path = base_path / "ConvE_top50_predict/scores_test_ranked_750_ConvE.csv"

    # Step 1: Combine filtered files with test triples (deduplicate)
    compgcn_combined = combine_compgcn_with_test(str(compgcn_filtered_path), str(compgcn_test_path))
    conve_combined = combine_conve_with_test(str(conve_filtered_path), str(conve_test_path))

    # Step 2+3: Filter positives to test triples, sample negatives per test disease
    test_compgcn = pl.read_csv(str(compgcn_test_path))
    compgcn_sampled, conve_sampled = sample_and_filter(compgcn_combined, conve_combined, test_compgcn)

    # Step 4: Save outputs
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
