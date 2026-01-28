"""
Filter score matrices to keep only pairs where drug or disease exists in the test triples.

For each score matrix:
1. Load the 750 test triples to get the set of drugs and diseases
2. Keep rows where:
   - is_treat=True (always keep known treatment pairs), OR
   - drug exists in test triples, OR
   - disease exists in test triples
3. Ensure both filtered files have the same pairs in the same order
"""

import polars as pl
from pathlib import Path


def load_test_triples(test_path: str) -> tuple[set[str], set[str]]:
    """
    Load test triples and return sets of drugs and diseases.

    Returns:
        Tuple of (drug_set, disease_set)
    """
    df = pl.read_csv(test_path)
    drugs = set(df["head_label"].to_list())
    diseases = set(df["tail_label"].to_list())
    print(f"Loaded test triples from: {test_path}")
    print(f"  Total test triples: {len(df):,}")
    print(f"  Unique drugs in test: {len(drugs):,}")
    print(f"  Unique diseases in test: {len(diseases):,}")
    return drugs, diseases


def filter_compgcn(
    input_path: str,
    test_drugs: set[str],
    test_diseases: set[str]
) -> pl.DataFrame:
    """
    Filter CompGCN scores matrix.

    CompGCN format columns: head_id, head, relation, tail_id, tail, raw_score, sigmoid_score, is_treat
    head = Drug, tail = Disease
    """
    print(f"\nLoading CompGCN file: {input_path}")
    df = pl.read_csv(input_path)
    print(f"  Total rows: {len(df):,}")

    # Filter: keep if is_treat=True OR drug in test OR disease in test
    filtered = df.filter(
        (pl.col("is_treat") == True) |
        (pl.col("head").is_in(test_drugs)) |
        (pl.col("tail").is_in(test_diseases))
    )

    print(f"  Filtered rows: {len(filtered):,}")
    print(f"  Rows with is_treat=True: {filtered.filter(pl.col('is_treat')).height:,}")

    return filtered


def filter_conve(
    input_path: str,
    test_drugs: set[str],
    test_diseases: set[str]
) -> pl.DataFrame:
    """
    Filter ConvE scores matrix.

    ConvE format columns: head_id, head_label, relation_id, relation_label, tail_id, tail_label, score, is_treat
    head_label = Drug, tail_label = Disease
    """
    print(f"\nLoading ConvE file: {input_path}")
    df = pl.read_csv(input_path)
    print(f"  Total rows: {len(df):,}")

    # Filter: keep if is_treat=True OR drug in test OR disease in test
    filtered = df.filter(
        (pl.col("is_treat") == True) |
        (pl.col("head_label").is_in(test_drugs)) |
        (pl.col("tail_label").is_in(test_diseases))
    )

    print(f"  Filtered rows: {len(filtered):,}")
    print(f"  Rows with is_treat=True: {filtered.filter(pl.col('is_treat')).height:,}")

    return filtered


def align_dataframes(
    compgcn_df: pl.DataFrame,
    conve_df: pl.DataFrame
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Align both DataFrames to have the same drug-disease pairs in the same order.

    Returns:
        Tuple of (aligned_compgcn, aligned_conve)
    """
    print("\nAligning DataFrames to have the same pairs...")

    # Create pair keys for both dataframes
    compgcn_pairs = set(
        zip(compgcn_df["head"].to_list(), compgcn_df["tail"].to_list())
    )
    conve_pairs = set(
        zip(conve_df["head_label"].to_list(), conve_df["tail_label"].to_list())
    )

    # Find common pairs
    common_pairs = compgcn_pairs & conve_pairs
    print(f"  CompGCN unique pairs: {len(compgcn_pairs):,}")
    print(f"  ConvE unique pairs: {len(conve_pairs):,}")
    print(f"  Common pairs: {len(common_pairs):,}")

    # Create a lookup DataFrame for ordering
    common_pairs_list = sorted(list(common_pairs))  # Sort for consistent ordering
    order_df = pl.DataFrame({
        "drug": [p[0] for p in common_pairs_list],
        "disease": [p[1] for p in common_pairs_list],
        "order_idx": list(range(len(common_pairs_list)))
    })

    # Filter and sort CompGCN
    compgcn_aligned = (
        compgcn_df
        .join(
            order_df,
            left_on=["head", "tail"],
            right_on=["drug", "disease"],
            how="inner"
        )
        .sort("order_idx")
        .drop("order_idx")
    )

    # Filter and sort ConvE
    conve_aligned = (
        conve_df
        .join(
            order_df,
            left_on=["head_label", "tail_label"],
            right_on=["drug", "disease"],
            how="inner"
        )
        .sort("order_idx")
        .drop("order_idx")
    )

    print(f"  Aligned CompGCN rows: {len(compgcn_aligned):,}")
    print(f"  Aligned ConvE rows: {len(conve_aligned):,}")

    return compgcn_aligned, conve_aligned


def main():
    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline")

    # Input paths
    test_path = base_path / "ConvE_top50_predict/scores_test_ranked_750_ConvE.csv"
    compgcn_input = base_path / "scores_matrix_CompGCN_with_is_treat.txt"
    conve_input = base_path / "scores_matrix_ConvE_with_is_treat.txt"

    # Output paths
    compgcn_output = base_path / "scores_matrix_CompGCN_filtered.txt"
    conve_output = base_path / "scores_matrix_ConvE_filtered.txt"

    # Load test triples
    test_drugs, test_diseases = load_test_triples(str(test_path))

    # Filter both matrices
    compgcn_filtered = filter_compgcn(str(compgcn_input), test_drugs, test_diseases)
    conve_filtered = filter_conve(str(conve_input), test_drugs, test_diseases)

    # Align both DataFrames to have the same pairs in the same order
    compgcn_aligned, conve_aligned = align_dataframes(compgcn_filtered, conve_filtered)

    # Save outputs
    print(f"\nSaving outputs...")
    compgcn_aligned.write_csv(str(compgcn_output))
    print(f"  CompGCN saved to: {compgcn_output}")

    conve_aligned.write_csv(str(conve_output))
    print(f"  ConvE saved to: {conve_output}")

    # Verify alignment
    print("\nVerification:")
    print(f"  CompGCN first 3 pairs: {list(zip(compgcn_aligned['head'].head(3).to_list(), compgcn_aligned['tail'].head(3).to_list()))}")
    print(f"  ConvE first 3 pairs: {list(zip(conve_aligned['head_label'].head(3).to_list(), conve_aligned['tail_label'].head(3).to_list()))}")

    print("\nDone!")


if __name__ == "__main__":
    main()
