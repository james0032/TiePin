"""
Script to add 'is_treat' column to score matrix files based on treats.txt data.
Uses Polars for efficient processing of large files with join-based lookup.
"""

import polars as pl
from pathlib import Path


def load_treats_data(treats_path: str) -> pl.DataFrame:
    """
    Load treats.txt and return a DataFrame with Drug-Disease pairs.
    """
    treats_df = pl.read_csv(treats_path, separator="\t")
    print(f"Loaded {len(treats_df):,} treatment relationships from treats.txt")
    return treats_df


def add_is_treat_column_compgcn(
    input_path: str,
    output_path: str,
    treats_df: pl.DataFrame
) -> None:
    """
    Add is_treat column to CompGCN scores matrix using efficient join.

    CompGCN format:
    head_id,head,relation,tail_id,tail,raw_score,sigmoid_score

    head = Drug, tail = Disease
    """
    print(f"\nProcessing CompGCN file: {input_path}")

    # Create lookup DataFrame with just Drug-Disease pairs
    treats_lookup = treats_df.select([
        pl.col("Drug").alias("head"),
        pl.col("Disease").alias("tail")
    ]).with_columns(
        pl.lit(True).alias("is_treat")
    ).unique()

    # Read the CSV file using lazy evaluation for memory efficiency
    df = pl.scan_csv(input_path)

    # Left join with treats lookup to add is_treat column
    df = df.join(
        treats_lookup.lazy(),
        on=["head", "tail"],
        how="left"
    ).with_columns(
        pl.col("is_treat").fill_null(False)
    )

    # Collect and write output
    result = df.collect()
    print(f"Processed {len(result):,} rows")

    treat_count = result.filter(pl.col("is_treat")).height
    print(f"Found {treat_count:,} rows with is_treat=True")

    result.write_csv(output_path)
    print(f"Saved to: {output_path}")


def add_is_treat_column_conve(
    input_path: str,
    output_path: str,
    treats_df: pl.DataFrame
) -> None:
    """
    Add is_treat column to ConvE scores matrix using efficient join.

    ConvE format:
    head_id,head_label,relation_id,relation_label,tail_id,tail_label,score

    head_label = Drug, tail_label = Disease
    """
    print(f"\nProcessing ConvE file: {input_path}")

    # Create lookup DataFrame with just Drug-Disease pairs
    treats_lookup = treats_df.select([
        pl.col("Drug").alias("head_label"),
        pl.col("Disease").alias("tail_label")
    ]).with_columns(
        pl.lit(True).alias("is_treat")
    ).unique()

    # Read the CSV file using lazy evaluation for memory efficiency
    df = pl.scan_csv(input_path)

    # Left join with treats lookup to add is_treat column
    df = df.join(
        treats_lookup.lazy(),
        on=["head_label", "tail_label"],
        how="left"
    ).with_columns(
        pl.col("is_treat").fill_null(False)
    )

    # Collect and write output
    result = df.collect()
    print(f"Processed {len(result):,} rows")

    treat_count = result.filter(pl.col("is_treat")).height
    print(f"Found {treat_count:,} rows with is_treat=True")

    result.write_csv(output_path)
    print(f"Saved to: {output_path}")


def main():
    # Define paths
    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline")

    treats_path = base_path / "treats.txt"
    compgcn_input = base_path / "scores_matrix_CompGCN.txt"
    conve_input = base_path / "scores_matrix_ConvE.txt"

    # Output paths (same location, with _with_is_treat suffix)
    compgcn_output = base_path / "scores_matrix_CompGCN_with_is_treat.txt"
    conve_output = base_path / "scores_matrix_ConvE_with_is_treat.txt"

    # Load treats data
    treats_df = load_treats_data(str(treats_path))

    # Process CompGCN
    add_is_treat_column_compgcn(
        str(compgcn_input),
        str(compgcn_output),
        treats_df
    )

    # Process ConvE
    add_is_treat_column_conve(
        str(conve_input),
        str(conve_output),
        treats_df
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
