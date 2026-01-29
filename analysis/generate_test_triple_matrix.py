"""
Generate drug-disease matrix from the 750 test triples.

Takes unique drugs (head_label) and diseases (tail_label) from the test triples
and generates all drug x disease combinations.

Output format (TSV):
    head_label    predicate:38    tail_label
"""

import pandas as pd
from pathlib import Path


def main():
    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline")
    input_path = base_path / "ConvE_top50_predict" / "scores_test_ranked_750_ConvE.csv"
    output_path = base_path / "matrix_test750.txt"

    predicate = "predicate:38"

    # Load test triples
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"  Rows: {len(df):,}")

    # Get unique drugs and diseases
    drugs = sorted(df["head_label"].unique().tolist())
    diseases = sorted(df["tail_label"].unique().tolist())

    print(f"  Unique drugs (head_label): {len(drugs):,}")
    print(f"  Unique diseases (tail_label): {len(diseases):,}")

    total_pairs = len(drugs) * len(diseases)
    print(f"\n  Total pairs to generate: {len(drugs):,} x {len(diseases):,} = {total_pairs:,}")

    # Generate all pairs
    print(f"\nGenerating pairs...")
    with open(output_path, 'w') as f:
        for drug in drugs:
            for disease in diseases:
                f.write(f"{drug}\t{predicate}\t{disease}\n")

    print(f"\nOutput saved to: {output_path}")
    print(f"Total lines written: {total_pairs:,}")

    print("\nDone!")


if __name__ == "__main__":
    main()
