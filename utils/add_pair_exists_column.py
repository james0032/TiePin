#!/usr/bin/env python3
"""
Add a column to indicate if Drug-Disease pair exists in a reference file.
"""
import csv
import sys


def load_drug_disease_pairs(reference_file: str) -> set:
    """Load drug-disease pairs from reference file.

    Args:
        reference_file: Path to TSV file with Drug, Predicate, Disease columns

    Returns:
        Set of (drug, disease) tuples
    """
    pairs = set()

    with open(reference_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Drug'):  # Skip header if present
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                drug = parts[0].strip()
                disease = parts[2].strip()  # Disease is in 3rd column
                pairs.add((drug, disease))

    print(f"Loaded {len(pairs)} drug-disease pairs from reference file")
    return pairs


def add_exists_column(input_file: str, reference_file: str, output_file: str,
                     column_name: str = "InReferenceSet"):
    """Add a column indicating if drug-disease pair exists in reference file.

    Args:
        input_file: Path to input CSV file with Drug, Disease, Intermediate Nodes
        reference_file: Path to reference TSV file with Drug, Predicate, Disease
        output_file: Path to output CSV file
        column_name: Name of the new column (default: "InReferenceSet")
    """
    # Load reference pairs
    reference_pairs = load_drug_disease_pairs(reference_file)

    # Process input file and add column
    with open(input_file, 'r') as f_in, open(output_file, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Read and write header
        header = next(reader)
        new_header = header + [column_name]
        writer.writerow(new_header)

        matched_count = 0
        total_count = 0

        # Process each row
        for row in reader:
            if len(row) < 2:
                print(f"Warning: Skipping invalid row: {row}", file=sys.stderr)
                continue

            drug = row[0].strip()
            disease = row[1].strip()

            # Check if pair exists in reference
            pair_exists = (drug, disease) in reference_pairs

            # Add column value (1 for exists, 0 for not exists)
            new_row = row + [1 if pair_exists else 0]
            writer.writerow(new_row)

            if pair_exists:
                matched_count += 1
            total_count += 1

    print(f"Processed {total_count} rows")
    print(f"Found {matched_count} matching pairs ({matched_count/total_count*100:.2f}%)")
    print(f"Output written to: {output_file}")


def main():
    if len(sys.argv) not in [4, 5]:
        print("Usage: python add_pair_exists_column.py <input_csv> <reference_tsv> <output_csv> [column_name]")
        print("\nArguments:")
        print("  input_csv     : CSV file with Drug, Disease, [Intermediate Nodes] columns")
        print("  reference_tsv : TSV file with Drug, Predicate, Disease columns")
        print("  output_csv    : Output CSV file with added column")
        print("  column_name   : (Optional) Name for new column (default: 'InReferenceSet')")
        print("\nExample:")
        print("  python add_pair_exists_column.py paths.csv treats.txt output.csv")
        print("  python add_pair_exists_column.py paths.csv treats.txt output.csv InTreatSet")
        sys.exit(1)

    input_file = sys.argv[1]
    reference_file = sys.argv[2]
    output_file = sys.argv[3]
    column_name = sys.argv[4] if len(sys.argv) == 5 else "InReferenceSet"

    add_exists_column(input_file, reference_file, output_file, column_name)


if __name__ == '__main__':
    main()
