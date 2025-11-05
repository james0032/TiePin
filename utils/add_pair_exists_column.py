#!/usr/bin/env python3
"""
Add columns to both files indicating if Drug-Disease pairs exist in the other file.
"""
import csv
import sys


def load_drug_disease_pairs_from_tsv(tsv_file: str) -> set:
    """Load drug-disease pairs from TSV file.

    Args:
        tsv_file: Path to TSV file with Drug, Predicate, Disease columns

    Returns:
        Set of (drug, disease) tuples
    """
    pairs = set()

    with open(tsv_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('Drug'):  # Skip header if present
                continue

            parts = line.split('\t')
            if len(parts) >= 3:
                drug = parts[0].strip()
                disease = parts[2].strip()  # Disease is in 3rd column
                pairs.add((drug, disease))

    print(f"Loaded {len(pairs)} drug-disease pairs from TSV file")
    return pairs


def load_drug_disease_pairs_from_csv(csv_file: str) -> set:
    """Load drug-disease pairs from CSV file.

    Args:
        csv_file: Path to CSV file with Drug, Disease, [Intermediate Nodes] columns

    Returns:
        Set of (drug, disease) tuples (excluding pairs with empty intermediate nodes)
    """
    pairs = set()
    skipped_empty = 0

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header

        for row in reader:
            if len(row) >= 3:
                drug = row[0].strip()
                disease = row[1].strip()
                intermediate_nodes = row[2].strip()

                # Skip pairs with empty intermediate nodes list
                if intermediate_nodes == "[]":
                    skipped_empty += 1
                    continue

                pairs.add((drug, disease))
            elif len(row) >= 2:
                # If intermediate nodes column is missing, still add the pair
                drug = row[0].strip()
                disease = row[1].strip()
                pairs.add((drug, disease))

    print(f"Loaded {len(pairs)} drug-disease pairs from CSV file (skipped {skipped_empty} pairs with empty intermediate nodes)")
    return pairs


def add_column_to_csv(input_csv: str, tsv_pairs: set, output_csv: str,
                     column_name: str = "InTSV") -> int:
    """Add a column to CSV indicating if drug-disease pair exists in TSV file.

    Args:
        input_csv: Path to input CSV file with Drug, Disease, Intermediate Nodes
        tsv_pairs: Set of (drug, disease) tuples from TSV file
        output_csv: Path to output CSV file
        column_name: Name of the new column (default: "InTSV")

    Returns:
        Number of matching pairs (count of 1s)
    """
    matched_count = 0
    total_count = 0

    with open(input_csv, 'r') as f_in, open(output_csv, 'w', newline='') as f_out:
        reader = csv.reader(f_in)
        writer = csv.writer(f_out)

        # Read and write header
        header = next(reader)
        new_header = header + [column_name]
        writer.writerow(new_header)

        # Process each row
        for row in reader:
            if len(row) < 2:
                print(f"Warning: Skipping invalid row in CSV: {row}", file=sys.stderr)
                continue

            drug = row[0].strip()
            disease = row[1].strip()

            # Check if pair exists in TSV
            pair_exists = (drug, disease) in tsv_pairs

            # Add column value (1 for exists, 0 for not exists)
            new_row = row + [1 if pair_exists else 0]
            writer.writerow(new_row)

            if pair_exists:
                matched_count += 1
            total_count += 1

    print(f"CSV: Processed {total_count} rows, found {matched_count} matching pairs ({matched_count/total_count*100:.2f}%)")
    return matched_count


def add_column_to_tsv(input_tsv: str, csv_pairs: set, output_tsv: str,
                     column_name: str = "InCSV") -> int:
    """Add a column to TSV indicating if drug-disease pair exists in CSV file.

    Args:
        input_tsv: Path to input TSV file with Drug, Predicate, Disease
        csv_pairs: Set of (drug, disease) tuples from CSV file
        output_tsv: Path to output TSV file
        column_name: Name of the new column (default: "InCSV")

    Returns:
        Number of matching pairs (count of 1s)
    """
    matched_count = 0
    total_count = 0

    with open(input_tsv, 'r') as f_in, open(output_tsv, 'w') as f_out:
        for line in f_in:
            line = line.strip()

            # Handle header
            if line.startswith('Drug'):
                f_out.write(f"{line}\t{column_name}\n")
                continue

            if not line:
                f_out.write("\n")
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                print(f"Warning: Skipping invalid row in TSV: {parts}", file=sys.stderr)
                f_out.write(f"{line}\n")
                continue

            drug = parts[0].strip()
            disease = parts[2].strip()  # Disease is in 3rd column

            # Check if pair exists in CSV
            pair_exists = (drug, disease) in csv_pairs

            # Add column value
            new_line = f"{line}\t{1 if pair_exists else 0}\n"
            f_out.write(new_line)

            if pair_exists:
                matched_count += 1
            total_count += 1

    print(f"TSV: Processed {total_count} rows, found {matched_count} matching pairs ({matched_count/total_count*100:.2f}%)")
    return matched_count


def filter_tsv_matches_only(input_tsv_with_column: str, output_filtered_tsv: str) -> int:
    """Create a filtered TSV containing only rows where the pair exists in both files.

    Output contains only the first 3 columns (Drug, Predicate, Disease) without the InCSV column.

    Args:
        input_tsv_with_column: Path to TSV file that already has the existence column
        output_filtered_tsv: Path to output filtered TSV file

    Returns:
        Number of rows in filtered output
    """
    filtered_count = 0

    with open(input_tsv_with_column, 'r') as f_in, open(output_filtered_tsv, 'w') as f_out:
        for line in f_in:
            line = line.strip()

            # Write header (only first 3 columns)
            if line.startswith('Drug'):
                parts = line.split('\t')
                header = '\t'.join(parts[:3])
                f_out.write(f"{header}\n")
                continue

            if not line:
                continue

            parts = line.split('\t')
            if len(parts) < 4:  # Need at least Drug, Predicate, Disease, and the new column
                continue

            # Check if the last column (existence indicator) is 1
            if parts[-1] == '1':
                # Only write first 3 columns (Drug, Predicate, Disease)
                output_line = '\t'.join(parts[:3])
                f_out.write(f"{output_line}\n")
                filtered_count += 1

    print(f"Filtered TSV: {filtered_count} rows with matching pairs (first 3 columns only)")
    return filtered_count


def process_both_files(input_csv: str, input_tsv: str, output_csv: str, output_tsv: str,
                       csv_column_name: str = "InTSV", tsv_column_name: str = "InCSV",
                       filtered_tsv: str = None):
    """Process both files and add existence columns to each.

    Args:
        input_csv: Path to input CSV file
        input_tsv: Path to input TSV file
        output_csv: Path to output CSV file
        output_tsv: Path to output TSV file
        csv_column_name: Name of column to add to CSV (default: "InTSV")
        tsv_column_name: Name of column to add to TSV (default: "InCSV")
        filtered_tsv: (Optional) Path to output filtered TSV with only matching pairs
    """
    print("=" * 80)
    print("Loading drug-disease pairs from both files...")
    print("=" * 80)

    # Load pairs from both files
    tsv_pairs = load_drug_disease_pairs_from_tsv(input_tsv)
    csv_pairs = load_drug_disease_pairs_from_csv(input_csv)

    print("\n" + "=" * 80)
    print("Adding columns to both files...")
    print("=" * 80)

    # Add column to CSV file
    csv_matched = add_column_to_csv(input_csv, tsv_pairs, output_csv, csv_column_name)
    print(f"Output written to: {output_csv}")

    # Add column to TSV file
    tsv_matched = add_column_to_tsv(input_tsv, csv_pairs, output_tsv, tsv_column_name)
    print(f"Output written to: {output_tsv}")

    # Create filtered TSV if requested
    if filtered_tsv:
        print("\n" + "=" * 80)
        print("Creating filtered TSV with matching pairs only...")
        print("=" * 80)
        filtered_count = filter_tsv_matches_only(output_tsv, filtered_tsv)
        print(f"Output written to: {filtered_tsv}")

    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)
    print(f"CSV matched pairs (1s): {csv_matched}")
    print(f"TSV matched pairs (1s): {tsv_matched}")

    if filtered_tsv:
        print(f"Filtered TSV rows: {filtered_count}")

    if csv_matched == tsv_matched:
        print(f"✓ SUCCESS: Match counts are equal ({csv_matched})")
        if filtered_tsv and filtered_count != csv_matched:
            print(f"⚠ WARNING: Filtered count ({filtered_count}) differs from match count ({csv_matched})")
            print(f"  This could indicate duplicate pairs in the TSV file.")
    else:
        diff = abs(csv_matched - tsv_matched)
        print(f"⚠ WARNING: Match counts differ by {diff}!")
        print(f"  This could indicate duplicate pairs in one of the files.")

    print("=" * 80)


def main():
    if len(sys.argv) not in [5, 6, 7, 8]:
        print("Usage: python add_pair_exists_column.py <input_csv> <input_tsv> <output_csv> <output_tsv> [filtered_tsv] [csv_col_name] [tsv_col_name]")
        print("\nArguments:")
        print("  input_csv      : CSV file with Drug, Disease, [Intermediate Nodes] columns")
        print("  input_tsv      : TSV file with Drug, Predicate, Disease columns")
        print("  output_csv     : Output CSV file with added column")
        print("  output_tsv     : Output TSV file with added column")
        print("  filtered_tsv   : (Optional) Output TSV file with only matching pairs (rows with 1)")
        print("  csv_col_name   : (Optional) Name for column added to CSV (default: 'InTSV')")
        print("  tsv_col_name   : (Optional) Name for column added to TSV (default: 'InCSV')")
        print("\nExamples:")
        print("  # Basic usage (no filtered output)")
        print("  python add_pair_exists_column.py paths.csv treats.txt paths_out.csv treats_out.txt")
        print("")
        print("  # With filtered output")
        print("  python add_pair_exists_column.py paths.csv treats.txt paths_out.csv treats_out.txt treats_filtered.txt")
        print("")
        print("  # With custom column names")
        print("  python add_pair_exists_column.py paths.csv treats.txt paths_out.csv treats_out.txt treats_filtered.txt InTreat InPaths")
        print("\nThe script will:")
        print("  1. Add a column to the CSV indicating if each pair exists in the TSV")
        print("  2. Add a column to the TSV indicating if each pair exists in the CSV")
        print("  3. (Optional) Create a filtered TSV with only rows where pairs exist in both files")
        print("  4. Verify that both files have the same count of matching pairs (1s)")
        sys.exit(1)

    input_csv = sys.argv[1]
    input_tsv = sys.argv[2]
    output_csv = sys.argv[3]
    output_tsv = sys.argv[4]

    # Parse optional arguments
    filtered_tsv = None
    csv_column_name = "InTSV"
    tsv_column_name = "InCSV"

    if len(sys.argv) >= 6:
        filtered_tsv = sys.argv[5]
    if len(sys.argv) >= 7:
        csv_column_name = sys.argv[6]
    if len(sys.argv) >= 8:
        tsv_column_name = sys.argv[7]

    process_both_files(input_csv, input_tsv, output_csv, output_tsv,
                      csv_column_name, tsv_column_name, filtered_tsv)


if __name__ == '__main__':
    main()
