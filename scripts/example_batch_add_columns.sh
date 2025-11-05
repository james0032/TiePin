#!/bin/bash
# Example: Batch process all TracIn CSV files to add both IsGroundTruth and In_path columns

# Configuration
CSV_FOLDER="dmdb_results/batch_tracin_top50"
GROUND_TRUTH_JSONL="ground_truth/drugmechdb_edges.jsonl"
MECHANISTIC_PATHS_CSV="dedup_treats_mechanistic_paths.txt"
OUTPUT_SUFFIX="_annotated"

echo "This example will:"
echo "  1. Find all TracIn CSV files in: ${CSV_FOLDER}"
echo "  2. Add IsGroundTruth column using: ${GROUND_TRUTH_JSONL}"
echo "  3. Add In_path column using: ${MECHANISTIC_PATHS_CSV}"
echo "  4. Save output files with suffix: ${OUTPUT_SUFFIX}"
echo ""
echo "Example output filename:"
echo "  triple_000_CHEBI_17154_MONDO_0019975_tracin.csv"
echo "  → triple_000_CHEBI_17154_MONDO_0019975_tracin_annotated.csv"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the batch processing script
bash "${SCRIPT_DIR}/add_ground_truth_to_all_csv.sh" \
    "$CSV_FOLDER" \
    "$GROUND_TRUTH_JSONL" \
    "$MECHANISTIC_PATHS_CSV" \
    "$OUTPUT_SUFFIX"

echo ""
echo "Done! Check the output files in: ${CSV_FOLDER}"
echo ""
echo "Each file now has these additional columns:"
echo "  - IsGroundTruth: 1 if training edge is in ground truth JSONL"
echo "  - In_path: 1 if training edge connects nodes in mechanistic path"
echo ""
