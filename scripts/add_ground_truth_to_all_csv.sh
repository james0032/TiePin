#!/bin/bash

# Add ground truth indicator column to all TracIn CSV files in a folder
#
# Usage:
#   bash add_ground_truth_to_all_csv.sh <csv_folder> <ground_truth_jsonl> [output_suffix]
#
# Arguments:
#   csv_folder         - Directory containing TracIn CSV files
#   ground_truth_jsonl - Path to ground truth edges JSONL file
#   output_suffix      - Optional suffix for output files (default: "_with_gt")
#
# Example:
#   bash add_ground_truth_to_all_csv.sh dmdb_results ground_truth/drugmechdb_edges.jsonl
#   bash add_ground_truth_to_all_csv.sh dmdb_results ground_truth/drugmechdb_edges.jsonl "_gt"

# Note: Not using 'set -e' to allow processing all files even if one fails

# Check arguments
if [ "$#" -lt 2 ]; then
    echo "Error: Missing required arguments"
    echo ""
    echo "Usage: $0 <csv_folder> <ground_truth_jsonl> [output_suffix]"
    echo ""
    echo "Arguments:"
    echo "  csv_folder         - Directory containing TracIn CSV files"
    echo "  ground_truth_jsonl - Path to ground truth edges JSONL file"
    echo "  output_suffix      - Optional suffix for output files (default: '_with_gt')"
    echo ""
    echo "Example:"
    echo "  $0 dmdb_results ground_truth/drugmechdb_edges.jsonl"
    echo "  $0 dmdb_results ground_truth/drugmechdb_edges.jsonl '_gt'"
    exit 1
fi

CSV_FOLDER="$1"
GROUND_TRUTH_JSONL="$2"
OUTPUT_SUFFIX="${3:-_with_gt}"

# Validate inputs
if [ ! -d "$CSV_FOLDER" ]; then
    echo "Error: CSV folder not found: $CSV_FOLDER"
    exit 1
fi

if [ ! -f "$GROUND_TRUTH_JSONL" ]; then
    echo "Error: Ground truth JSONL file not found: $GROUND_TRUTH_JSONL"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Path to the Python script
PYTHON_SCRIPT="$PROJECT_ROOT/add_ground_truth_column.py"

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: add_ground_truth_column.py not found at: $PYTHON_SCRIPT"
    exit 1
fi

echo "=========================================="
echo "Add Ground Truth Column to TracIn CSV Files"
echo "=========================================="
echo "CSV folder:        $CSV_FOLDER"
echo "Ground truth file: $GROUND_TRUTH_JSONL"
echo "Output suffix:     $OUTPUT_SUFFIX"
echo "=========================================="
echo ""

# Find all CSV files in the folder
CSV_FILES=("$CSV_FOLDER"/*.csv)

# Check if any CSV files exist
if [ ! -e "${CSV_FILES[0]}" ]; then
    echo "Error: No CSV files found in $CSV_FOLDER"
    exit 1
fi

# Count files
TOTAL_FILES=${#CSV_FILES[@]}
echo "Found $TOTAL_FILES CSV file(s) to process"
echo ""

# Process each CSV file
PROCESSED=0
FAILED=0

for CSV_FILE in "${CSV_FILES[@]}"; do
    # Get filename without path
    FILENAME=$(basename "$CSV_FILE")

    # Skip if filename already has the suffix (avoid reprocessing)
    if [[ "$FILENAME" == *"$OUTPUT_SUFFIX.csv" ]]; then
        echo "⊘ Skipping $FILENAME (already processed)"
        continue
    fi

    # Generate output filename
    OUTPUT_FILE="${CSV_FILE%.csv}${OUTPUT_SUFFIX}.csv"

    echo "----------------------------------------"
    echo "Processing: $FILENAME"
    echo "Output:     $(basename "$OUTPUT_FILE")"
    echo ""

    # Run the Python script
    if python "$PYTHON_SCRIPT" \
        --tracin-csv "$CSV_FILE" \
        --ground-truth "$GROUND_TRUTH_JSONL" \
        --output "$OUTPUT_FILE"; then

        PROCESSED=$((PROCESSED + 1))
        echo ""
        echo "✓ Successfully processed $FILENAME"
    else
        FAILED=$((FAILED + 1))
        echo ""
        echo "✗ Failed to process $FILENAME"
    fi
    echo ""
done

echo "=========================================="
echo "Summary"
echo "=========================================="
echo "Total files found:  $TOTAL_FILES"
echo "Successfully processed: $PROCESSED"
echo "Failed: $FAILED"
echo "=========================================="

if [ $FAILED -gt 0 ]; then
    exit 1
fi

echo ""
echo "✓ All files processed successfully!"
