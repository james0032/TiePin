#!/bin/bash
# Example script to add both ground truth and mechanistic path columns to TracIn CSV

# Example usage for the CHEBI:17154 -> MONDO:0019975 test triple

TRACIN_CSV="dmdb_results/triple_000_CHEBI_17154_MONDO_0019975_tracin.csv"
GROUND_TRUTH_JSONL="ground_truth/drugmechdb_edges.jsonl"
MECHANISTIC_PATHS_CSV="dedup_treats_mechanistic_paths.txt"
OUTPUT_CSV="dmdb_results/triple_000_CHEBI_17154_MONDO_0019975_tracin_with_gt_and_path.csv"

echo "Adding ground truth and mechanistic path columns to TracIn CSV..."
echo ""
echo "Input files:"
echo "  TracIn CSV:          ${TRACIN_CSV}"
echo "  Ground truth JSONL:  ${GROUND_TRUTH_JSONL}"
echo "  Mechanistic paths:   ${MECHANISTIC_PATHS_CSV}"
echo ""
echo "Output:"
echo "  ${OUTPUT_CSV}"
echo ""

python add_ground_truth_column.py \
    --tracin-csv "${TRACIN_CSV}" \
    --ground-truth "${GROUND_TRUTH_JSONL}" \
    --mechanistic-paths "${MECHANISTIC_PATHS_CSV}" \
    --output "${OUTPUT_CSV}"

echo ""
echo "Done! Check the output CSV for the new columns:"
echo "  - IsGroundTruth: 1 if training edge is in ground truth, 0 otherwise"
echo "  - In_path: 1 if training edge connects nodes in the mechanistic path, 0 otherwise"
echo ""
echo "Example query to find edges in mechanistic paths:"
echo "  grep ',1$' ${OUTPUT_CSV} | head -10"
echo ""
