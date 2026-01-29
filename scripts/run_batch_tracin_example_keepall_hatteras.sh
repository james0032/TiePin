#!/bin/bash
# Example script to run batch TracIn analysis on top test triples
#
# This script processes all 25 top test triples with automatic filtering
# and TracIn analysis, generating CSV outputs with influence scores.

# Configuration - UPDATE THESE PATHS
MODEL_PATH="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/models/conve/checkpoints/checkpoint_epoch_0000050.pt"
TRAIN_FILE="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/train.txt"
ENTITY_TO_ID="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/processed/node_dict.txt"
RELATION_TO_ID="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/processed/rel_dict.txt"
EDGE_MAP="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/edge_map.json"
NODE_NAME_DICT="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/processed/node_name_dict.txt"
GRAPH_CACHE="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/train_graph_cache.pkl"

# Test triples file
TEST_TRIPLES="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/test.txt"

# Output directory
OUTPUT_DIR="/projects/aixb/jchung/everycure/influence_estimate/robokop/keepall/results/batch_tracin_750"

# Run batch TracIn analysis
# Change to the parent directory (where batch_tracin_with_filtering.py is located)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/.."

echo "========================================"
echo "Batch TracIn Analysis"
echo "========================================"
echo "Test triples: ${TEST_TRIPLES}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Configuration:"
echo "  - Filter method: igraph (fast + transparent, C implementation)"
echo "  - N-hops: 2"
echo "  - Min degree: 1"
echo "  - Path filtering: enabled"
echo "  - Max total path length: 3"
echo "  - Batch size: 32"
echo "  - Top-k influences: 100"
echo "  - Device: cuda"
echo "  - Last layers only: enabled (2 layers for speed)"
echo "  - Mixed precision (FP16): ENABLED (2x memory + 2x speed!)"
echo "  - Optimized TracIn: ENABLED (20-40x faster with vectorized gradients + test caching!)"
echo "  - Torch compile: ENABLED (1.5x additional speedup)"
echo "  - Resume mode: ENABLED (skips already completed triples)"
echo ""
echo "Expected time: ~4-6 minutes for 25 triples (with FP16 optimization)"
echo "========================================"
echo ""

# Create output directory for logs
mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/batch_tracin_run_$(date +%Y%m%d_%H%M%S).log"

echo "All output will be logged to: ${LOG_FILE}"
echo ""

# Run with tee to show output and save to log file simultaneously
python batch_tracin_with_filtering.py \
    --test-triples "${TEST_TRIPLES}" \
    --model-path "${MODEL_PATH}" \
    --train "${TRAIN_FILE}" \
    --entity-to-id "${ENTITY_TO_ID}" \
    --relation-to-id "${RELATION_TO_ID}" \
    --edge-map "${EDGE_MAP}" \
    --node-name-dict "${NODE_NAME_DICT}" \
    --output-dir "${OUTPUT_DIR}" \
    --filter-method igraph \
    --cache "${GRAPH_CACHE}" \
    --n-hops 2 \
    --min-degree 1 \
    --path-filtering \
    --max-total-path-length 3 \
    --strict-path-filtering \
    --device cuda \
    --batch-size 32 \
    --use-mixed-precision \
    --use-optimized-tracin \
    --use-torch-compile \
    --skip-existing \
    2>&1 | tee "${LOG_FILE}"


echo ""
echo "========================================"
echo "Batch processing complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - Log file: ${LOG_FILE}"
echo "  - Filtered training: ${OUTPUT_DIR}/filtered_training/"
echo "  - TracIn CSV files: ${OUTPUT_DIR}/*_tracin.csv"
echo "  - TracIn JSON files: ${OUTPUT_DIR}/*_tracin.json"
echo "  - Summary: ${OUTPUT_DIR}/batch_tracin_summary.json"
echo ""
echo "To view log file:"
echo "  less ${LOG_FILE}"
echo ""
echo "To analyze results:"
echo "  python -c 'import json; print(json.load(open(\"${OUTPUT_DIR}/batch_tracin_summary.json\", \"r\")))'"
echo ""
