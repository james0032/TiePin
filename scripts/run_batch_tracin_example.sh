#!/bin/bash
# Example script to run batch TracIn analysis on top test triples
#
# This script processes all 25 top test triples with automatic filtering
# and TracIn analysis, generating CSV outputs with influence scores.

# Configuration - UPDATE THESE PATHS
MODEL_PATH="/workspace/data/robokop/CGGD_alltreat/model/conve/checkpoints/checkpoint_epoch_0000016.pt"
TRAIN_FILE="/workspace/data/robokop/CGGD_alltreat/train.txt"
ENTITY_TO_ID="/workspace/data/robokop/CGGD_alltreat/processed/node_dict.txt"
RELATION_TO_ID="/workspace/data/robokop/CGGD_alltreat/processed/rel_dict.txt"
EDGE_MAP="/workspace/data/robokop/CGGD_alltreat/edge_map.json"
NODE_NAME_DICT="/workspace/data/robokop/CGGD_alltreat/processed/node_name_dict.txt"
GRAPH_CACHE="/workspace/data/robokop/CGGD_alltreat/train_graph_cache.pkl"

# Test triples file
TEST_TRIPLES="/workspace/data/robokop/CGGD_alltreat/scores_test_top50.txt"

# Output directory
OUTPUT_DIR="/workspace/data/robokop/CGGD_alltreat/results/batch_tracin_top50"

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
echo "  - N-hops: 2"
echo "  - Min degree: 2"
echo "  - Strict hop constraint: enabled"
echo "  - Batch size: 16"
echo "  - Top-k influences: 100"
echo "  - Device: cuda"
echo "  - Last layers only: enabled (2 layers for speed)"
echo ""
echo "Expected time: ~8-12 minutes for 25 triples"
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
    --n-hops 2 \
    --min-degree 1 \
    --strict-hop-constraint \
    --cache "${GRAPH_CACHE}" \
    --device cuda \
    --batch-size 16 \
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
