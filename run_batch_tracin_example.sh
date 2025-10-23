#!/bin/bash
# Example script to run batch TracIn analysis on top test triples
#
# This script processes all 25 top test triples with automatic filtering
# and TracIn analysis, generating CSV outputs with influence scores.

# Configuration - UPDATE THESE PATHS
MODEL_PATH="/workspace/data/robokop/CGGD_alltreat/checkpoints/conve_checkpoint_004.pt"
TRAIN_FILE="/workspace/data/robokop/CGGD_alltreat/train.txt"
ENTITY_TO_ID="/workspace/data/robokop/CGGD_alltreat/processed/entity_to_id.tsv"
RELATION_TO_ID="/workspace/data/robokop/CGGD_alltreat/processed/elation_to_id.tsv"
EDGE_MAP="/workspace/data/robokop/CGGD_alltreat/edge_map.json"
NODE_NAME_DICT="/workspace/data/robokop/CGGD_alltreat/node_name_dict.txt"
GRAPH_CACHE="/workspace/data/robokop/CGGD_alltreat/processed/train_graph_cache.pkl"

# Test triples file
TEST_TRIPLES="/workspace/data/robokop/CGGD_alltreat/results/20251017_top_test_triples.txt"

# Output directory
OUTPUT_DIR="/workspace/data/robokop/CGGD_alltreat/results/batch_tracin_top25"

# Run batch TracIn analysis
cd "$(dirname "$0")"

echo "========================================"
echo "Batch TracIn Analysis"
echo "========================================"
echo "Test triples: examples/${TEST_TRIPLES}"
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Configuration:"
echo "  - N-hops: 2"
echo "  - Min degree: 2"
echo "  - Top-k influences: 100"
echo "  - Device: cuda"
echo "  - Last layers only: enabled"
echo ""
echo "Expected time: ~8-12 minutes for 25 triples"
echo "========================================"
echo ""

python batch_tracin_with_filtering.py \
    --test-triples "${TEST_TRIPLES}" \
    --model-path "${MODEL_PATH}" \
    --train "${TRAIN_FILE}" \
    --entity-to-id "${ENTITY_TO_ID}" \
    --relation-to-id "${RELATION_TO_ID}" \
    --edge-map "${EDGE_MAP}" \
    --node-name-dict "${NODE_NAME_DICT}" \
    --output-dir "${OUTPUT_DIR}" \
    --n-hops 1 \
    --min-degree 2 \
    --cache "${GRAPH_CACHE}" \
    --device cuda \
    --batch-size 4 \
    --num-last-layers 2

echo ""
echo "========================================"
echo "Batch processing complete!"
echo "========================================"
echo ""
echo "Output files:"
echo "  - Filtered training: ${OUTPUT_DIR}/filtered_training/"
echo "  - TracIn CSV files: ${OUTPUT_DIR}/*_tracin.csv"
echo "  - TracIn JSON files: ${OUTPUT_DIR}/*_tracin.json"
echo "  - Summary: ${OUTPUT_DIR}/batch_tracin_summary.json"
echo ""
echo "To analyze results:"
echo "  python -c 'import json; print(json.load(open(\"${OUTPUT_DIR}/batch_tracin_summary.json\", \"r\")))'"
echo ""
