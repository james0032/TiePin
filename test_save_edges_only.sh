#!/bin/bash

# Test script to demonstrate the --save-edges-only flag

TRAIN_FILE="path/to/your/train.txt"
TEST_FILE="path/to/your/test.txt"
OUTPUT_DIR="test_output"

mkdir -p $OUTPUT_DIR

echo "================================================"
echo "Test 1: Default behavior (save all triples)"
echo "================================================"

python filter_training_igraph.py \
    --train $TRAIN_FILE \
    --test $TEST_FILE \
    --output $OUTPUT_DIR/filtered_all_triples.txt \
    --n-hops 3 \
    --path-filtering \
    --strict-path-filtering

echo ""
echo "================================================"
echo "Test 2: Save edges only (one triple per edge)"
echo "================================================"

python filter_training_igraph.py \
    --train $TRAIN_FILE \
    --test $TEST_FILE \
    --output $OUTPUT_DIR/filtered_edges_only.txt \
    --n-hops 3 \
    --path-filtering \
    --strict-path-filtering \
    --save-edges-only

echo ""
echo "================================================"
echo "Comparison:"
echo "================================================"

echo "All triples: $(wc -l < $OUTPUT_DIR/filtered_all_triples.txt) lines"
echo "Edges only:  $(wc -l < $OUTPUT_DIR/filtered_edges_only.txt) lines"
