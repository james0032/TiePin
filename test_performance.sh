#!/bin/bash
# Test script to demonstrate performance profiling of filter_training_by_proximity_pyg.py

echo "=========================================="
echo "Performance Analysis Test"
echo "=========================================="
echo ""

# Check if required arguments are provided
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <train_file> <test_file> <cache_file>"
    echo ""
    echo "Example:"
    echo "  $0 train.txt test.txt graph_cache.pkl"
    echo ""
    exit 1
fi

TRAIN_FILE=$1
TEST_FILE=$2
CACHE_FILE=$3
OUTPUT_FILE="train_filtered_perf_test.txt"

echo "Configuration:"
echo "  Train file: $TRAIN_FILE"
echo "  Test file:  $TEST_FILE"
echo "  Cache file: $CACHE_FILE"
echo "  Output:     $OUTPUT_FILE"
echo ""

# Run 1: Without profiling (just timing breakdown)
echo "=========================================="
echo "RUN 1: Standard run with timing breakdown"
echo "=========================================="
python filter_training_by_proximity_pyg.py \
    --train "$TRAIN_FILE" \
    --test "$TEST_FILE" \
    --output "$OUTPUT_FILE" \
    --cache "$CACHE_FILE" \
    --n-hops 2 \
    --min-degree 2

echo ""
echo ""

# Run 2: With detailed profiling
echo "=========================================="
echo "RUN 2: Detailed profiling with cProfile"
echo "=========================================="
python filter_training_by_proximity_pyg.py \
    --train "$TRAIN_FILE" \
    --test "$TEST_FILE" \
    --output "$OUTPUT_FILE" \
    --cache "$CACHE_FILE" \
    --n-hops 2 \
    --min-degree 2 \
    --profile

echo ""
echo ""
echo "=========================================="
echo "Analysis Complete"
echo "=========================================="
echo "Profile data saved to: profile.stats"
echo ""
echo "To view detailed profile interactively:"
echo "  python -m pstats profile.stats"
echo ""
echo "Common pstats commands:"
echo "  stats 20           - Show top 20 functions"
echo "  sort cumulative    - Sort by cumulative time"
echo "  sort time          - Sort by internal time"
echo "  callers <func>     - Show what calls this function"
echo "  callees <func>     - Show what this function calls"
echo ""
