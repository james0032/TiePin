#!/usr/bin/env python3
"""
Analyze edges.jsonl file to calculate frequency statistics.

This script reads a JSONL file containing edge data and calculates:
1. Frequency of each predicate
2. Frequency of each primary_knowledge_source
"""

import json
from collections import Counter
from pathlib import Path
import argparse


def analyze_edges(input_file: str):
    """
    Analyze edges.jsonl file and print frequency statistics.

    Args:
        input_file: Path to the edges.jsonl file
    """
    predicate_counter = Counter()
    knowledge_source_counter = Counter()

    total_lines = 0

    # Read and process the JSONL file
    with open(input_file, 'r') as f:
        for line in f:
            total_lines += 1
            try:
                edge = json.loads(line.strip())

                # Count predicates
                if 'predicate' in edge:
                    predicate_counter[edge['predicate']] += 1

                # Count primary knowledge sources
                if 'primary_knowledge_source' in edge:
                    knowledge_source_counter[edge['primary_knowledge_source']] += 1

            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse line {total_lines}: {e}")
                continue

    # Print results
    print(f"\n{'='*60}")
    print(f"EDGE ANALYSIS SUMMARY")
    print(f"{'='*60}")
    print(f"Total edges processed: {total_lines:,}")
    print()

    # Print predicate frequencies
    print(f"\n{'='*60}")
    print(f"PREDICATE FREQUENCIES")
    print(f"{'='*60}")
    print(f"{'Predicate':<50} {'Count':>10}")
    print(f"{'-'*50} {'-'*10}")

    for predicate, count in predicate_counter.most_common():
        percentage = (count / total_lines) * 100
        print(f"{predicate:<50} {count:>10,} ({percentage:>5.2f}%)")

    print(f"\nTotal unique predicates: {len(predicate_counter)}")

    # Print knowledge source frequencies
    print(f"\n{'='*60}")
    print(f"PRIMARY KNOWLEDGE SOURCE FREQUENCIES")
    print(f"{'='*60}")
    print(f"{'Knowledge Source':<50} {'Count':>10}")
    print(f"{'-'*50} {'-'*10}")

    for source, count in knowledge_source_counter.most_common():
        percentage = (count / total_lines) * 100
        print(f"{source:<50} {count:>10,} ({percentage:>5.2f}%)")

    print(f"\nTotal unique knowledge sources: {len(knowledge_source_counter)}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze edges.jsonl file for predicate and knowledge source frequencies'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        default='edges.jsonl',
        help='Path to the edges.jsonl file (default: edges.jsonl)'
    )

    args = parser.parse_args()

    # Check if file exists
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: File '{args.input_file}' not found!")
        return 1

    analyze_edges(args.input_file)
    return 0


if __name__ == '__main__':
    exit(main())
