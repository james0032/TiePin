#!/usr/bin/env python3
"""
Add a ground truth indicator column to TracIn CSV output.

Matches training edges (TrainHead, TrainRel_label, TrainTail) against
ground truth edges extracted from edges.jsonl.

Also adds an "In_path" column to indicate if the training edge is part of
the mechanistic path (connects test entities or intermediate nodes).
"""

import argparse
import ast
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Set, Tuple, List, Dict

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_ground_truth_edges(jsonl_file: str) -> Set[Tuple[str, str, str]]:
    """Load ground truth edges from JSONL file.

    Args:
        jsonl_file: Path to JSONL file with ground truth edges

    Returns:
        Set of (subject, predicate, object) tuples
    """
    edges = set()

    logger.info(f"Loading ground truth edges from {jsonl_file}")

    with open(jsonl_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                edge = json.loads(line)
                subject = edge.get('subject', '')
                predicate = edge.get('predicate', '')
                obj = edge.get('object', '')

                if subject and predicate and obj:
                    edges.add((subject, predicate, obj))
                else:
                    logger.warning(f"Line {line_num}: Missing subject/predicate/object")

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue

    logger.info(f"Loaded {len(edges)} unique ground truth edges")
    return edges


def load_mechanistic_paths(csv_file: str) -> Dict[Tuple[str, str], List[str]]:
    """Load mechanistic paths from CSV file.

    Args:
        csv_file: Path to CSV file with Drug, Disease, [Intermediate Nodes] columns

    Returns:
        Dictionary mapping (drug, disease) tuples to lists of intermediate nodes
    """
    paths = {}

    logger.info(f"Loading mechanistic paths from {csv_file}")

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row_num, row in enumerate(reader, 1):
            drug = row.get('Drug', '').strip()
            disease = row.get('Disease', '').strip()
            intermediate_nodes_str = row.get('[Intermediate Nodes]', '').strip()

            if not drug or not disease:
                logger.warning(f"Row {row_num}: Missing Drug or Disease")
                continue

            # Parse intermediate nodes list
            try:
                # Handle empty list
                if intermediate_nodes_str == '[]':
                    intermediate_nodes = []
                else:
                    # Parse as Python literal (list of strings)
                    intermediate_nodes = ast.literal_eval(intermediate_nodes_str)
                    if not isinstance(intermediate_nodes, list):
                        intermediate_nodes = []

                paths[(drug, disease)] = intermediate_nodes

            except (ValueError, SyntaxError) as e:
                logger.warning(f"Row {row_num}: Failed to parse intermediate nodes '{intermediate_nodes_str}': {e}")
                paths[(drug, disease)] = []

    logger.info(f"Loaded {len(paths)} mechanistic paths")
    return paths


def is_in_path(
    train_head: str,
    train_tail: str,
    test_head: str,
    test_tail: str,
    intermediate_nodes: List[str]
) -> bool:
    """Check if a training edge is part of the mechanistic path.

    An edge is in the path if it connects:
    - test_head to an intermediate node
    - test_tail to an intermediate node
    - an intermediate node to another intermediate node
    - Since the graph is undirected, also check reverse connections

    Args:
        train_head: Training edge head entity
        train_tail: Training edge tail entity
        test_head: Test triple head entity (drug)
        test_tail: Test triple tail entity (disease)
        intermediate_nodes: List of intermediate node entities

    Returns:
        True if edge is part of the mechanistic path
    """
    # Create set of all path nodes (test entities + intermediate nodes)
    path_nodes = {test_head, test_tail}
    path_nodes.update(intermediate_nodes)

    # Check if both endpoints of the training edge are in the path
    # This covers all cases:
    # - [test_head, intermediate_node]
    # - [test_tail, intermediate_node]
    # - [intermediate_node1, intermediate_node2]
    # - All reversed versions (since graph is undirected)
    return train_head in path_nodes and train_tail in path_nodes


def add_ground_truth_column(
    tracin_csv: str,
    ground_truth_jsonl: str,
    output_csv: str,
    mechanistic_paths_csv: str = None
) -> None:
    """Add ground truth indicator column to TracIn CSV.

    Args:
        tracin_csv: Path to TracIn CSV output
        ground_truth_jsonl: Path to ground truth edges JSONL
        output_csv: Path to output CSV with added column
        mechanistic_paths_csv: Optional path to mechanistic paths CSV
    """
    # Load ground truth edges
    gt_edges = load_ground_truth_edges(ground_truth_jsonl)

    # Load mechanistic paths if provided
    mechanistic_paths = {}
    if mechanistic_paths_csv:
        mechanistic_paths = load_mechanistic_paths(mechanistic_paths_csv)

    logger.info(f"Processing TracIn CSV: {tracin_csv}")

    # Process CSV
    matched_count = 0
    in_path_count = 0
    total_rows = 0

    with open(tracin_csv, 'r') as f_in, open(output_csv, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.rstrip('\n')

            # Handle header
            if line_num == 1:
                # Add new columns to header
                if mechanistic_paths_csv:
                    f_out.write(f"{line},IsGroundTruth,In_path\n")
                else:
                    f_out.write(f"{line},IsGroundTruth\n")
                continue

            total_rows += 1

            # Split CSV line
            parts = line.split(',')

            # Expected columns (14 fields):
            # 0: TestHead, 1: TestHead_label, 2: TestRel, 3: TestRel_label,
            # 4: TestTail, 5: TestTail_label, 6: TrainHead, 7: TrainHead_label,
            # 8: TrainRel, 9: TrainRel_label, 10: TrainTail, 11: TrainTail_label,
            # 12: TracInScore, 13: SelfInfluence

            if len(parts) < 14:
                logger.warning(f"Line {line_num}: Expected at least 14 fields, got {len(parts)}")
                # Write original line with 0s
                if mechanistic_paths_csv:
                    f_out.write(f"{line},0,0\n")
                else:
                    f_out.write(f"{line},0\n")
                continue

            # Extract test triple components
            test_head = parts[0]  # TestHead (CURIE)
            test_tail = parts[4]  # TestTail (CURIE)

            # Extract training edge components
            train_head = parts[6]  # TrainHead (CURIE)
            train_rel_label = parts[9]  # TrainRel_label (e.g., biolink:related_to)
            train_tail = parts[10]  # TrainTail (CURIE)

            # Check if this edge exists in ground truth
            edge_tuple = (train_head, train_rel_label, train_tail)
            is_ground_truth = 1 if edge_tuple in gt_edges else 0

            if is_ground_truth:
                matched_count += 1

            # Check if edge is in mechanistic path
            in_path = 0
            if mechanistic_paths_csv:
                # Look up intermediate nodes for this test triple
                intermediate_nodes = mechanistic_paths.get((test_head, test_tail), [])

                # Check if training edge is part of the path
                if is_in_path(train_head, train_tail, test_head, test_tail, intermediate_nodes):
                    in_path = 1
                    in_path_count += 1

            # Write row with new columns
            if mechanistic_paths_csv:
                f_out.write(f"{line},{is_ground_truth},{in_path}\n")
            else:
                f_out.write(f"{line},{is_ground_truth}\n")

    logger.info(f"Processed {total_rows} training edges")
    logger.info(f"Matched {matched_count} ground truth edges ({matched_count/total_rows*100:.2f}%)")

    if mechanistic_paths_csv:
        logger.info(f"Found {in_path_count} edges in mechanistic paths ({in_path_count/total_rows*100:.2f}%)")

    logger.info(f"Output written to: {output_csv}")

    # Print some statistics
    if matched_count > 0:
        logger.info(f"\n✓ Found {matched_count} training edges that match ground truth")
    else:
        logger.warning(f"\n⚠ No matches found - check that edge formats match")

    if mechanistic_paths_csv and in_path_count > 0:
        logger.info(f"✓ Found {in_path_count} training edges in mechanistic paths")


def main():
    parser = argparse.ArgumentParser(
        description='Add ground truth indicator column to TracIn CSV output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add ground truth column to TracIn output
  python add_ground_truth_column.py \\
      --tracin-csv dmdb_results/test_scores_tracin.csv \\
      --ground-truth ground_truth/drugmechdb_edges.jsonl \\
      --output dmdb_results/test_scores_tracin_with_gt.csv

  # Add both ground truth and mechanistic path columns
  python add_ground_truth_column.py \\
      --tracin-csv dmdb_results/triple_000_CHEBI_17154_MONDO_0019975_tracin.csv \\
      --ground-truth ground_truth/drugmechdb_edges.jsonl \\
      --mechanistic-paths dedup_treats_mechanistic_paths.txt \\
      --output dmdb_results/triple_000_with_gt_and_path.csv

  # Process multiple files
  python add_ground_truth_column.py \\
      --tracin-csv results/batch_1_tracin.csv \\
      --ground-truth ground_truth/drugmechdb_edges.jsonl \\
      --output results/batch_1_tracin_with_gt.csv
        """
    )

    parser.add_argument(
        '--tracin-csv', type=str, required=True,
        help='Path to TracIn CSV output file'
    )
    parser.add_argument(
        '--ground-truth', type=str, required=True,
        help='Path to ground truth edges JSONL file'
    )
    parser.add_argument(
        '--mechanistic-paths', type=str,
        help='Optional path to mechanistic paths CSV file (Drug, Disease, [Intermediate Nodes])'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Path to output CSV file with ground truth column'
    )

    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.tracin_csv).exists():
        logger.error(f"TracIn CSV file not found: {args.tracin_csv}")
        sys.exit(1)

    if not Path(args.ground_truth).exists():
        logger.error(f"Ground truth JSONL file not found: {args.ground_truth}")
        sys.exit(1)

    if args.mechanistic_paths and not Path(args.mechanistic_paths).exists():
        logger.error(f"Mechanistic paths CSV file not found: {args.mechanistic_paths}")
        sys.exit(1)

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process files
    add_ground_truth_column(
        tracin_csv=args.tracin_csv,
        ground_truth_jsonl=args.ground_truth,
        output_csv=args.output,
        mechanistic_paths_csv=args.mechanistic_paths
    )


if __name__ == '__main__':
    main()
