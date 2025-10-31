#!/usr/bin/env python3
"""
Add a ground truth indicator column to TracIn CSV output.

Matches training edges (TrainHead, TrainRel_label, TrainTail) against
ground truth edges extracted from edges.jsonl.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Set, Tuple

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


def add_ground_truth_column(
    tracin_csv: str,
    ground_truth_jsonl: str,
    output_csv: str
) -> None:
    """Add ground truth indicator column to TracIn CSV.

    Args:
        tracin_csv: Path to TracIn CSV output
        ground_truth_jsonl: Path to ground truth edges JSONL
        output_csv: Path to output CSV with added column
    """
    # Load ground truth edges
    gt_edges = load_ground_truth_edges(ground_truth_jsonl)

    logger.info(f"Processing TracIn CSV: {tracin_csv}")

    # Process CSV
    matched_count = 0
    total_rows = 0

    with open(tracin_csv, 'r') as f_in, open(output_csv, 'w') as f_out:
        for line_num, line in enumerate(f_in, 1):
            line = line.rstrip('\n')

            # Handle header
            if line_num == 1:
                # Add new column to header
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
                # Write original line with 0
                f_out.write(f"{line},0\n")
                continue

            # Extract training edge components
            train_head = parts[6]  # TrainHead (CURIE)
            train_rel_label = parts[9]  # TrainRel_label (e.g., biolink:related_to)
            train_tail = parts[10]  # TrainTail (CURIE)

            # Check if this edge exists in ground truth
            edge_tuple = (train_head, train_rel_label, train_tail)
            is_ground_truth = 1 if edge_tuple in gt_edges else 0

            if is_ground_truth:
                matched_count += 1

            # Write row with new column
            f_out.write(f"{line},{is_ground_truth}\n")

    logger.info(f"Processed {total_rows} training edges")
    logger.info(f"Matched {matched_count} ground truth edges ({matched_count/total_rows*100:.2f}%)")
    logger.info(f"Output written to: {output_csv}")

    # Print some statistics
    if matched_count > 0:
        logger.info(f"\n✓ Found {matched_count} training edges that match ground truth")
    else:
        logger.warning(f"\n⚠ No matches found - check that edge formats match")


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

    # Create output directory if needed
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process files
    add_ground_truth_column(
        tracin_csv=args.tracin_csv,
        ground_truth_jsonl=args.ground_truth,
        output_csv=args.output
    )


if __name__ == '__main__':
    main()
