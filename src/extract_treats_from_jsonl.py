#!/usr/bin/env python3
"""
Extract biolink:treats edges from edges.jsonl file.

This script reads a JSONL file containing knowledge graph edges and extracts
only the treats relationships. It requires an edge_map.json to determine the
correct predicate index (e.g., predicate:28) for biolink:treats.

Input:
- edges.jsonl: JSONL file with edges (one JSON object per line)
- edge_map.json: JSON mapping of predicate details to predicate IDs

Output:
- treat_edges.txt: Tab-separated triples (subject\tpredicate:N\tobject)
  No header, using the correct predicate index from edge_map.json

Example output:
CHEBI:4846      predicate:28    MONDO:0005976
CHEBI:4508      predicate:28    MONDO:0005178
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Set, List, Tuple

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_edge_map(edge_map_path: str) -> dict:
    """Load edge map from JSON file.

    Args:
        edge_map_path: Path to edge_map.json file

    Returns:
        Dictionary mapping predicate details to predicate IDs
    """
    logger.info(f"Loading edge map from {edge_map_path}")
    with open(edge_map_path, 'r') as f:
        edge_map = json.load(f)
    logger.info(f"Loaded {len(edge_map)} predicate mappings")
    return edge_map


def find_treats_predicate(edge_map: dict) -> str:
    """Find the predicate ID for biolink:treats.

    Args:
        edge_map: Dictionary mapping predicate details to predicate IDs

    Returns:
        Predicate ID string (e.g., "predicate:28")

    Raises:
        ValueError: If no treats predicate is found
    """
    for predicate_detail, predicate_id in edge_map.items():
        try:
            pred_dict = json.loads(predicate_detail)
            if pred_dict.get("predicate") == "biolink:treats":
                logger.info(f"Found biolink:treats mapped to {predicate_id}")
                return predicate_id
        except json.JSONDecodeError:
            continue

    raise ValueError("No biolink:treats predicate found in edge_map.json")


def extract_treats_edges(
    jsonl_path: str,
    output_path: str,
    treats_predicate_id: str
) -> None:
    """Extract treats edges from JSONL file.

    Args:
        jsonl_path: Path to edges.jsonl file
        output_path: Path to output file
        treats_predicate_id: Predicate ID to use for treats edges (e.g., "predicate:28")
    """
    logger.info(f"Extracting biolink:treats edges from {jsonl_path}")

    treats_edges = []
    total_lines = 0
    treats_count = 0

    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            total_lines += 1
            line = line.strip()

            if not line:
                continue

            try:
                edge = json.loads(line)

                # Check if this is a treats edge
                predicate = edge.get('predicate', '')
                if predicate == 'biolink:treats':
                    subject = edge.get('subject', '')
                    obj = edge.get('object', '')

                    if subject and obj:
                        treats_edges.append((subject, treats_predicate_id, obj))
                        treats_count += 1
                    else:
                        logger.warning(f"Line {line_num}: Missing subject or object")

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                continue

            if line_num % 100000 == 0:
                logger.info(f"Processed {line_num} lines, found {treats_count} treats edges...")

    logger.info(f"Processed {total_lines} total lines")
    logger.info(f"Found {treats_count} biolink:treats edges")

    # Deduplicate
    unique_treats_edges = list(set(treats_edges))
    dup_count = len(treats_edges) - len(unique_treats_edges)
    if dup_count > 0:
        logger.info(f"Removed {dup_count} duplicate edges")
    logger.info(f"Writing {len(unique_treats_edges)} unique treats edges to {output_path}")

    # Write to output file
    with open(output_path, 'w') as f:
        for subject, predicate, obj in unique_treats_edges:
            f.write(f"{subject}\t{predicate}\t{obj}\n")

    logger.info(f"Successfully wrote treats edges to {output_path}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract biolink:treats edges from edges.jsonl file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script extracts only the biolink:treats edges from a JSONL file and
outputs them in tab-separated format using the correct predicate index
from edge_map.json.

Examples:
  # Basic usage
  python extract_treats_from_jsonl.py \\
    --input edges.jsonl \\
    --edge-map edge_map.json \\
    --output treat_edges.txt

  # With custom log level
  python extract_treats_from_jsonl.py \\
    --input edges.jsonl \\
    --edge-map edge_map.json \\
    --output treat_edges.txt \\
    --log-level DEBUG
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input edges.jsonl file'
    )

    parser.add_argument(
        '--edge-map',
        type=str,
        required=True,
        help='Path to edge_map.json file'
    )

    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output file (treat_edges.txt)'
    )

    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Logging level (default: INFO)'
    )

    return parser.parse_args()


def main():
    """Main entry point for the script."""
    args = parse_arguments()

    # Setup logging
    setup_logging(getattr(logging, args.log_level))

    logger.info("Starting treats edge extraction")

    try:
        # Validate input files
        if not Path(args.input).exists():
            logger.error(f"Input JSONL file not found: {args.input}")
            return 1

        if not Path(args.edge_map).exists():
            logger.error(f"Edge map file not found: {args.edge_map}")
            return 1

        # Create output directory if needed
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Load edge map and find treats predicate
        edge_map = load_edge_map(args.edge_map)
        treats_predicate_id = find_treats_predicate(edge_map)

        # Extract treats edges
        extract_treats_edges(args.input, args.output, treats_predicate_id)

        logger.info("=" * 80)
        logger.info("Extraction complete!")
        logger.info(f"  Output file: {args.output}")
        logger.info(f"  Treats predicate used: {treats_predicate_id}")
        logger.info("=" * 80)

        return 0

    except ValueError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
