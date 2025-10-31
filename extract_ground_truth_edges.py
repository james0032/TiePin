#!/usr/bin/env python3
"""
Extract ground truth edges from edges.jsonl file based on knowledge source.

This script filters edges from a JSONL file by primary_knowledge_source
and outputs them in JSONL format, preserving all metadata.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_ground_truth_edges(
    edges_file: str,
    output_file: str,
    knowledge_source: str = "infores:drugmechdb"
):
    """Extract edges from JSONL file matching the knowledge source.

    Preserves all metadata in the original JSON format.

    Args:
        edges_file: Path to edges.jsonl file
        output_file: Path to output JSONL file
        knowledge_source: Filter by this primary_knowledge_source (default: infores:drugmechdb)
    """
    logger.info(f"Reading edges from {edges_file}")
    logger.info(f"Filtering by primary_knowledge_source: {knowledge_source}")

    matched_edges = []
    total_edges = 0
    skipped_lines = 0

    # Track unique predicates and subjects/objects
    predicates_seen = set()
    subjects_seen = set()
    objects_seen = set()

    with open(edges_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            total_edges += 1

            try:
                edge = json.loads(line)

                # Check if this edge matches the knowledge source filter
                if edge.get('primary_knowledge_source') == knowledge_source:
                    subject = edge.get('subject', '')
                    predicate = edge.get('predicate', '')
                    obj = edge.get('object', '')

                    # Skip if any required field is missing
                    if not subject or not predicate or not obj:
                        logger.warning(f"Line {line_num}: Missing required field (subject/predicate/object)")
                        skipped_lines += 1
                        continue

                    # Track statistics
                    subjects_seen.add(subject)
                    predicates_seen.add(predicate)
                    objects_seen.add(obj)

                    # Store the complete edge with all metadata
                    matched_edges.append(edge)

            except json.JSONDecodeError as e:
                logger.warning(f"Line {line_num}: Invalid JSON - {e}")
                skipped_lines += 1
                continue

            # Progress logging
            if total_edges % 100000 == 0:
                logger.info(f"Processed {total_edges:,} edges, found {len(matched_edges):,} matches...")

    logger.info(f"Finished reading {total_edges:,} total edges")
    logger.info(f"Found {len(matched_edges):,} edges matching '{knowledge_source}'")
    logger.info(f"Skipped {skipped_lines} invalid lines")

    # Write output in JSONL format (preserving all metadata)
    logger.info(f"Writing {len(matched_edges):,} edges to {output_file}")

    # Create output directory if needed
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for edge in matched_edges:
            # Write each edge as a JSON line
            f.write(json.dumps(edge) + '\n')

    logger.info(f"✓ Output written to: {output_file}")

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("Statistics:")
    logger.info("=" * 80)
    logger.info(f"Total edges in file: {total_edges:,}")
    logger.info(f"Matched edges: {len(matched_edges):,}")
    logger.info(f"Match rate: {len(matched_edges)/total_edges*100:.2f}%")
    logger.info(f"Unique subjects: {len(subjects_seen):,}")
    logger.info(f"Unique predicates: {len(predicates_seen):,}")
    logger.info(f"Unique objects: {len(objects_seen):,}")
    logger.info("=" * 80)

    # Show predicate distribution
    logger.info("\nPredicate distribution:")
    predicate_counts = {}
    for edge in matched_edges:
        pred = edge['predicate']
        predicate_counts[pred] = predicate_counts.get(pred, 0) + 1

    for pred, count in sorted(predicate_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {pred}: {count:,} edges ({count/len(matched_edges)*100:.1f}%)")

    # Show sample edges
    logger.info("\nSample edges (first 5):")
    for i, edge in enumerate(matched_edges[:5], 1):
        logger.info(f"  {i}. {edge['subject']} --[{edge['predicate']}]--> {edge['object']}")


def main():
    parser = argparse.ArgumentParser(
        description='Extract ground truth edges from edges.jsonl file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract DrugMechDB edges (JSONL format with all metadata)
  python extract_ground_truth_edges.py \\
      --edges edges.jsonl \\
      --output ground_truth/drugmechdb_edges.jsonl \\
      --source infores:drugmechdb

  # Extract from different source
  python extract_ground_truth_edges.py \\
      --edges edges.jsonl \\
      --output ground_truth/pharos_edges.jsonl \\
      --source infores:pharos
        """
    )

    parser.add_argument(
        '--edges', type=str, required=True,
        help='Path to edges.jsonl file'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Path to output JSONL file'
    )
    parser.add_argument(
        '--source', type=str, default='infores:drugmechdb',
        help='Filter by primary_knowledge_source (default: infores:drugmechdb)'
    )

    args = parser.parse_args()

    extract_ground_truth_edges(
        edges_file=args.edges,
        output_file=args.output,
        knowledge_source=args.source
    )


if __name__ == '__main__':
    main()
