#!/usr/bin/env python3
"""
Create train_candidates.txt by removing test edges from rotorobo.txt

This script takes a full subgraph (rotorobo.txt) and a fixed test set (test.txt),
and removes all test edges to create train_candidates.txt for subsequent train/valid splitting.

This ensures the test set remains identical across multiple runs, enabling reproducible experiments.

Usage:
    python create_train_candidates.py \
        --subgraph /path/to/rotorobo.txt \
        --test /path/to/test.txt \
        --output /path/to/train_candidates.txt
"""

import argparse
import logging
from pathlib import Path
from typing import Set, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_triples(file_path: str) -> Set[Tuple[str, str, str]]:
    """Load triples from TSV file into a set.

    Args:
        file_path: Path to TSV file with triples (head, relation, tail)

    Returns:
        Set of (head, relation, tail) tuples
    """
    triples = set()

    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                logger.warning(f"Line {line_num}: Expected 3 columns, got {len(parts)}. Skipping: {line}")
                continue

            head, relation, tail = parts
            triples.add((head, relation, tail))

    logger.info(f"Loaded {len(triples):,} triples from {file_path}")
    return triples


def create_train_candidates(
    subgraph_path: str,
    test_path: str,
    output_path: str
):
    """Create train_candidates.txt by removing test edges from subgraph.

    Args:
        subgraph_path: Path to full subgraph file (rotorobo.txt)
        test_path: Path to fixed test set file (test.txt)
        output_path: Path to write train_candidates.txt
    """
    logger.info("=" * 80)
    logger.info("Creating train_candidates.txt from subgraph and fixed test set")
    logger.info("=" * 80)

    # Load subgraph and test triples
    logger.info(f"\nStep 1: Loading subgraph from {subgraph_path}")
    subgraph_triples = load_triples(subgraph_path)

    logger.info(f"\nStep 2: Loading test set from {test_path}")
    test_triples = load_triples(test_path)

    # Remove test triples from subgraph
    logger.info(f"\nStep 3: Removing test triples from subgraph")
    initial_count = len(subgraph_triples)

    # Check how many test triples are actually in the subgraph
    test_in_subgraph = test_triples & subgraph_triples
    test_not_in_subgraph = test_triples - subgraph_triples

    if test_not_in_subgraph:
        logger.warning(f"  Warning: {len(test_not_in_subgraph)} test triples not found in subgraph")
        logger.warning(f"  These triples may have been filtered out during subgraph creation")

    # Create train_candidates by removing test triples
    train_candidates = subgraph_triples - test_triples
    removed_count = initial_count - len(train_candidates)

    logger.info(f"  Subgraph triples: {initial_count:,}")
    logger.info(f"  Test triples: {len(test_triples):,}")
    logger.info(f"  Test triples in subgraph: {len(test_in_subgraph):,}")
    logger.info(f"  Removed from subgraph: {removed_count:,}")
    logger.info(f"  Train candidates: {len(train_candidates):,}")

    # Write train_candidates to file
    logger.info(f"\nStep 4: Writing train_candidates to {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for head, relation, tail in sorted(train_candidates):
            f.write(f"{head}\t{relation}\t{tail}\n")

    logger.info(f"  Wrote {len(train_candidates):,} triples to {output_path}")

    # Summary statistics
    logger.info("\n" + "=" * 80)
    logger.info("Summary Statistics")
    logger.info("=" * 80)
    logger.info(f"Input subgraph:        {initial_count:,} triples")
    logger.info(f"Fixed test set:        {len(test_triples):,} triples")
    logger.info(f"Test triples removed:  {removed_count:,} triples ({removed_count/initial_count*100:.2f}%)")
    logger.info(f"Output train_candidates: {len(train_candidates):,} triples ({len(train_candidates)/initial_count*100:.2f}%)")
    logger.info("=" * 80)

    logger.info("\n✓ Successfully created train_candidates.txt")
    logger.info(f"  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Create train_candidates.txt by removing test edges from subgraph',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Create train_candidates.txt with fixed test set
    python create_train_candidates.py \\
        --subgraph /workspace/data/robokop/CGGD_alltreat/rotorobo.txt \\
        --test /workspace/data/robokop/CGGD_alltreat/test.txt \\
        --output /workspace/data/robokop/CGGD_alltreat/train_candidates.txt

This script ensures reproducible experiments by using a fixed test set across multiple runs.
The test.txt file should be created once and reused for all subsequent experiments.
        """
    )

    parser.add_argument('--subgraph', required=True,
                        help='Path to full subgraph file (rotorobo.txt)')
    parser.add_argument('--test', required=True,
                        help='Path to fixed test set file (test.txt)')
    parser.add_argument('--output', required=True,
                        help='Path to write train_candidates.txt')

    args = parser.parse_args()

    # Validate input files exist
    if not Path(args.subgraph).exists():
        logger.error(f"Error: Subgraph file not found: {args.subgraph}")
        return 1

    if not Path(args.test).exists():
        logger.error(f"Error: Test file not found: {args.test}")
        return 1

    # Create train_candidates
    create_train_candidates(
        subgraph_path=args.subgraph,
        test_path=args.test,
        output_path=args.output
    )

    return 0


if __name__ == '__main__':
    exit(main())
