#!/usr/bin/env python3
"""
Extract test edges from filtered DrugMechDB treats edges.

This script uses a filtered TSV file (from add_pair_exists_column.py) that contains
treats edges verified to exist in both the knowledge graph and DrugMechDB.
It samples a user-defined percentage of these edges for the test set, then removes
them from the full rotorobo.txt to create train_candidates.txt.

Input:
- rotorobo.txt: Tab-separated triples (subject\tpredicate\tobject)
- edge_map.json: JSON mapping of predicate details to predicate IDs
- filtered_tsv: TSV file with verified treats edges (from add_pair_exists_column.py)

Output:
- test.txt: Sampled test edges from filtered_tsv
- train_candidates.txt: rotorobo.txt with test edges removed
- test_statistics.json: Statistics about the test set
"""

import argparse
import json
import logging
import os
import random
from typing import Dict, List, Set, Tuple

# Configure logger
logger = logging.getLogger(__name__)


def setup_logging(log_level):
    """Configure logging with the specified level."""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_edge_map(edge_map_path: str) -> Dict[str, str]:
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


def find_treats_predicates(edge_map: Dict[str, str]) -> Set[str]:
    """Find all predicate IDs that correspond to biolink:treats.

    Args:
        edge_map: Dictionary mapping predicate details to predicate IDs

    Returns:
        Set of predicate IDs that represent treats relationships
    """
    treats_predicates = set()

    for predicate_detail, predicate_id in edge_map.items():
        # Parse the JSON string to check for treats predicate
        try:
            pred_dict = json.loads(predicate_detail)
            if pred_dict.get("predicate") == "biolink:treats":
                treats_predicates.add(predicate_id)
        except json.JSONDecodeError:
            logger.warning(f"Could not parse predicate detail: {predicate_detail}")
            continue

    logger.info(f"Found {len(treats_predicates)} treats predicate IDs: {treats_predicates}")
    return treats_predicates


def load_filtered_treats_edges(filtered_tsv_path: str) -> List[Tuple[str, str, str]]:
    """Load treats edges from filtered TSV file.

    Args:
        filtered_tsv_path: Path to filtered TSV file (from add_pair_exists_column.py)

    Returns:
        List of (subject, predicate, object) tuples
    """
    logger.info(f"Loading filtered treats edges from {filtered_tsv_path}")
    edges = []

    with open(filtered_tsv_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip header
            if line.startswith('Drug') or not line:
                continue

            parts = line.split('\t')
            if len(parts) < 3:
                logger.warning(f"Line {line_num}: Invalid format (expected at least 3 columns, got {len(parts)})")
                continue

            subject = parts[0].strip()
            predicate = parts[1].strip()
            obj = parts[2].strip()

            edges.append((subject, predicate, obj))

    logger.info(f"Loaded {len(edges)} filtered treats edges")
    return edges


def load_all_edges(rotorobo_path: str) -> List[Tuple[str, str, str]]:
    """Load all edges from rotorobo.txt.

    Args:
        rotorobo_path: Path to rotorobo.txt file

    Returns:
        List of (subject, predicate, object) tuples
    """
    logger.info(f"Loading all edges from {rotorobo_path}")
    edges = []

    with open(rotorobo_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = line.split('\t')
            if len(parts) != 3:
                logger.warning(f"Line {line_num}: Invalid format (expected 3 columns, got {len(parts)})")
                continue

            subject, predicate, obj = parts
            edges.append((subject, predicate, obj))

            if (line_num) % 100000 == 0:
                logger.debug(f"Processed {line_num} edges...")

    logger.info(f"Loaded {len(edges)} total edges from rotorobo.txt")
    return edges


def sample_test_edges(
    filtered_edges: List[Tuple[str, str, str]],
    test_percentage: float,
    seed: int = 42
) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    """Sample test edges from filtered treats edges.

    Args:
        filtered_edges: List of filtered treats edges
        test_percentage: Percentage of edges to use for test (e.g., 0.10 for 10%)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (test_edges, remaining_filtered_edges)
    """
    random.seed(seed)
    logger.info("=" * 80)
    logger.info(f"Sampling test edges from {len(filtered_edges)} filtered treats edges")
    logger.info(f"Test percentage: {test_percentage * 100}%")
    logger.info("=" * 80)

    # Calculate target test size
    target_test_size = int(len(filtered_edges) * test_percentage)

    # Shuffle and sample
    shuffled_edges = filtered_edges.copy()
    random.shuffle(shuffled_edges)

    test_edges = shuffled_edges[:target_test_size]
    remaining_filtered_edges = shuffled_edges[target_test_size:]

    logger.info(f"Sampled {len(test_edges)} test edges ({len(test_edges)/len(filtered_edges)*100:.2f}%)")
    logger.info(f"Remaining filtered edges: {len(remaining_filtered_edges)}")

    return test_edges, remaining_filtered_edges


def write_edges(edges: List[Tuple[str, str, str]], output_path: str):
    """Write edges to file.

    Args:
        edges: List of (subject, predicate, object) tuples
        output_path: Path to output file
    """
    logger.info(f"Writing {len(edges)} edges to {output_path}")
    with open(output_path, 'w') as f:
        for subject, predicate, obj in edges:
            f.write(f"{subject}\t{predicate}\t{obj}\n")


def save_statistics(stats: Dict, output_path: str):
    """Save statistics to JSON file.

    Args:
        stats: Dictionary of statistics
        output_path: Path to output JSON file
    """
    logger.info(f"Saving statistics to {output_path}")
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Extract test edges from filtered DrugMechDB treats edges.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates a test set from DrugMechDB-verified treats edges and removes
them from the full knowledge graph to create training candidates.

The filtered_tsv file should be the output from add_pair_exists_column.py,
containing only treats edges that exist in both the knowledge graph and DrugMechDB.

Examples:
  # Basic usage with 10% test split
  python make_test_with_drugmechdb_treat.py \\
    --input-dir robokop/CGGD_alltreat \\
    --filtered-tsv drugmechdb_treats_filtered.txt \\
    --test-pct 0.10

  # Custom seed for reproducibility
  python make_test_with_drugmechdb_treat.py \\
    --input-dir robokop/CGGD_alltreat \\
    --filtered-tsv drugmechdb_treats_filtered.txt \\
    --test-pct 0.15 \\
    --seed 123
        """
    )

    parser.add_argument(
        '--input-dir',
        type=str,
        required=True,
        help='Input directory containing rotorobo.txt and edge_map.json'
    )

    parser.add_argument(
        '--filtered-tsv',
        type=str,
        required=True,
        help='Path to filtered TSV file (output from add_pair_exists_column.py)'
    )

    parser.add_argument(
        '--triples-file',
        type=str,
        default='rotorobo.txt',
        help='Name of triples file (default: rotorobo.txt)'
    )

    parser.add_argument(
        '--edge-map-file',
        type=str,
        default='edge_map.json',
        help='Name of edge map file (default: edge_map.json)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as input-dir)'
    )

    parser.add_argument(
        '--test-pct',
        type=float,
        required=True,
        help='Percentage of filtered treats edges to use for test set (e.g., 0.10 for 10%%)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
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

    logger.info("Starting test edge extraction with DrugMechDB filtered treats")
    logger.info(f"Working directory: {os.getcwd()}")

    try:
        # Determine paths
        input_dir = args.input_dir
        output_dir = args.output_dir if args.output_dir else input_dir

        triples_path = os.path.join(input_dir, args.triples_file)
        edge_map_path = os.path.join(input_dir, args.edge_map_file)

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        logger.info("=" * 80)
        logger.info("Configuration:")
        logger.info(f"  Input directory: {input_dir}")
        logger.info(f"  Triples file: {triples_path}")
        logger.info(f"  Edge map file: {edge_map_path}")
        logger.info(f"  Filtered TSV: {args.filtered_tsv}")
        logger.info(f"  Output directory: {output_dir}")
        logger.info(f"  Test percentage: {args.test_pct * 100}%")
        logger.info(f"  Random seed: {args.seed}")
        logger.info("=" * 80)

        # Validate input files
        if not os.path.exists(triples_path):
            logger.error(f"Triples file not found: {triples_path}")
            return 1

        if not os.path.exists(edge_map_path):
            logger.error(f"Edge map file not found: {edge_map_path}")
            return 1

        if not os.path.exists(args.filtered_tsv):
            logger.error(f"Filtered TSV file not found: {args.filtered_tsv}")
            return 1

        # Validate test percentage
        if args.test_pct <= 0 or args.test_pct >= 1:
            logger.error(f"Test percentage must be between 0 and 1 (got {args.test_pct})")
            return 1

        # Load edge map and find treats predicates
        edge_map = load_edge_map(edge_map_path)
        treats_predicates = find_treats_predicates(edge_map)

        if not treats_predicates:
            logger.error("No treats predicates found in edge map")
            return 1

        # Load filtered treats edges
        filtered_edges = load_filtered_treats_edges(args.filtered_tsv)

        if not filtered_edges:
            logger.error("No edges found in filtered TSV file")
            return 1

        # Deduplicate filtered edges
        logger.info(f"Deduplicating filtered edges (original count: {len(filtered_edges)})")
        unique_filtered_edges = list(set(filtered_edges))
        dup_count = len(filtered_edges) - len(unique_filtered_edges)
        if dup_count > 0:
            logger.info(f"Removed {dup_count} duplicate edges from filtered TSV")
        logger.info(f"Unique filtered edges: {len(unique_filtered_edges)}")

        # Sample test edges
        test_edges, remaining_filtered_edges = sample_test_edges(
            unique_filtered_edges,
            args.test_pct,
            args.seed
        )

        # Load all edges from rotorobo.txt
        all_edges = load_all_edges(triples_path)

        # Remove test edges from all edges to create train candidates
        logger.info("Creating train candidates by removing test edges from rotorobo.txt")
        test_edge_set = set(test_edges)
        train_candidates = [edge for edge in all_edges if edge not in test_edge_set]

        removed_count = len(all_edges) - len(train_candidates)
        logger.info(f"Removed {removed_count} test edges from rotorobo.txt")
        logger.info(f"Train candidates: {len(train_candidates)} edges")

        if removed_count != len(test_edges):
            logger.warning(f"Mismatch: Expected to remove {len(test_edges)} edges but removed {removed_count}")
            logger.warning(f"This could mean {len(test_edges) - removed_count} test edges were not found in rotorobo.txt")

        # Write output files
        test_output = os.path.join(output_dir, 'test.txt')
        train_candidates_output = os.path.join(output_dir, 'train_candidates.txt')

        write_edges(test_edges, test_output)
        write_edges(train_candidates, train_candidates_output)

        # Save statistics
        stats = {
            'total_edges_in_rotorobo': len(all_edges),
            'filtered_treats_edges': len(filtered_edges),
            'filtered_treats_edges_unique': len(unique_filtered_edges),
            'filtered_treats_duplicates': dup_count,
            'test_edges': len(test_edges),
            'test_percentage': args.test_pct * 100,
            'actual_test_percentage': len(test_edges) / len(unique_filtered_edges) * 100,
            'test_edges_removed_from_rotorobo': removed_count,
            'train_candidate_edges': len(train_candidates),
            'remaining_filtered_edges': len(remaining_filtered_edges),
            'config': {
                'filtered_tsv': args.filtered_tsv,
                'test_pct': args.test_pct,
                'seed': args.seed
            }
        }

        stats_output = os.path.join(output_dir, 'test_statistics.json')
        save_statistics(stats, stats_output)

        logger.info("\n" + "=" * 80)
        logger.info("Test edge extraction complete!")
        logger.info(f"  Test edges: {test_output} ({len(test_edges)} edges)")
        logger.info(f"  Train candidates: {train_candidates_output} ({len(train_candidates)} edges)")
        logger.info(f"  Statistics: {stats_output}")
        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.exception(f"Error occurred during processing: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
