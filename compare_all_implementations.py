"""
Compare all three filtering implementations: PyG, NetworkX, and igraph.

This script runs all three implementations with the same parameters and compares:
- Number of filtered triples
- Execution time
- Memory usage
- Exact triple matches/differences

Use this to validate implementations and choose the best one for your use case.
"""

import logging
import argparse
from typing import Dict, List, Tuple, Set
from pathlib import Path
import numpy as np
import time
import sys

logger = logging.getLogger(__name__)


def load_triples_from_file(filepath: str) -> List[List[str]]:
    """Load triples from TSV file."""
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append(parts[:3])
    return triples


def create_entity_mappings(train_triples: List, test_triples: List) -> Tuple[Dict, Dict, Dict]:
    """Create entity and relation mappings."""
    entity_to_idx = {}
    idx_to_entity = {}
    relation_to_idx = {}

    current_entity_idx = 0
    current_relation_idx = 0

    for triple in list(train_triples) + list(test_triples):
        h, r, t = triple[0], triple[1], triple[2]

        if h not in entity_to_idx:
            entity_to_idx[h] = current_entity_idx
            idx_to_entity[current_entity_idx] = h
            current_entity_idx += 1

        if t not in entity_to_idx:
            entity_to_idx[t] = current_entity_idx
            idx_to_entity[current_entity_idx] = t
            current_entity_idx += 1

        if r not in relation_to_idx:
            relation_to_idx[r] = current_relation_idx
            current_relation_idx += 1

    return entity_to_idx, idx_to_entity, relation_to_idx


def run_pyg_filter(
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
    n_hops: int,
    min_degree: int,
    preserve_test_edges: bool,
    strict_hop_constraint: bool,
    path_filtering: bool,
    max_total_path_length: int
) -> Tuple[np.ndarray, float]:
    """Run PyTorch Geometric filter."""
    try:
        from filter_training_by_proximity_pyg import ProximityFilterPyG

        logger.info("\n" + "="*60)
        logger.info("Running PyTorch Geometric (PyG) Filter")
        logger.info("="*60)

        start_time = time.time()

        filter_obj = ProximityFilterPyG(train_numeric)
        filtered = filter_obj.filter_for_multiple_test_triples(
            test_triples=test_numeric,
            n_hops=n_hops,
            min_degree=min_degree,
            preserve_test_entity_edges=preserve_test_edges,
            strict_hop_constraint=strict_hop_constraint,
            path_filtering=path_filtering,
            max_total_path_length=max_total_path_length
        )

        elapsed = time.time() - start_time
        logger.info(f"PyG completed in {elapsed:.2f}s")

        return filtered, elapsed

    except ImportError as e:
        logger.error(f"Failed to import PyG implementation: {e}")
        return None, 0.0
    except Exception as e:
        logger.error(f"PyG filter failed: {e}")
        return None, 0.0


def run_networkx_filter(
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
    n_hops: int,
    min_degree: int,
    preserve_test_edges: bool,
    strict_hop_constraint: bool,
    path_filtering: bool,
    max_total_path_length: int
) -> Tuple[np.ndarray, float]:
    """Run NetworkX filter."""
    try:
        from filter_training_networkx import NetworkXProximityFilter

        logger.info("\n" + "="*60)
        logger.info("Running NetworkX Filter")
        logger.info("="*60)

        start_time = time.time()

        filter_obj = NetworkXProximityFilter(train_numeric)
        filtered = filter_obj.filter_for_multiple_test_triples(
            test_triples=test_numeric,
            n_hops=n_hops,
            min_degree=min_degree,
            preserve_test_entity_edges=preserve_test_edges,
            strict_hop_constraint=strict_hop_constraint,
            path_filtering=path_filtering,
            max_total_path_length=max_total_path_length
        )

        elapsed = time.time() - start_time
        logger.info(f"NetworkX completed in {elapsed:.2f}s")

        return filtered, elapsed

    except ImportError as e:
        logger.error(f"Failed to import NetworkX implementation: {e}")
        return None, 0.0
    except Exception as e:
        logger.error(f"NetworkX filter failed: {e}")
        return None, 0.0


def run_igraph_filter(
    train_numeric: np.ndarray,
    test_numeric: np.ndarray,
    n_hops: int,
    min_degree: int,
    preserve_test_edges: bool,
    strict_hop_constraint: bool,
    path_filtering: bool,
    max_total_path_length: int
) -> Tuple[np.ndarray, float]:
    """Run igraph filter."""
    try:
        from filter_training_igraph import IGraphProximityFilter

        logger.info("\n" + "="*60)
        logger.info("Running igraph Filter")
        logger.info("="*60)

        start_time = time.time()

        filter_obj = IGraphProximityFilter(train_numeric)
        filtered, _ = filter_obj.filter_for_test_triples(
            test_triples=test_numeric,
            n_hops=n_hops,
            min_degree=min_degree,
            preserve_test_entity_edges=preserve_test_edges,
            path_filtering=path_filtering,
            max_total_path_length=max_total_path_length
        )

        elapsed = time.time() - start_time
        logger.info(f"igraph completed in {elapsed:.2f}s")

        return filtered, elapsed

    except ImportError as e:
        logger.error(f"Failed to import igraph implementation: {e}")
        logger.error("Install with: pip install igraph")
        return None, 0.0
    except Exception as e:
        logger.error(f"igraph filter failed: {e}")
        return None, 0.0


def compare_results(
    results: Dict[str, np.ndarray],
    idx_to_entity: Dict,
    idx_to_relation: Dict
) -> Dict:
    """Compare results from different implementations."""

    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)

    # Convert to sets of tuples for comparison
    result_sets = {}
    for name, triples in results.items():
        if triples is not None:
            result_sets[name] = set(tuple(t) for t in triples)

    # Print triple counts
    logger.info("\nTriple Counts:")
    for name, triple_set in result_sets.items():
        logger.info(f"  {name:20s}: {len(triple_set):,} triples")

    # Compare pairwise
    implementations = list(result_sets.keys())
    if len(implementations) < 2:
        logger.warning("Need at least 2 implementations to compare!")
        return {}

    logger.info("\nPairwise Comparisons:")
    comparison_stats = {}

    for i in range(len(implementations)):
        for j in range(i + 1, len(implementations)):
            impl1 = implementations[i]
            impl2 = implementations[j]

            set1 = result_sets[impl1]
            set2 = result_sets[impl2]

            intersection = set1 & set2
            only_in_1 = set1 - set2
            only_in_2 = set2 - set1

            match_pct = (len(intersection) / max(len(set1), len(set2)) * 100) if max(len(set1), len(set2)) > 0 else 0

            logger.info(f"\n  {impl1} vs {impl2}:")
            logger.info(f"    In both:          {len(intersection):,}")
            logger.info(f"    Only in {impl1:10s}: {len(only_in_1):,}")
            logger.info(f"    Only in {impl2:10s}: {len(only_in_2):,}")
            logger.info(f"    Match percentage: {match_pct:.2f}%")

            comparison_stats[f"{impl1}_vs_{impl2}"] = {
                'intersection': len(intersection),
                'only_in_1': len(only_in_1),
                'only_in_2': len(only_in_2),
                'match_percentage': match_pct
            }

            # Show sample differences if there are any
            if only_in_1 and len(only_in_1) <= 10:
                logger.info(f"\n    Sample triples only in {impl1}:")
                for h, r, t in list(only_in_1)[:10]:
                    h_name = idx_to_entity.get(h, h)
                    r_name = idx_to_relation.get(r, r)
                    t_name = idx_to_entity.get(t, t)
                    logger.info(f"      ({h_name}, {r_name}, {t_name})")

            if only_in_2 and len(only_in_2) <= 10:
                logger.info(f"\n    Sample triples only in {impl2}:")
                for h, r, t in list(only_in_2)[:10]:
                    h_name = idx_to_entity.get(h, h)
                    r_name = idx_to_relation.get(r, r)
                    t_name = idx_to_entity.get(t, t)
                    logger.info(f"      ({h_name}, {r_name}, {t_name})")

    return comparison_stats


def main():
    parser = argparse.ArgumentParser(
        description='Compare PyG, NetworkX, and igraph filtering implementations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python compare_all_implementations.py \\
      --train train.txt \\
      --test test.txt \\
      --n-hops 2 \\
      --min-degree 2 \\
      --path-filtering \\
      --implementations pyg networkx igraph

This will run all specified implementations and compare results.
        """
    )

    parser.add_argument('--train', type=str, required=True,
                       help='Path to training triples file')
    parser.add_argument('--test', type=str, required=True,
                       help='Path to test triples file')
    parser.add_argument('--n-hops', type=int, default=2,
                       help='Number of hops from test triples (default: 2)')
    parser.add_argument('--min-degree', type=int, default=2,
                       help='Minimum degree threshold (default: 2)')
    parser.add_argument('--preserve-test-edges', action='store_true', default=True,
                       help='Preserve edges with test entities (default: True)')
    parser.add_argument('--strict-hop-constraint', action='store_true',
                       help='Enforce strict n-hop constraint')
    parser.add_argument('--path-filtering', action='store_true',
                       help='Only keep edges on paths between drug and disease')
    parser.add_argument('--max-total-path-length', type=int, default=None,
                       help='Maximum total path length (drug_dist + disease_dist)')
    parser.add_argument('--implementations', nargs='+',
                       choices=['pyg', 'networkx', 'igraph', 'all'],
                       default=['all'],
                       help='Which implementations to run (default: all)')
    parser.add_argument('--save-outputs', action='store_true',
                       help='Save outputs from each implementation')
    parser.add_argument('--output-dir', type=str, default='comparison_outputs',
                       help='Directory to save outputs (default: comparison_outputs)')

    args = parser.parse_args()

    # Determine which implementations to run
    if 'all' in args.implementations:
        implementations_to_run = ['pyg', 'networkx', 'igraph']
    else:
        implementations_to_run = args.implementations

    logger.info("="*60)
    logger.info("FILTERING IMPLEMENTATION COMPARISON")
    logger.info("="*60)
    logger.info(f"Training file: {args.train}")
    logger.info(f"Test file: {args.test}")
    logger.info(f"Parameters:")
    logger.info(f"  n_hops: {args.n_hops}")
    logger.info(f"  min_degree: {args.min_degree}")
    logger.info(f"  preserve_test_edges: {args.preserve_test_edges}")
    logger.info(f"  strict_hop_constraint: {args.strict_hop_constraint}")
    logger.info(f"  path_filtering: {args.path_filtering}")
    logger.info(f"  max_total_path_length: {args.max_total_path_length}")
    logger.info(f"Implementations to run: {', '.join(implementations_to_run)}")

    # Load data
    logger.info("\n" + "="*60)
    logger.info("Loading Data")
    logger.info("="*60)

    train_triples = load_triples_from_file(args.train)
    test_triples = load_triples_from_file(args.test)

    logger.info(f"Loaded {len(train_triples)} training triples")
    logger.info(f"Loaded {len(test_triples)} test triples")

    # Create mappings
    entity_to_idx, idx_to_entity, relation_to_idx = create_entity_mappings(
        train_triples, test_triples
    )
    idx_to_relation = {v: k for k, v in relation_to_idx.items()}

    logger.info(f"Entities: {len(entity_to_idx)}")
    logger.info(f"Relations: {len(relation_to_idx)}")

    # Convert to numeric
    train_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in train_triples
    ])

    test_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in test_triples
    ])

    # Run implementations
    results = {}
    timings = {}

    if 'pyg' in implementations_to_run:
        filtered, elapsed = run_pyg_filter(
            train_numeric, test_numeric,
            args.n_hops, args.min_degree, args.preserve_test_edges,
            args.strict_hop_constraint, args.path_filtering,
            args.max_total_path_length
        )
        if filtered is not None:
            results['PyG'] = filtered
            timings['PyG'] = elapsed

    if 'networkx' in implementations_to_run:
        filtered, elapsed = run_networkx_filter(
            train_numeric, test_numeric,
            args.n_hops, args.min_degree, args.preserve_test_edges,
            args.strict_hop_constraint, args.path_filtering,
            args.max_total_path_length
        )
        if filtered is not None:
            results['NetworkX'] = filtered
            timings['NetworkX'] = elapsed

    if 'igraph' in implementations_to_run:
        filtered, elapsed = run_igraph_filter(
            train_numeric, test_numeric,
            args.n_hops, args.min_degree, args.preserve_test_edges,
            args.strict_hop_constraint, args.path_filtering,
            args.max_total_path_length
        )
        if filtered is not None:
            results['igraph'] = filtered
            timings['igraph'] = elapsed

    # Save outputs if requested
    if args.save_outputs:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("\n" + "="*60)
        logger.info("Saving Outputs")
        logger.info("="*60)

        for name, filtered in results.items():
            output_path = output_dir / f"filtered_{name.lower()}.txt"

            with open(output_path, 'w') as f:
                for h_idx, r_idx, t_idx in filtered:
                    h = idx_to_entity[h_idx]
                    r = idx_to_relation[r_idx]
                    t = idx_to_entity[t_idx]
                    f.write(f"{h}\t{r}\t{t}\n")

            logger.info(f"Saved {name} results to {output_path}")

    # Compare results
    if len(results) >= 2:
        comparison_stats = compare_results(results, idx_to_entity, idx_to_relation)

    # Print timing summary
    logger.info("\n" + "="*60)
    logger.info("TIMING SUMMARY")
    logger.info("="*60)

    if timings:
        fastest = min(timings.values())
        for name, elapsed in sorted(timings.items(), key=lambda x: x[1]):
            speedup = fastest / elapsed if elapsed > 0 else 0
            logger.info(f"  {name:20s}: {elapsed:8.2f}s  (speedup: {speedup:.2f}x)")
    else:
        logger.warning("No timing information available")

    # Final verdict
    logger.info("\n" + "="*60)
    logger.info("RECOMMENDATION")
    logger.info("="*60)

    if len(results) >= 2:
        # Check if all results match
        result_sets = [set(tuple(t) for t in r) for r in results.values()]
        all_match = all(s == result_sets[0] for s in result_sets)

        if all_match:
            logger.info("✓ All implementations produce IDENTICAL results!")
            logger.info("  → Choose based on performance:")
            if timings:
                fastest_impl = min(timings.items(), key=lambda x: x[1])[0]
                logger.info(f"    Fastest: {fastest_impl}")
        else:
            logger.warning("⚠️  Implementations produce DIFFERENT results!")
            logger.warning("  → This indicates a bug in one or more implementations")
            logger.warning("  → Investigate differences before using in production")
    else:
        logger.info("Need at least 2 implementations to compare")

    logger.info("="*60)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main()
