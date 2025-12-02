"""
Validate filter_training_by_proximity_pyg.py results using NetworkX.

This module provides an independent implementation using NetworkX to:
1. Filter training triples based on N-hop proximity to test triples
2. Validate results from the PyG implementation
3. Provide detailed diagnostics about edges and paths

NetworkX is slower but more transparent for debugging.
"""

import logging
import argparse
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import numpy as np
import networkx as nx
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class NetworkXProximityFilter:
    """Filter training triples using NetworkX for transparent path analysis."""

    def __init__(self, training_triples: np.ndarray):
        """Initialize with training triples.

        Args:
            training_triples: Array of shape (N, 3) with [head, relation, tail]
        """
        self.training_triples = training_triples
        self.graph = None
        self.edge_to_triples = defaultdict(list)
        self._build_graph()

    def _build_graph(self):
        """Build NetworkX undirected graph from training triples."""
        logger.info("Building NetworkX graph...")
        self.graph = nx.Graph()

        for idx, (h, r, t) in enumerate(self.training_triples):
            h, t = int(h), int(t)

            # Add edge (NetworkX handles undirected automatically)
            if not self.graph.has_edge(h, t):
                self.graph.add_edge(h, t, triple_indices=[])

            # Store triple index on the edge
            self.graph[h][t]['triple_indices'].append(idx)

            # Also maintain separate mapping for quick lookup
            self.edge_to_triples[(h, t)].append(idx)
            self.edge_to_triples[(t, h)].append(idx)

        logger.info(f"Built NetworkX graph: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")

    def compute_hop_distances_from_nodes(
        self,
        source_nodes: List[int],
        max_hops: int
    ) -> Dict[int, int]:
        """Compute minimum hop distance from any source node to all reachable nodes.

        Args:
            source_nodes: List of source node IDs
            max_hops: Maximum number of hops to explore

        Returns:
            Dictionary mapping node_id -> minimum_hop_distance
        """
        distances = {}

        for source in source_nodes:
            if source not in self.graph:
                logger.warning(f"Source node {source} not in graph")
                continue

            # Get shortest path lengths from this source within cutoff
            source_distances = nx.single_source_shortest_path_length(
                self.graph, source, cutoff=max_hops
            )

            # Keep minimum distance across all sources
            for node, dist in source_distances.items():
                if node not in distances or dist < distances[node]:
                    distances[node] = dist

        return distances

    def is_edge_on_drug_disease_path(
        self,
        src: int,
        dst: int,
        drug_distances: Dict[int, int],
        disease_distances: Dict[int, int],
        max_hops: int,
        max_total_path_length: Optional[int] = None
    ) -> Tuple[bool, str]:
        """Check if edge lies on a path between drug and disease nodes.

        An edge (src, dst) is on a valid path if it forms a monotonic progression
        from drug to disease.

        Args:
            src: Source node of edge
            dst: Destination node of edge
            drug_distances: Dict of node -> hop distance from drugs
            disease_distances: Dict of node -> hop distance from diseases
            max_hops: Maximum hops allowed from each endpoint
            max_total_path_length: Maximum total path length (optional)

        Returns:
            Tuple of (is_valid, reason_string)
        """
        # Check if both endpoints are reachable
        if src not in drug_distances:
            return False, f"src {src} not reachable from drugs"
        if src not in disease_distances:
            return False, f"src {src} not reachable from diseases"
        if dst not in drug_distances:
            return False, f"dst {dst} not reachable from drugs"
        if dst not in disease_distances:
            return False, f"dst {dst} not reachable from diseases"

        src_drug_dist = drug_distances[src]
        src_disease_dist = disease_distances[src]
        dst_drug_dist = drug_distances[dst]
        dst_disease_dist = disease_distances[dst]

        # Check if within hop limits
        if src_drug_dist > max_hops or src_disease_dist > max_hops:
            return False, f"src distances exceed max_hops: drug={src_drug_dist}, disease={src_disease_dist}"
        if dst_drug_dist > max_hops or dst_disease_dist > max_hops:
            return False, f"dst distances exceed max_hops: drug={dst_drug_dist}, disease={dst_disease_dist}"

        # Path direction 1: drug -> src -> dst -> disease
        # src closer to drug, dst closer to disease
        path1_valid = (src_drug_dist <= dst_drug_dist and dst_disease_dist <= src_disease_dist)
        path1_length = src_drug_dist + dst_disease_dist if path1_valid else None

        # Check total path length for path1
        if path1_valid and max_total_path_length is not None:
            if path1_length > max_total_path_length:
                path1_valid = False

        # Path direction 2: drug -> dst -> src -> disease
        # dst closer to drug, src closer to disease
        path2_valid = (dst_drug_dist <= src_drug_dist and src_disease_dist <= dst_disease_dist)
        path2_length = dst_drug_dist + src_disease_dist if path2_valid else None

        # Check total path length for path2
        if path2_valid and max_total_path_length is not None:
            if path2_length > max_total_path_length:
                path2_valid = False

        if path1_valid or path2_valid:
            reason = f"Valid path: "
            if path1_valid:
                reason += f"drug->src({src_drug_dist})->dst({dst_disease_dist})->disease (len={path1_length})"
            if path2_valid:
                if path1_valid:
                    reason += " OR "
                reason += f"drug->dst({dst_drug_dist})->src({src_disease_dist})->disease (len={path2_length})"
            return True, reason
        else:
            return False, f"No monotonic path: src(drug={src_drug_dist},dis={src_disease_dist}), dst(drug={dst_drug_dist},dis={dst_disease_dist})"

    def filter_for_test_triples(
        self,
        test_triples: np.ndarray,
        n_hops: int = 2,
        min_degree: int = 2,
        preserve_test_entity_edges: bool = True,
        path_filtering: bool = False,
        max_total_path_length: Optional[int] = None,
        verbose: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Filter training triples based on proximity to test triples.

        Args:
            test_triples: Array of test triples (M, 3)
            n_hops: Number of hops from test entities
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: Always keep edges with test entities
            path_filtering: Only keep edges on paths between drugs and diseases
            max_total_path_length: Maximum total path length (optional)
            verbose: Print detailed diagnostics

        Returns:
            Tuple of (filtered_triples, diagnostics_dict)
        """
        logger.info(f"Filtering for {len(test_triples)} test triples")
        logger.info(f"Parameters: n_hops={n_hops}, min_degree={min_degree}, "
                   f"path_filtering={path_filtering}")

        # Separate drugs (heads) and diseases (tails)
        drug_nodes = set(int(h) for h, r, t in test_triples)
        disease_nodes = set(int(t) for h, r, t in test_triples)
        test_entities = drug_nodes | disease_nodes

        logger.info(f"Drug nodes (heads): {len(drug_nodes)}")
        logger.info(f"Disease nodes (tails): {len(disease_nodes)}")

        # Compute hop distances from drugs and diseases
        logger.info("Computing hop distances from drug nodes...")
        drug_distances = self.compute_hop_distances_from_nodes(
            list(drug_nodes), n_hops
        )

        logger.info("Computing hop distances from disease nodes...")
        disease_distances = self.compute_hop_distances_from_nodes(
            list(disease_nodes), n_hops
        )

        # Find intersection: nodes reachable from BOTH drugs and diseases
        drug_reachable = set(drug_distances.keys())
        disease_reachable = set(disease_distances.keys())
        intersection_nodes = drug_reachable & disease_reachable

        logger.info(f"Nodes reachable from drugs: {len(drug_reachable)}")
        logger.info(f"Nodes reachable from diseases: {len(disease_reachable)}")
        logger.info(f"Intersection (reachable from both): {len(intersection_nodes)}")

        # Filter edges
        filtered_triple_indices = set()
        edge_diagnostics = {
            'total_edges': 0,
            'in_intersection': 0,
            'meets_degree': 0,
            'preserved_test_edges': 0,
            'on_valid_path': 0,
            'final_kept': 0,
            'rejection_reasons': defaultdict(int)
        }

        # Iterate over all edges in the graph
        for src, dst in self.graph.edges():
            edge_diagnostics['total_edges'] += 1

            # Skip if already processed (since graph is undirected)
            if src > dst:
                continue

            should_keep = False
            rejection_reason = None

            # Check intersection constraint
            if src not in intersection_nodes or dst not in intersection_nodes:
                rejection_reason = "not_in_intersection"
                edge_diagnostics['rejection_reasons'][rejection_reason] += 1
                continue

            edge_diagnostics['in_intersection'] += 1

            # Rule 1: Preserve test entity edges
            if preserve_test_entity_edges:
                if src in test_entities or dst in test_entities:
                    should_keep = True
                    edge_diagnostics['preserved_test_edges'] += 1

            # Rule 2: Check degree threshold
            if not should_keep:
                src_degree = self.graph.degree(src)
                dst_degree = self.graph.degree(dst)

                if src_degree >= min_degree or dst_degree >= min_degree:
                    should_keep = True
                    edge_diagnostics['meets_degree'] += 1
                else:
                    rejection_reason = "low_degree"

            # Rule 3: Path filtering
            if should_keep and path_filtering:
                is_on_path, reason = self.is_edge_on_drug_disease_path(
                    src, dst, drug_distances, disease_distances,
                    n_hops, max_total_path_length
                )

                if is_on_path:
                    edge_diagnostics['on_valid_path'] += 1
                    if verbose:
                        logger.info(f"Edge ({src}, {dst}): {reason}")
                else:
                    should_keep = False
                    rejection_reason = "not_on_valid_path"
                    if verbose:
                        logger.info(f"Edge ({src}, {dst}) REJECTED: {reason}")

            if not should_keep and rejection_reason:
                edge_diagnostics['rejection_reasons'][rejection_reason] += 1

            # Add triple indices if keeping this edge
            if should_keep:
                edge_diagnostics['final_kept'] += 1
                triple_indices = self.edge_to_triples.get((src, dst), [])
                filtered_triple_indices.update(triple_indices)

        # Get filtered triples
        filtered_indices = sorted(list(filtered_triple_indices))
        filtered_triples = self.training_triples[filtered_indices]

        reduction_pct = (1 - len(filtered_triples) / len(self.training_triples)) * 100
        logger.info(f"\nFiltering Results:")
        logger.info(f"  Original triples: {len(self.training_triples)}")
        logger.info(f"  Filtered triples: {len(filtered_triples)}")
        logger.info(f"  Reduction: {reduction_pct:.1f}%")

        logger.info(f"\nEdge Diagnostics:")
        logger.info(f"  Total edges examined: {edge_diagnostics['total_edges']}")
        logger.info(f"  Edges in intersection: {edge_diagnostics['in_intersection']}")
        logger.info(f"  Edges meeting degree threshold: {edge_diagnostics['meets_degree']}")
        logger.info(f"  Preserved test entity edges: {edge_diagnostics['preserved_test_edges']}")
        if path_filtering:
            logger.info(f"  Edges on valid paths: {edge_diagnostics['on_valid_path']}")
        logger.info(f"  Final edges kept: {edge_diagnostics['final_kept']}")

        if edge_diagnostics['rejection_reasons']:
            logger.info(f"\nRejection Reasons:")
            for reason, count in sorted(edge_diagnostics['rejection_reasons'].items()):
                logger.info(f"  {reason}: {count}")

        return filtered_triples, edge_diagnostics


def load_triples_from_file(filepath: str) -> List[List[str]]:
    """Load triples from TSV file.

    Args:
        filepath: Path to triples file

    Returns:
        List of triples [head, relation, tail]
    """
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append(parts[:3])
    return triples


def create_entity_mappings(train_triples: List, test_triples: List) -> Tuple[Dict, Dict, Dict]:
    """Create entity and relation mappings.

    Args:
        train_triples: List of training triples
        test_triples: List of test triples

    Returns:
        Tuple of (entity_to_idx, idx_to_entity, relation_to_idx)
    """
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


def compare_triple_sets(
    networkx_triples: np.ndarray,
    pyg_triples: np.ndarray,
    entity_mapping: Dict[int, str]
) -> Dict:
    """Compare two sets of filtered triples.

    Args:
        networkx_triples: Triples from NetworkX filter
        pyg_triples: Triples from PyG filter
        entity_mapping: Mapping from idx to entity string

    Returns:
        Dictionary with comparison statistics
    """
    # Convert to sets of tuples for comparison
    nx_set = set(tuple(t) for t in networkx_triples)
    pyg_set = set(tuple(t) for t in pyg_triples)

    only_in_nx = nx_set - pyg_set
    only_in_pyg = pyg_set - nx_set
    in_both = nx_set & pyg_set

    comparison = {
        'networkx_count': len(nx_set),
        'pyg_count': len(pyg_set),
        'in_both': len(in_both),
        'only_in_networkx': len(only_in_nx),
        'only_in_pyg': len(only_in_pyg),
        'match_percentage': (len(in_both) / max(len(nx_set), len(pyg_set)) * 100) if max(len(nx_set), len(pyg_set)) > 0 else 0
    }

    logger.info(f"\n{'='*60}")
    logger.info("COMPARISON RESULTS")
    logger.info(f"{'='*60}")
    logger.info(f"NetworkX filtered triples: {comparison['networkx_count']}")
    logger.info(f"PyG filtered triples:      {comparison['pyg_count']}")
    logger.info(f"In both:                   {comparison['in_both']}")
    logger.info(f"Only in NetworkX:          {comparison['only_in_networkx']}")
    logger.info(f"Only in PyG:               {comparison['only_in_pyg']}")
    logger.info(f"Match percentage:          {comparison['match_percentage']:.2f}%")

    if only_in_nx and len(only_in_nx) <= 20:
        logger.info(f"\nSample triples only in NetworkX:")
        for h, r, t in list(only_in_nx)[:20]:
            logger.info(f"  ({entity_mapping.get(h, h)}, {r}, {entity_mapping.get(t, t)})")

    if only_in_pyg and len(only_in_pyg) <= 20:
        logger.info(f"\nSample triples only in PyG:")
        for h, r, t in list(only_in_pyg)[:20]:
            logger.info(f"  ({entity_mapping.get(h, h)}, {r}, {entity_mapping.get(t, t)})")

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description='Validate PyG filter results using NetworkX',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate PyG results with NetworkX
  python validate_filter_networkx.py \\
      --train train.txt \\
      --test test.txt \\
      --pyg-output train_filtered_pyg.txt \\
      --n-hops 2 \\
      --min-degree 2 \\
      --path-filtering

  # Run NetworkX filter only (no comparison)
  python validate_filter_networkx.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered_nx.txt \\
      --n-hops 2 \\
      --path-filtering \\
      --verbose
        """
    )

    parser.add_argument('--train', type=str, required=True,
                       help='Path to training triples file')
    parser.add_argument('--test', type=str, required=True,
                       help='Path to test triples file')
    parser.add_argument('--output', '-o', type=str,
                       help='Path to save NetworkX filtered results')
    parser.add_argument('--pyg-output', type=str,
                       help='Path to PyG filtered results (for comparison)')
    parser.add_argument('--n-hops', type=int, default=2,
                       help='Number of hops from test triples (default: 2)')
    parser.add_argument('--min-degree', type=int, default=2,
                       help='Minimum degree threshold (default: 2)')
    parser.add_argument('--preserve-test-edges', action='store_true', default=True,
                       help='Preserve edges with test entities (default: True)')
    parser.add_argument('--path-filtering', action='store_true',
                       help='Only keep edges on paths between drug and disease')
    parser.add_argument('--max-total-path-length', type=int, default=None,
                       help='Maximum total path length (drug_dist + disease_dist)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed edge diagnostics')

    args = parser.parse_args()

    # Load triples
    logger.info(f"Loading training triples from {args.train}")
    train_triples = load_triples_from_file(args.train)

    logger.info(f"Loading test triples from {args.test}")
    test_triples = load_triples_from_file(args.test)

    logger.info(f"Loaded {len(train_triples)} training, {len(test_triples)} test triples")

    # Create mappings
    logger.info("Creating entity/relation mappings...")
    entity_to_idx, idx_to_entity, relation_to_idx = create_entity_mappings(
        train_triples, test_triples
    )
    logger.info(f"Entities: {len(entity_to_idx)}, Relations: {len(relation_to_idx)}")

    # Convert to numeric
    train_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in train_triples
    ])

    test_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in test_triples
    ])

    # Run NetworkX filter
    logger.info("\n" + "="*60)
    logger.info("Running NetworkX Filter")
    logger.info("="*60)

    start_time = time.time()
    nx_filter = NetworkXProximityFilter(train_numeric)

    nx_filtered, diagnostics = nx_filter.filter_for_test_triples(
        test_numeric,
        n_hops=args.n_hops,
        min_degree=args.min_degree,
        preserve_test_entity_edges=args.preserve_test_edges,
        path_filtering=args.path_filtering,
        max_total_path_length=args.max_total_path_length,
        verbose=args.verbose
    )

    elapsed = time.time() - start_time
    logger.info(f"\nNetworkX filtering completed in {elapsed:.2f}s")

    # Save NetworkX results if requested
    if args.output:
        logger.info(f"\nSaving NetworkX results to {args.output}")
        output_dir = Path(args.output).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        idx_to_relation = {v: k for k, v in relation_to_idx.items()}

        with open(args.output, 'w') as f:
            for h_idx, r_idx, t_idx in nx_filtered:
                h = idx_to_entity[h_idx]
                r = idx_to_relation[r_idx]
                t = idx_to_entity[t_idx]
                f.write(f"{h}\t{r}\t{t}\n")

        logger.info(f"Saved {len(nx_filtered)} triples")

    # Compare with PyG results if provided
    if args.pyg_output:
        logger.info(f"\nLoading PyG results from {args.pyg_output}")
        pyg_triples = load_triples_from_file(args.pyg_output)

        # Convert to numeric using same mappings
        pyg_numeric = np.array([
            [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
            for h, r, t in pyg_triples
        ])

        # Compare
        comparison = compare_triple_sets(nx_filtered, pyg_numeric, idx_to_entity)

        if comparison['match_percentage'] < 100:
            logger.warning(f"\n⚠️  Results do NOT match perfectly!")
            logger.warning(f"    This suggests a bug in one of the implementations.")
        else:
            logger.info(f"\n✓ Results match perfectly!")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main()
