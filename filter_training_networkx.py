"""
NetworkX-based implementation for filtering training triples by proximity to test triples.

This is a production-ready alternative to the PyTorch Geometric implementation.
While slower than PyG, it's more transparent and easier to debug/understand.

Key Features:
- Clear, readable implementation using NetworkX
- Multiple filtering modes (intersection, strict hop, path filtering)
- Degree-based filtering
- Comprehensive statistics and logging
- Can be used as the primary filter or for validation

Performance: Suitable for graphs with up to ~500K edges. For larger graphs, use PyG.
"""

import logging
import argparse
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import numpy as np
import networkx as nx
from collections import defaultdict
import time
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def timer(operation_name: str):
    """Context manager for timing operations."""
    start_time = time.time()
    logger.info(f"⏱️  Starting: {operation_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start_time
        logger.info(f"✓ Completed {operation_name} in {elapsed:.2f}s")


class NetworkXProximityFilter:
    """
    Filter training triples using NetworkX based on proximity to test triples.

    This implementation provides transparent, easy-to-understand filtering logic
    for knowledge graph triple filtering based on N-hop neighborhoods.
    """

    def __init__(self, training_triples: np.ndarray):
        """Initialize with training triples.

        Args:
            training_triples: Array of shape (N, 3) with [head, relation, tail]
        """
        self.training_triples = training_triples
        self.graph = None
        self.edge_to_triples = defaultdict(list)
        self.relation_info = {}
        self._build_graph()

    def _build_graph(self):
        """Build NetworkX undirected graph from training triples."""
        with timer("Building NetworkX graph"):
            self.graph = nx.Graph()

            for idx, (h, r, t) in enumerate(self.training_triples):
                h, t, r = int(h), int(t), int(r)

                # Add edge (NetworkX handles undirected automatically)
                if not self.graph.has_edge(h, t):
                    self.graph.add_edge(h, t)
                    # Store triple indices for this edge
                    self.graph[h][t]['triple_indices'] = []
                    self.graph[h][t]['relations'] = set()

                # Add triple index and relation info to edge
                self.graph[h][t]['triple_indices'].append(idx)
                self.graph[h][t]['relations'].add(r)

                # Maintain separate mapping for quick lookup
                self.edge_to_triples[(h, t)].append(idx)
                self.edge_to_triples[(t, h)].append(idx)

                # Store relation info
                self.relation_info[idx] = r

            logger.info(f"Built NetworkX graph: {self.graph.number_of_nodes()} nodes, "
                       f"{self.graph.number_of_edges()} edges")

    def compute_hop_distances_from_nodes(
        self,
        source_nodes: List[int],
        max_hops: int
    ) -> Dict[int, int]:
        """
        Compute minimum hop distance from any source node to all reachable nodes.

        Uses BFS from each source and keeps minimum distance.

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

    def get_n_hop_neighborhood(
        self,
        source_nodes: List[int],
        n_hops: int
    ) -> Set[int]:
        """
        Get all nodes within n_hops of any source node.

        Args:
            source_nodes: List of source node IDs
            n_hops: Number of hops

        Returns:
            Set of node IDs within n_hops
        """
        distances = self.compute_hop_distances_from_nodes(source_nodes, n_hops)
        return set(distances.keys())

    def find_all_paths_between_nodes(
        self,
        drug_nodes: List[int],
        disease_nodes: List[int],
        max_path_length: int
    ) -> Tuple[List[List[int]], Dict]:
        """
        Find all simple paths between drug and disease nodes up to max_path_length.

        Args:
            drug_nodes: List of drug (source) node IDs
            disease_nodes: List of disease (target) node IDs
            max_path_length: Maximum path length to consider

        Returns:
            Tuple of (all_paths, path_statistics)
            - all_paths: List of paths, where each path is a list of node IDs
            - path_statistics: Dict with path count by length
        """
        all_paths = []
        path_lengths = defaultdict(int)

        logger.info(f"Finding all simple paths between {len(drug_nodes)} drugs and "
                   f"{len(disease_nodes)} diseases (cutoff={max_path_length})...")

        total_pairs = len(drug_nodes) * len(disease_nodes)
        pairs_with_paths = 0

        for drug in drug_nodes:
            if drug not in self.graph:
                logger.warning(f"Drug node {drug} not in graph")
                continue

            for disease in disease_nodes:
                if disease not in self.graph:
                    logger.warning(f"Disease node {disease} not in graph")
                    continue

                if drug == disease:
                    continue

                # Check if path exists first
                try:
                    if not nx.has_path(self.graph, drug, disease):
                        continue
                except nx.NodeNotFound:
                    continue

                # Find all simple paths up to cutoff length
                try:
                    paths = list(nx.all_simple_paths(
                        self.graph, drug, disease, cutoff=max_path_length
                    ))

                    if paths:
                        pairs_with_paths += 1
                        for path in paths:
                            all_paths.append(path)
                            path_len = len(path) - 1  # Number of edges
                            path_lengths[path_len] += 1

                except nx.NetworkXNoPath:
                    continue

        # Create statistics
        stats = {
            'total_paths': len(all_paths),
            'drug_disease_pairs': total_pairs,
            'pairs_with_paths': pairs_with_paths,
            'path_length_distribution': dict(sorted(path_lengths.items())),
            'shortest_path_length': min(path_lengths.keys()) if path_lengths else None,
            'longest_path_length': max(path_lengths.keys()) if path_lengths else None,
            'avg_path_length': (sum(l * c for l, c in path_lengths.items()) /
                               len(all_paths)) if all_paths else 0
        }

        logger.info(f"Found {len(all_paths)} paths connecting {pairs_with_paths}/{total_pairs} "
                   f"drug-disease pairs")
        if path_lengths:
            logger.info(f"Path length distribution:")
            for length in sorted(path_lengths.keys()):
                count = path_lengths[length]
                logger.info(f"  {length}-hop paths: {count}")

        return all_paths, stats

    def extract_edges_from_paths(
        self,
        paths: List[List[int]]
    ) -> Set[Tuple[int, int]]:
        """
        Extract all unique edges from a list of paths.

        Args:
            paths: List of paths, where each path is a list of node IDs

        Returns:
            Set of edges (as tuples with smaller node ID first)
        """
        edges_on_paths = set()

        for path in paths:
            for i in range(len(path) - 1):
                # Store edge with consistent ordering (smaller ID first)
                src, dst = path[i], path[i + 1]
                edge = (min(src, dst), max(src, dst))
                edges_on_paths.add(edge)

        logger.info(f"Extracted {len(edges_on_paths)} unique edges from {len(paths)} paths")

        return edges_on_paths

    def filter_for_single_test_triple(
        self,
        test_triple: Tuple[int, int, int],
        n_hops: int = 2,
        min_degree: int = 2,
        preserve_test_entity_edges: bool = True,
        strict_hop_constraint: bool = False,
        path_filtering: bool = False,
        max_total_path_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Filter training triples for a single test triple.

        Args:
            test_triple: Single test triple (head, relation, tail)
            n_hops: Number of hops from test entities
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: If True, always keep edges with test entities
            strict_hop_constraint: If True, enforce that BOTH endpoints are within n_hops
            path_filtering: If True, only keep edges on paths between drug and disease
            max_total_path_length: Maximum total path length (optional)

        Returns:
            Filtered training triples array
        """
        test_h, test_r, test_t = test_triple
        test_h, test_t = int(test_h), int(test_t)

        logger.info(f"Filtering for test triple: ({test_h}, {test_r}, {test_t})")
        logger.info(f"Parameters: n_hops={n_hops}, min_degree={min_degree}, "
                   f"strict_hop_constraint={strict_hop_constraint}, "
                   f"path_filtering={path_filtering}")

        # Get n-hop neighborhoods
        with timer(f"Computing {n_hops}-hop neighborhoods"):
            drug_nodes = [test_h]
            disease_nodes = [test_t]

            head_neighborhood = self.get_n_hop_neighborhood(drug_nodes, n_hops)
            tail_neighborhood = self.get_n_hop_neighborhood(disease_nodes, n_hops)

            # Intersection: nodes reachable from BOTH head and tail
            intersection_nodes = head_neighborhood & tail_neighborhood

            # Always include test entities
            intersection_nodes.add(test_h)
            intersection_nodes.add(test_t)

            logger.info(f"Head neighborhood: {len(head_neighborhood)} nodes")
            logger.info(f"Tail neighborhood: {len(tail_neighborhood)} nodes")
            logger.info(f"Intersection: {len(intersection_nodes)} nodes")

        # Compute hop distances if needed for strict mode
        hop_distances = None
        if strict_hop_constraint:
            with timer("Computing hop distances (strict mode)"):
                hop_distances = self.compute_hop_distances_from_nodes(
                    [test_h, test_t], n_hops
                )

        # Path-based filtering: Find all paths and extract edges
        edges_on_paths = None
        path_stats = None
        if path_filtering:
            # Determine max path length
            if max_total_path_length is not None:
                max_path_len = max_total_path_length
            else:
                max_path_len = n_hops * 2

            with timer(f"Finding all paths between drug and disease (cutoff={max_path_len})"):
                all_paths, path_stats = self.find_all_paths_between_nodes(
                    drug_nodes, disease_nodes, max_path_len
                )
                edges_on_paths = self.extract_edges_from_paths(all_paths)

        # Filter edges
        with timer("Filtering edges"):
            filtered_triple_indices = self._filter_edges(
                intersection_nodes=intersection_nodes,
                test_entities={test_h, test_t},
                min_degree=min_degree,
                preserve_test_entity_edges=preserve_test_entity_edges,
                strict_hop_constraint=strict_hop_constraint,
                hop_distances=hop_distances,
                n_hops=n_hops,
                path_filtering=path_filtering,
                edges_on_paths=edges_on_paths,
                path_stats=path_stats
            )

        # Get filtered triples
        filtered_indices = sorted(list(filtered_triple_indices))
        filtered_triples = self.training_triples[filtered_indices]

        reduction_pct = (1 - len(filtered_triples) / len(self.training_triples)) * 100
        logger.info(f"Filtered: {len(self.training_triples)} → {len(filtered_triples)} "
                   f"({reduction_pct:.1f}% reduction)")

        return filtered_triples

    def filter_for_multiple_test_triples(
        self,
        test_triples: np.ndarray,
        n_hops: int = 2,
        min_degree: int = 2,
        preserve_test_entity_edges: bool = True,
        strict_hop_constraint: bool = False,
        path_filtering: bool = False,
        max_total_path_length: Optional[int] = None
    ) -> np.ndarray:
        """
        Filter training triples for multiple test triples.

        Args:
            test_triples: Array of shape (M, 3) with [head, relation, tail]
            n_hops: Number of hops from test entities
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: If True, always keep edges with test entities
            strict_hop_constraint: If True, enforce that BOTH endpoints are within n_hops
            path_filtering: If True, only keep edges on paths between drugs and diseases
            max_total_path_length: Maximum total path length (optional)

        Returns:
            Filtered training triples array
        """
        logger.info(f"Filtering for {len(test_triples)} test triples")
        logger.info(f"Parameters: n_hops={n_hops}, min_degree={min_degree}, "
                   f"strict_hop_constraint={strict_hop_constraint}, "
                   f"path_filtering={path_filtering}")

        # Extract test entities - separate heads (drugs) and tails (diseases)
        with timer("Extracting test entities"):
            drug_nodes = list(set(int(h) for h, r, t in test_triples))
            disease_nodes = list(set(int(t) for h, r, t in test_triples))
            test_entities = set(drug_nodes) | set(disease_nodes)

            logger.info(f"Drug nodes (heads): {len(drug_nodes)}")
            logger.info(f"Disease nodes (tails): {len(disease_nodes)}")

        # Get n-hop neighborhoods
        with timer(f"Computing {n_hops}-hop neighborhoods"):
            head_neighborhood = self.get_n_hop_neighborhood(drug_nodes, n_hops)
            tail_neighborhood = self.get_n_hop_neighborhood(disease_nodes, n_hops)

            # Intersection: nodes reachable from BOTH heads and tails
            intersection_nodes = head_neighborhood & tail_neighborhood

            # Always include test entities
            intersection_nodes.update(test_entities)

            logger.info(f"Head neighborhood: {len(head_neighborhood)} nodes")
            logger.info(f"Tail neighborhood: {len(tail_neighborhood)} nodes")
            logger.info(f"Intersection: {len(intersection_nodes)} nodes")

        # Compute hop distances if needed for strict mode
        hop_distances = None
        if strict_hop_constraint:
            with timer("Computing hop distances (strict mode)"):
                hop_distances = self.compute_hop_distances_from_nodes(
                    list(test_entities), n_hops
                )

        # Path-based filtering: Find all paths and extract edges
        edges_on_paths = None
        path_stats = None
        if path_filtering:
            # Determine max path length
            if max_total_path_length is not None:
                max_path_len = max_total_path_length
            else:
                max_path_len = n_hops * 2  # Default: n_hops from drug + n_hops to disease

            with timer(f"Finding all paths between drugs and diseases (cutoff={max_path_len})"):
                all_paths, path_stats = self.find_all_paths_between_nodes(
                    drug_nodes, disease_nodes, max_path_len
                )
                edges_on_paths = self.extract_edges_from_paths(all_paths)

        # Filter edges
        with timer("Filtering edges"):
            filtered_triple_indices = self._filter_edges(
                intersection_nodes=intersection_nodes,
                test_entities=test_entities,
                min_degree=min_degree,
                preserve_test_entity_edges=preserve_test_entity_edges,
                strict_hop_constraint=strict_hop_constraint,
                hop_distances=hop_distances,
                n_hops=n_hops,
                path_filtering=path_filtering,
                edges_on_paths=edges_on_paths,
                path_stats=path_stats
            )

        # Get filtered triples
        filtered_indices = sorted(list(filtered_triple_indices))
        filtered_triples = self.training_triples[filtered_indices]

        reduction_pct = (1 - len(filtered_triples) / len(self.training_triples)) * 100
        logger.info(f"Filtered: {len(self.training_triples)} → {len(filtered_triples)} "
                   f"({reduction_pct:.1f}% reduction)")

        return filtered_triples

    def _filter_edges(
        self,
        intersection_nodes: Set[int],
        test_entities: Set[int],
        min_degree: int,
        preserve_test_entity_edges: bool,
        strict_hop_constraint: bool,
        hop_distances: Optional[Dict[int, int]],
        n_hops: int,
        path_filtering: bool,
        edges_on_paths: Optional[Set[Tuple[int, int]]],
        path_stats: Optional[Dict]
    ) -> Set[int]:
        """
        Core edge filtering logic.

        Args:
            intersection_nodes: Set of nodes in intersection
            test_entities: Set of test entity node IDs
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: Whether to preserve test entity edges
            strict_hop_constraint: Whether to enforce strict hop constraint
            hop_distances: Hop distances from test entities (if strict mode)
            n_hops: Maximum number of hops
            path_filtering: Whether to use path filtering
            edges_on_paths: Set of edges that lie on drug-disease paths (if path filtering)
            path_stats: Path statistics dict (if path filtering)

        Returns:
            Set of filtered triple indices
        """
        filtered_triple_indices = set()

        # Iterate over edges in the graph
        for src, dst in self.graph.edges():
            # Ensure consistent ordering
            if src > dst:
                src, dst = dst, src

            # INTERSECTION CONSTRAINT: Both endpoints must be in intersection
            if src not in intersection_nodes or dst not in intersection_nodes:
                continue

            should_keep = False

            # Rule 1: Preserve test entity edges
            if preserve_test_entity_edges:
                if src in test_entities or dst in test_entities:
                    should_keep = True

            # Rule 2: Check degree threshold
            if not should_keep:
                src_degree = self.graph.degree(src)
                dst_degree = self.graph.degree(dst)

                if src_degree >= min_degree or dst_degree >= min_degree:
                    should_keep = True

            # STRICT HOP CONSTRAINT: Both endpoints must be within n_hops
            if should_keep and strict_hop_constraint:
                src_dist = hop_distances.get(src, -1)
                dst_dist = hop_distances.get(dst, -1)

                if src_dist < 0 or src_dist > n_hops or dst_dist < 0 or dst_dist > n_hops:
                    should_keep = False

            # PATH FILTERING: Edge must be on a drug-disease path
            # Now we simply check if the edge is in our pre-computed set
            if should_keep and path_filtering:
                if edges_on_paths is not None:
                    edge = (src, dst)  # Already in canonical form (src < dst)
                    if edge not in edges_on_paths:
                        should_keep = False

            # Add triple indices if keeping this edge
            if should_keep:
                # Get triple indices for this edge from graph attribute
                triple_indices = self.graph[src][dst].get('triple_indices', [])
                filtered_triple_indices.update(triple_indices)

        return filtered_triple_indices

    def get_statistics(self, filtered_triples: np.ndarray) -> Dict:
        """
        Get statistics about filtered triples.

        Args:
            filtered_triples: Filtered training triples

        Returns:
            Dictionary with statistics
        """
        # Count entities and relations
        entities = set()
        relations = set()

        for h, r, t in filtered_triples:
            entities.add(int(h))
            entities.add(int(t))
            relations.add(int(r))

        # Build subgraph for degree statistics
        subgraph = nx.Graph()
        for h, r, t in filtered_triples:
            subgraph.add_edge(int(h), int(t))

        degrees = [d for n, d in subgraph.degree()]

        stats = {
            'num_triples': len(filtered_triples),
            'num_entities': len(entities),
            'num_relations': len(relations),
            'avg_degree': np.mean(degrees) if degrees else 0,
            'max_degree': max(degrees) if degrees else 0,
            'min_degree': min(degrees) if degrees else 0,
        }

        return stats


def load_triples_from_file(filepath: str) -> List[List[str]]:
    """Load triples from TSV file."""
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append(parts[:3])
    return triples


def create_entity_mappings(
    train_triples: List,
    test_triples: List
) -> Tuple[Dict, Dict, Dict]:
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


def filter_training_file(
    train_path: str,
    test_path: str,
    output_path: str,
    n_hops: int = 2,
    min_degree: int = 2,
    preserve_test_entity_edges: bool = True,
    use_single_triple_mode: bool = False,
    strict_hop_constraint: bool = False,
    path_filtering: bool = False,
    max_total_path_length: Optional[int] = None
):
    """
    Filter training file based on proximity to test triples using NetworkX.

    Args:
        train_path: Path to training triples file
        test_path: Path to test triples file
        output_path: Path to save filtered training triples
        n_hops: Number of hops from test triples
        min_degree: Minimum degree threshold
        preserve_test_entity_edges: Keep edges with test entities
        use_single_triple_mode: If True, analyze first test triple only
        strict_hop_constraint: If True, enforce strict n-hop constraint
        path_filtering: If True, only keep edges on paths between drug and disease
        max_total_path_length: Maximum total path length when path_filtering is enabled
    """
    total_start = time.time()

    # Load training triples
    with timer("Load training triples"):
        logger.info(f"Loading training triples from {train_path}")
        train_triples = load_triples_from_file(train_path)

    # Load test triples
    with timer("Load test triples"):
        logger.info(f"Loading test triples from {test_path}")
        test_triples = load_triples_from_file(test_path)
        logger.info(f"Loaded {len(train_triples)} training, {len(test_triples)} test triples")

    # Create entity/relation mappings
    with timer("Build entity/relation mappings"):
        entity_to_idx, idx_to_entity, relation_to_idx = create_entity_mappings(
            train_triples, test_triples
        )
        logger.info(f"Created mappings: {len(entity_to_idx)} entities, "
                   f"{len(relation_to_idx)} relations")

    # Convert to numeric arrays
    with timer("Convert to numeric format"):
        train_numeric = np.array([
            [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
            for h, r, t in train_triples
        ])

        test_numeric = np.array([
            [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
            for h, r, t in test_triples
        ])

    # Create NetworkX filter
    filter_obj = NetworkXProximityFilter(train_numeric)

    # Filter triples
    if use_single_triple_mode and len(test_numeric) > 0:
        logger.info("Using single test triple mode (first triple only)")
        test_triple = tuple(test_numeric[0])
        filtered_numeric = filter_obj.filter_for_single_test_triple(
            test_triple=test_triple,
            n_hops=n_hops,
            min_degree=min_degree,
            preserve_test_entity_edges=preserve_test_entity_edges,
            strict_hop_constraint=strict_hop_constraint,
            path_filtering=path_filtering,
            max_total_path_length=max_total_path_length
        )
    else:
        logger.info("Using multiple test triples mode")
        filtered_numeric = filter_obj.filter_for_multiple_test_triples(
            test_triples=test_numeric,
            n_hops=n_hops,
            min_degree=min_degree,
            preserve_test_entity_edges=preserve_test_entity_edges,
            strict_hop_constraint=strict_hop_constraint,
            path_filtering=path_filtering,
            max_total_path_length=max_total_path_length
        )

    # Convert back to string labels
    with timer("Convert back to string labels"):
        idx_to_relation = {v: k for k, v in relation_to_idx.items()}

        filtered_triples = []
        for h_idx, r_idx, t_idx in filtered_numeric:
            h = idx_to_entity[h_idx]
            r = idx_to_relation[r_idx]
            t = idx_to_entity[t_idx]
            filtered_triples.append([h, r, t])

    # Write filtered triples
    with timer("Write filtered triples"):
        logger.info(f"Writing {len(filtered_triples)} filtered triples to {output_path}")
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for h, r, t in filtered_triples:
                f.write(f"{h}\t{r}\t{t}\n")

    # Compute statistics
    with timer("Compute statistics"):
        stats = filter_obj.get_statistics(filtered_numeric)

    total_time = time.time() - total_start

    # Print statistics
    logger.info("\n" + "="*60)
    logger.info("FILTERING STATISTICS")
    logger.info("="*60)
    logger.info(f"  Original training triples: {len(train_triples)}")
    logger.info(f"  Filtered training triples: {stats['num_triples']}")
    logger.info(f"  Reduction: {(1 - stats['num_triples']/len(train_triples))*100:.1f}%")
    logger.info(f"  Entities in filtered graph: {stats['num_entities']}")
    logger.info(f"  Relations in filtered graph: {stats['num_relations']}")
    logger.info(f"  Average degree: {stats['avg_degree']:.2f}")
    logger.info(f"  Degree range: [{stats['min_degree']}, {stats['max_degree']}]")
    logger.info("="*60)
    logger.info(f"Total execution time: {total_time:.2f}s")
    logger.info("="*60)
    logger.info(f"\nFiltered training saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Filter training triples using NetworkX (transparent implementation)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard filtering
  python filter_training_networkx.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 2 \\
      --min-degree 2

  # Path filtering mode
  python filter_training_networkx.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 2 \\
      --path-filtering \\
      --max-total-path-length 4

  # Strict hop constraint
  python filter_training_networkx.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 2 \\
      --strict-hop-constraint

Benefits:
  - Transparent, easy to understand implementation
  - Easier to debug than PyG
  - Suitable for small to medium graphs (<500K edges)
  - Good for validation and prototyping
        """
    )

    parser.add_argument('--train', type=str, required=True,
                       help='Path to training triples file')
    parser.add_argument('--test', type=str, required=True,
                       help='Path to test triples file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to output filtered training file')
    parser.add_argument('--n-hops', type=int, default=2,
                       help='Number of hops from test triples (default: 2)')
    parser.add_argument('--min-degree', type=int, default=2,
                       help='Minimum degree threshold (default: 2)')

    preserve_group = parser.add_mutually_exclusive_group()
    preserve_group.add_argument('--preserve-test-edges', dest='preserve_test_edges',
                               action='store_true',
                               help='Always preserve edges containing test entities (default)')
    preserve_group.add_argument('--no-preserve-test-edges', dest='preserve_test_edges',
                               action='store_false',
                               help='Apply strict degree filtering to all edges')
    parser.set_defaults(preserve_test_edges=True)

    parser.add_argument('--single-triple', action='store_true',
                       help='Filter for single test triple only (first one)')
    parser.add_argument('--strict-hop-constraint', action='store_true',
                       help='Enforce strict n-hop constraint: both endpoints '
                            'must be within n_hops')
    parser.add_argument('--path-filtering', action='store_true',
                       help='Only keep edges on paths between drug and disease within n_hops. '
                            'Stricter than intersection filtering.')
    parser.add_argument('--max-total-path-length', type=int, default=None,
                       help='Maximum total path length (drug_dist + disease_dist). '
                            'Only used when --path-filtering is enabled.')

    args = parser.parse_args()

    filter_training_file(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        n_hops=args.n_hops,
        min_degree=args.min_degree,
        preserve_test_entity_edges=args.preserve_test_edges,
        use_single_triple_mode=args.single_triple,
        strict_hop_constraint=args.strict_hop_constraint,
        path_filtering=args.path_filtering,
        max_total_path_length=args.max_total_path_length
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main()
