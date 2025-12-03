"""
Alternative implementation using igraph for filtering training triples.

igraph is faster than NetworkX while still being relatively easy to use.
This provides another independent validation of the PyG implementation.

Installation: pip install igraph
"""

import logging
import argparse
from typing import List, Set, Tuple, Dict, Optional
from pathlib import Path
import numpy as np
import igraph as ig
from collections import defaultdict
import time
import pickle

logger = logging.getLogger(__name__)


class IGraphProximityFilter:
    """Filter training triples using igraph."""

    def __init__(self, training_triples: np.ndarray, cache_path: Optional[str] = None):
        """Initialize with training triples.

        Args:
            training_triples: Array of shape (N, 3) with [head, relation, tail]
            cache_path: Optional path to cached graph file
        """
        self.training_triples = training_triples
        self.graph = None
        self.edge_to_triples = defaultdict(list)

        # Try to load from cache first
        if cache_path and Path(cache_path).exists():
            if self._load_graph_cache(cache_path):
                return

        # Build graph if cache not available
        self._build_graph()

        # Save to cache if path provided
        if cache_path:
            self._save_graph_cache(cache_path)

    def _build_graph(self):
        """Build igraph graph from training triples."""
        logger.info("Building igraph graph...")

        # Create edge list
        edges = []
        edge_triple_map = defaultdict(list)

        for idx, (h, r, t) in enumerate(self.training_triples):
            h, t = int(h), int(t)

            # igraph uses vertex indices, so we need consecutive integers
            # The triples should already be in this format
            edges.append((h, t))

            # Map edge to triple indices (both directions)
            edge_triple_map[(h, t)].append(idx)
            edge_triple_map[(t, h)].append(idx)

        # Create undirected graph
        # igraph will automatically handle vertex creation
        num_nodes = max(self.training_triples[:, [0, 2]].max() + 1, 0)

        self.graph = ig.Graph(n=num_nodes, edges=edges, directed=False)

        # Store triple indices as edge attributes
        for edge in self.graph.es:
            src, dst = edge.tuple
            # Store triple indices for this edge
            edge['triple_indices'] = edge_triple_map[(src, dst)]

        # Store in our dict too for quick lookup
        self.edge_to_triples = edge_triple_map

        logger.info(f"Built igraph graph: {self.graph.vcount()} nodes, "
                   f"{self.graph.ecount()} edges")

    def _save_graph_cache(self, cache_path: str):
        """Save graph and edge mapping to cache file.

        Args:
            cache_path: Path to save cache file
        """
        try:
            cache_data = {
                'graph': self.graph,
                'edge_to_triples': dict(self.edge_to_triples)
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved graph cache to {cache_path}")
            logger.info(f"  Nodes: {self.graph.vcount()}, Edges: {self.graph.ecount()}")
        except Exception as e:
            logger.warning(f"Failed to save graph cache: {e}")

    def _load_graph_cache(self, cache_path: str) -> bool:
        """Load graph and edge mapping from cache file.

        Args:
            cache_path: Path to cache file

        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            logger.info(f"Loading graph from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            self.graph = cache_data['graph']
            self.edge_to_triples = defaultdict(list, cache_data['edge_to_triples'])

            logger.info(f"Loaded graph from cache successfully")
            logger.info(f"  Nodes: {self.graph.vcount()}, Edges: {self.graph.ecount()}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load graph cache: {e}")
            return False

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
            if source >= self.graph.vcount():
                logger.warning(f"Source node {source} not in graph")
                continue

            # Get shortest path lengths from this source
            # igraph's distances() doesn't support cutoff, so we get all distances
            # and filter by max_hops
            source_distances = self.graph.distances(source=source)[0]

            # Process distances (igraph returns inf for unreachable nodes)
            for node_id, dist in enumerate(source_distances):
                if dist == float('inf') or dist > max_hops:
                    continue

                dist = int(dist)

                # Keep minimum distance across all sources
                if node_id not in distances or dist < distances[node_id]:
                    distances[node_id] = dist

        return distances

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
            if drug >= self.graph.vcount():
                logger.warning(f"Drug node {drug} not in graph")
                continue

            for disease in disease_nodes:
                if disease >= self.graph.vcount():
                    logger.warning(f"Disease node {disease} not in graph")
                    continue

                if drug == disease:
                    continue

                # Find all simple paths up to cutoff length
                # igraph's get_all_simple_paths returns paths as lists of vertex IDs
                try:
                    paths = self.graph.get_all_simple_paths(
                        drug, to=disease, cutoff=max_path_length
                    )

                    if paths:
                        pairs_with_paths += 1
                        for path in paths:
                            all_paths.append(path)
                            path_len = len(path) - 1  # Number of edges
                            path_lengths[path_len] += 1

                except Exception as e:
                    logger.warning(f"Error finding paths from {drug} to {disease}: {e}")
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

    def filter_for_test_triples(
        self,
        test_triples: np.ndarray,
        n_hops: int = 2,
        min_degree: int = 2,
        preserve_test_entity_edges: bool = True,
        path_filtering: bool = False,
        max_total_path_length: int = None,
        strict_path_filtering: bool = False
    ) -> Tuple[np.ndarray, Dict]:
        """Filter training triples based on proximity to test triples.

        Args:
            test_triples: Array of test triples (M, 3)
            n_hops: Number of hops from test entities
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: Always keep edges with test entities
            path_filtering: Only keep edges on paths between drugs and diseases
            max_total_path_length: Maximum total path length (optional)
            strict_path_filtering: If True, ONLY keep edges on paths (ignores other rules)

        Returns:
            Tuple of (filtered_triples, diagnostics_dict)
        """
        logger.info(f"Filtering for {len(test_triples)} test triples")
        logger.info(f"Parameters: n_hops={n_hops}, min_degree={min_degree}, "
                   f"path_filtering={path_filtering}, strict_path_filtering={strict_path_filtering}")

        # Separate drugs (heads) and diseases (tails)
        drug_nodes = list(set(int(h) for h, r, t in test_triples))
        disease_nodes = list(set(int(t) for h, r, t in test_triples))
        test_entities = set(drug_nodes) | set(disease_nodes)

        logger.info(f"Drug nodes (heads): {len(drug_nodes)}")
        logger.info(f"Disease nodes (tails): {len(disease_nodes)}")

        # Compute hop distances from drugs and diseases
        logger.info("Computing hop distances from drug nodes...")
        drug_distances = self.compute_hop_distances_from_nodes(drug_nodes, n_hops)

        logger.info("Computing hop distances from disease nodes...")
        disease_distances = self.compute_hop_distances_from_nodes(disease_nodes, n_hops)

        # Find intersection: nodes reachable from BOTH drugs and diseases
        drug_reachable = set(drug_distances.keys())
        disease_reachable = set(disease_distances.keys())
        intersection_nodes = drug_reachable & disease_reachable

        logger.info(f"Nodes reachable from drugs: {len(drug_reachable)}")
        logger.info(f"Nodes reachable from diseases: {len(disease_reachable)}")
        logger.info(f"Intersection (reachable from both): {len(intersection_nodes)}")

        # Path-based filtering: Find all paths and extract edges
        edges_on_paths = None
        path_stats = None
        if path_filtering:
            # Determine max path length
            if max_total_path_length is not None:
                max_path_len = max_total_path_length
            else:
                max_path_len = n_hops * 2  # Default: n_hops from drug + n_hops to disease

            logger.info(f"Finding all paths between drugs and diseases (cutoff={max_path_len})...")
            all_paths, path_stats = self.find_all_paths_between_nodes(
                drug_nodes, disease_nodes, max_path_len
            )
            edges_on_paths = self.extract_edges_from_paths(all_paths)

        # Filter edges
        filtered_triple_indices = set()
        edges_kept = 0

        # Get degrees
        degrees = self.graph.degree()

        # Iterate over all edges
        for edge in self.graph.es:
            src, dst = edge.tuple

            # Ensure consistent ordering
            if src > dst:
                src, dst = dst, src

            # Check intersection constraint
            if src not in intersection_nodes or dst not in intersection_nodes:
                continue

            should_keep = False

            # Strict path filtering mode: ONLY keep edges on paths
            if strict_path_filtering:
                if path_filtering and edges_on_paths is not None:
                    edge_tuple = (min(src, dst), max(src, dst))
                    if edge_tuple in edges_on_paths:
                        should_keep = True
                else:
                    # Strict mode requires path_filtering to be enabled
                    logger.warning("strict_path_filtering=True but path_filtering=False, no edges will be kept!")
            else:
                # Normal filtering mode: Apply rules in order

                # Rule 1: Preserve test entity edges
                if preserve_test_entity_edges:
                    if src in test_entities or dst in test_entities:
                        should_keep = True

                # Rule 2: Check degree threshold
                if not should_keep:
                    src_degree = degrees[src]
                    dst_degree = degrees[dst]

                    if src_degree >= min_degree or dst_degree >= min_degree:
                        should_keep = True

                # Rule 3: Path filtering - check if edge is on enumerated paths
                if should_keep and path_filtering:
                    if edges_on_paths is not None:
                        edge_tuple = (min(src, dst), max(src, dst))
                        if edge_tuple not in edges_on_paths:
                            should_keep = False

            # Add triple indices if keeping this edge
            if should_keep:
                edges_kept += 1
                triple_indices = edge['triple_indices']
                filtered_triple_indices.update(triple_indices)

        # Get filtered triples
        filtered_indices = sorted(list(filtered_triple_indices))
        filtered_triples = self.training_triples[filtered_indices]

        reduction_pct = (1 - len(filtered_triples) / len(self.training_triples)) * 100
        logger.info(f"\nFiltering Results:")
        logger.info(f"  Original triples: {len(self.training_triples)}")
        logger.info(f"  Filtered triples: {len(filtered_triples)}")
        logger.info(f"  Edges kept: {edges_kept}")
        logger.info(f"  Reduction: {reduction_pct:.1f}%")

        diagnostics = {
            'num_triples': len(filtered_triples),
            'edges_kept': edges_kept,
            'reduction_pct': reduction_pct
        }

        return filtered_triples, diagnostics


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


def load_or_create_mappings_cache(
    train_file: str,
    test_file: str,
    cache_path: Optional[str] = None
) -> Tuple[Dict, Dict, Dict, np.ndarray, np.ndarray]:
    """Load mappings and numeric triples from cache, or create and cache them.

    Args:
        train_file: Path to training triples file
        test_file: Path to test triples file
        cache_path: Optional path to cache file for mappings

    Returns:
        Tuple of (entity_to_idx, idx_to_entity, relation_to_idx, train_numeric, test_numeric)
    """
    # Try to load mappings from cache first
    entity_to_idx = None
    idx_to_entity = None
    relation_to_idx = None
    train_numeric = None
    cache_loaded = False

    if cache_path and Path(cache_path).exists():
        try:
            logger.info(f"Loading mappings and train numeric from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            entity_to_idx = cache_data['entity_to_idx']
            idx_to_entity = cache_data['idx_to_entity']
            relation_to_idx = cache_data['relation_to_idx']
            train_numeric = cache_data['train_numeric']
            cache_loaded = True

            logger.info(f"Loaded from cache: {len(entity_to_idx)} entities, "
                       f"{len(relation_to_idx)} relations, "
                       f"{len(train_numeric)} train triples")

        except Exception as e:
            logger.warning(f"Failed to load mappings cache: {e}")
            logger.info("Rebuilding mappings...")
            entity_to_idx = None
            cache_loaded = False

    # Load test triples (always loaded fresh, never cached)
    logger.info(f"Loading test triples from {test_file}")
    test_triples = load_triples_from_file(test_file)

    # If mappings not in cache, build them
    if entity_to_idx is None:
        logger.info(f"Loading training triples from {train_file}")
        train_triples = load_triples_from_file(train_file)

        logger.info(f"Loaded {len(train_triples)} training, {len(test_triples)} test triples")

        # Create mappings
        logger.info("Creating entity/relation mappings...")
        entity_to_idx, idx_to_entity, relation_to_idx = create_entity_mappings(
            train_triples, test_triples
        )
        logger.info(f"Entities: {len(entity_to_idx)}, Relations: {len(relation_to_idx)}")

        # Convert training to numeric
        logger.info("Converting training triples to numeric format...")
        train_numeric = np.array([
            [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
            for h, r, t in train_triples
        ])
    else:
        logger.info(f"Using cached mappings for {len(test_triples)} test triples")

    # Always convert test triples to numeric (never cached)
    test_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in test_triples
    ])

    # Save to cache if we just built the mappings (not loaded from cache)
    if cache_path and not cache_loaded and entity_to_idx is not None:
        try:
            logger.info(f"Saving mappings to cache: {cache_path}")
            cache_dir = Path(cache_path).parent
            cache_dir.mkdir(parents=True, exist_ok=True)

            # NOTE: We do NOT cache test_numeric because it changes in batch processing
            cache_data = {
                'entity_to_idx': entity_to_idx,
                'idx_to_entity': idx_to_entity,
                'relation_to_idx': relation_to_idx,
                'train_numeric': train_numeric
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Mappings cache saved successfully (test triples NOT cached)")

        except Exception as e:
            logger.warning(f"Failed to save mappings cache: {e}")

    return entity_to_idx, idx_to_entity, relation_to_idx, train_numeric, test_numeric


def main():
    parser = argparse.ArgumentParser(
        description='Filter training triples using igraph (faster than NetworkX)',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--train', type=str, required=True,
                       help='Path to training triples file')
    parser.add_argument('--test', type=str, required=True,
                       help='Path to test triples file')
    parser.add_argument('--output', '-o', type=str, required=True,
                       help='Path to save filtered results')
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
    parser.add_argument('--strict-path-filtering', action='store_true',
                       help='ONLY keep edges on paths (ignores degree and test entity edge rules)')
    parser.add_argument('--cache', type=str, default=None,
                       help='Path to cache file for graph (speeds up repeated runs)')
    parser.add_argument('--mappings-cache', type=str, default=None,
                       help='Path to cache file for entity/relation mappings and numeric triples (speeds up repeated runs)')

    args = parser.parse_args()

    # Determine mappings cache path
    # If --mappings-cache is not provided but --cache is, derive it from --cache
    mappings_cache = args.mappings_cache
    if mappings_cache is None and args.cache:
        cache_path = Path(args.cache)
        mappings_cache = str(cache_path.parent / f"{cache_path.stem}_mappings.pkl")

    # Load or create mappings with caching
    entity_to_idx, idx_to_entity, relation_to_idx, train_numeric, test_numeric = \
        load_or_create_mappings_cache(args.train, args.test, mappings_cache)

    # Run igraph filter
    logger.info("\n" + "="*60)
    logger.info("Running igraph Filter")
    logger.info("="*60)

    start_time = time.time()
    ig_filter = IGraphProximityFilter(train_numeric, cache_path=args.cache)

    ig_filtered, diagnostics = ig_filter.filter_for_test_triples(
        test_numeric,
        n_hops=args.n_hops,
        min_degree=args.min_degree,
        preserve_test_entity_edges=args.preserve_test_edges,
        path_filtering=args.path_filtering,
        max_total_path_length=args.max_total_path_length,
        strict_path_filtering=args.strict_path_filtering
    )

    elapsed = time.time() - start_time
    logger.info(f"\nigraph filtering completed in {elapsed:.2f}s")

    # Save results
    logger.info(f"\nSaving results to {args.output}")
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    idx_to_relation = {v: k for k, v in relation_to_idx.items()}

    with open(args.output, 'w') as f:
        for h_idx, r_idx, t_idx in ig_filtered:
            h = idx_to_entity[h_idx]
            r = idx_to_relation[r_idx]
            t = idx_to_entity[t_idx]
            f.write(f"{h}\t{r}\t{t}\n")

    logger.info(f"Saved {len(ig_filtered)} triples")


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    main()
