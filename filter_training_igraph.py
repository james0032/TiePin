"""
Alternative implementation using igraph for filtering training triples.

igraph is faster than NetworkX while still being relatively easy to use.
This provides another independent validation of the PyG implementation.

Installation: pip install igraph
"""

import logging
import argparse
from typing import List, Set, Tuple, Dict
from pathlib import Path
import numpy as np
import igraph as ig
from collections import defaultdict
import time

logger = logging.getLogger(__name__)


class IGraphProximityFilter:
    """Filter training triples using igraph."""

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
            # cutoff parameter limits the search depth
            source_distances = self.graph.distances(source=source, cutoff=max_hops)[0]

            # Process distances (igraph returns inf for unreachable nodes)
            for node_id, dist in enumerate(source_distances):
                if dist == float('inf') or dist > max_hops:
                    continue

                dist = int(dist)

                # Keep minimum distance across all sources
                if node_id not in distances or dist < distances[node_id]:
                    distances[node_id] = dist

        return distances

    def is_edge_on_drug_disease_path(
        self,
        src: int,
        dst: int,
        drug_distances: Dict[int, int],
        disease_distances: Dict[int, int],
        max_hops: int,
        max_total_path_length: int = None
    ) -> bool:
        """Check if edge lies on a path between drug and disease nodes.

        Args:
            src: Source node of edge
            dst: Destination node of edge
            drug_distances: Dict of node -> hop distance from drugs
            disease_distances: Dict of node -> hop distance from diseases
            max_hops: Maximum hops allowed from each endpoint
            max_total_path_length: Maximum total path length (optional)

        Returns:
            True if edge is on a valid path, False otherwise
        """
        # Check if both endpoints are reachable
        if src not in drug_distances or src not in disease_distances:
            return False
        if dst not in drug_distances or dst not in disease_distances:
            return False

        src_drug_dist = drug_distances[src]
        src_disease_dist = disease_distances[src]
        dst_drug_dist = drug_distances[dst]
        dst_disease_dist = disease_distances[dst]

        # Check if within hop limits
        if src_drug_dist > max_hops or src_disease_dist > max_hops:
            return False
        if dst_drug_dist > max_hops or dst_disease_dist > max_hops:
            return False

        # Path direction 1: drug -> src -> dst -> disease
        path1_valid = (src_drug_dist <= dst_drug_dist and dst_disease_dist <= src_disease_dist)

        # Check total path length for path1
        if path1_valid and max_total_path_length is not None:
            path1_length = src_drug_dist + dst_disease_dist
            path1_valid = path1_length <= max_total_path_length

        # Path direction 2: drug -> dst -> src -> disease
        path2_valid = (dst_drug_dist <= src_drug_dist and src_disease_dist <= dst_disease_dist)

        # Check total path length for path2
        if path2_valid and max_total_path_length is not None:
            path2_length = dst_drug_dist + src_disease_dist
            path2_valid = path2_length <= max_total_path_length

        return path1_valid or path2_valid

    def filter_for_test_triples(
        self,
        test_triples: np.ndarray,
        n_hops: int = 2,
        min_degree: int = 2,
        preserve_test_entity_edges: bool = True,
        path_filtering: bool = False,
        max_total_path_length: int = None
    ) -> Tuple[np.ndarray, Dict]:
        """Filter training triples based on proximity to test triples.

        Args:
            test_triples: Array of test triples (M, 3)
            n_hops: Number of hops from test entities
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: Always keep edges with test entities
            path_filtering: Only keep edges on paths between drugs and diseases
            max_total_path_length: Maximum total path length (optional)

        Returns:
            Tuple of (filtered_triples, diagnostics_dict)
        """
        logger.info(f"Filtering for {len(test_triples)} test triples")
        logger.info(f"Parameters: n_hops={n_hops}, min_degree={min_degree}, "
                   f"path_filtering={path_filtering}")

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

            # Rule 3: Path filtering
            if should_keep and path_filtering:
                if not self.is_edge_on_drug_disease_path(
                    src, dst, drug_distances, disease_distances,
                    n_hops, max_total_path_length
                ):
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

    # Run igraph filter
    logger.info("\n" + "="*60)
    logger.info("Running igraph Filter")
    logger.info("="*60)

    start_time = time.time()
    ig_filter = IGraphProximityFilter(train_numeric)

    ig_filtered, diagnostics = ig_filter.filter_for_test_triples(
        test_numeric,
        n_hops=args.n_hops,
        min_degree=args.min_degree,
        preserve_test_entity_edges=args.preserve_test_edges,
        path_filtering=args.path_filtering,
        max_total_path_length=args.max_total_path_length
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
