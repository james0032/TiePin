"""
Filter training triples based on proximity to test triples.

This module provides functions to filter training data by:
1. N-hop neighborhood from test triples
2. Degree filtering to remove low-connectivity edges
3. Special handling for edges directly containing test entities

This can dramatically reduce the number of training examples to consider
for TracIn analysis while focusing on the most relevant ones.
"""

import logging
from collections import defaultdict, deque
from typing import Dict, List, Set, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class ProximityFilter:
    """Filter training triples based on proximity to test triples."""

    def __init__(self, training_triples: np.ndarray, test_triples: np.ndarray):
        """Initialize proximity filter.

        Args:
            training_triples: Array of shape (N, 3) with [head, relation, tail]
            test_triples: Array of shape (M, 3) with [head, relation, tail]
        """
        self.training_triples = training_triples
        self.test_triples = test_triples

        # Build adjacency structures for efficient traversal
        self._build_adjacency()

        logger.info(f"Initialized filter with {len(training_triples)} training triples, "
                   f"{len(test_triples)} test triples")

    def _build_adjacency(self):
        """Build adjacency list and edge index for the training graph."""
        # Adjacency list: entity -> [(neighbor_entity, edge_idx), ...]
        self.entity_to_neighbors = defaultdict(list)

        # Edge to index mapping
        self.edge_to_idx = {}

        # Entity degree (number of edges connected to entity)
        self.entity_degree = defaultdict(int)

        for idx, (h, r, t) in enumerate(self.training_triples):
            h, r, t = int(h), int(r), int(t)

            # Store edge
            edge = (h, r, t)
            self.edge_to_idx[edge] = idx

            # Build adjacency (undirected for neighborhood purposes)
            self.entity_to_neighbors[h].append((t, idx))
            self.entity_to_neighbors[t].append((h, idx))

            # Update degrees
            self.entity_degree[h] += 1
            self.entity_degree[t] += 1

        logger.info(f"Built graph with {len(self.entity_to_neighbors)} entities")

    def get_n_hop_neighbors(self, seed_entities: Set[int], n_hops: int) -> Set[int]:
        """Get all entities within n hops from seed entities.

        Args:
            seed_entities: Set of starting entities
            n_hops: Number of hops to traverse

        Returns:
            Set of all entities within n hops
        """
        visited = set(seed_entities)
        current_level = set(seed_entities)

        for hop in range(n_hops):
            next_level = set()

            for entity in current_level:
                for neighbor, _ in self.entity_to_neighbors[entity]:
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)

            current_level = next_level

            if not current_level:
                break

        logger.info(f"Found {len(visited)} entities within {n_hops} hops from {len(seed_entities)} seeds")
        return visited

    def get_n_hop_edges(
        self,
        seed_entities: Set[int],
        n_hops: int,
        include_entity_edges: bool = True
    ) -> Set[int]:
        """Get all edge indices within n hops from seed entities.

        Args:
            seed_entities: Set of starting entities
            n_hops: Number of hops to traverse
            include_entity_edges: If True, include edges where one endpoint is in seeds

        Returns:
            Set of edge indices within n hops
        """
        # BFS to find all edges within n hops
        edge_indices = set()
        visited_entities = set(seed_entities)
        queue = deque([(entity, 0) for entity in seed_entities])

        while queue:
            entity, dist = queue.popleft()

            if dist >= n_hops:
                continue

            for neighbor, edge_idx in self.entity_to_neighbors[entity]:
                # Add edge
                edge_indices.add(edge_idx)

                # Continue BFS
                if neighbor not in visited_entities:
                    visited_entities.add(neighbor)
                    queue.append((neighbor, dist + 1))

        logger.info(f"Found {len(edge_indices)} edges within {n_hops} hops")
        return edge_indices

    def filter_by_n_hop_and_degree(
        self,
        n_hops: int = 2,
        min_degree: int = 2,
        preserve_test_entity_edges: bool = True
    ) -> np.ndarray:
        """Filter training triples by proximity and degree.

        Algorithm:
        1. Extract all entities from test triples
        2. Find all edges within n-hops from test entities
        3. Remove edges where both endpoints have degree 1 (after n-hop filtering)
        4. Exception: Keep edges that contain test head or tail entities

        Args:
            n_hops: Number of hops from test triples
            min_degree: Minimum degree threshold (typically 2)
            preserve_test_entity_edges: If True, always keep edges with test entities

        Returns:
            Filtered training triples array
        """
        logger.info("=" * 80)
        logger.info(f"Filtering with n_hops={n_hops}, min_degree={min_degree}")
        logger.info("=" * 80)

        # Step 1: Extract test entities
        test_entities = set()
        for h, r, t in self.test_triples:
            test_entities.add(int(h))
            test_entities.add(int(t))

        logger.info(f"Step 1: Extracted {len(test_entities)} unique test entities")

        # Step 2: Get n-hop neighborhood
        n_hop_entities = self.get_n_hop_neighbors(test_entities, n_hops)
        n_hop_edges = self.get_n_hop_edges(test_entities, n_hops)

        logger.info(f"Step 2: Found {len(n_hop_entities)} entities, {len(n_hop_edges)} edges in {n_hops}-hop neighborhood")

        # Step 3: Compute degrees within n-hop subgraph
        subgraph_degree = defaultdict(int)
        for edge_idx in n_hop_edges:
            h, r, t = self.training_triples[edge_idx]
            subgraph_degree[int(h)] += 1
            subgraph_degree[int(t)] += 1

        logger.info(f"Step 3: Computed degrees in subgraph")

        # Step 4: Filter by degree
        filtered_indices = []

        for edge_idx in n_hop_edges:
            h, r, t = self.training_triples[edge_idx]
            h, r, t = int(h), int(r), int(t)

            # Check if this edge should be preserved
            should_keep = False

            # Rule 1: If either endpoint is a test entity, always keep
            if preserve_test_entity_edges and (h in test_entities or t in test_entities):
                should_keep = True

            # Rule 2: If either endpoint has degree >= min_degree, keep
            elif subgraph_degree[h] >= min_degree or subgraph_degree[t] >= min_degree:
                should_keep = True

            if should_keep:
                filtered_indices.append(edge_idx)

        filtered_triples = self.training_triples[filtered_indices]

        logger.info(f"Step 4: After degree filtering ({min_degree}): {len(filtered_triples)} edges")
        logger.info(f"Reduction: {len(self.training_triples)} → {len(filtered_triples)} "
                   f"({len(filtered_triples)/len(self.training_triples)*100:.1f}%)")
        logger.info("=" * 80)

        return filtered_triples

    def get_statistics(self, filtered_triples: np.ndarray) -> Dict:
        """Get statistics about filtered triples.

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

        # Compute degree distribution
        degree_dist = defaultdict(int)
        entity_degree = defaultdict(int)

        for h, r, t in filtered_triples:
            entity_degree[int(h)] += 1
            entity_degree[int(t)] += 1

        for deg in entity_degree.values():
            degree_dist[deg] += 1

        stats = {
            'num_triples': len(filtered_triples),
            'num_entities': len(entities),
            'num_relations': len(relations),
            'avg_degree': np.mean(list(entity_degree.values())) if entity_degree else 0,
            'max_degree': max(entity_degree.values()) if entity_degree else 0,
            'min_degree': min(entity_degree.values()) if entity_degree else 0,
            'degree_distribution': dict(degree_dist)
        }

        return stats


def filter_training_file(
    train_path: str,
    test_path: str,
    output_path: str,
    n_hops: int = 2,
    min_degree: int = 2,
    preserve_test_entity_edges: bool = True
):
    """Filter training file based on proximity to test triples.

    Args:
        train_path: Path to training triples file
        test_path: Path to test triples file
        output_path: Path to save filtered training triples
        n_hops: Number of hops from test triples
        min_degree: Minimum degree threshold
        preserve_test_entity_edges: Keep edges with test entities
    """
    logger.info(f"Loading training triples from {train_path}")
    train_triples = []
    with open(train_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                train_triples.append(parts[:3])

    logger.info(f"Loading test triples from {test_path}")
    test_triples = []
    with open(test_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                test_triples.append(parts[:3])

    logger.info(f"Loaded {len(train_triples)} training, {len(test_triples)} test triples")

    # Convert to entity/relation indices (string labels)
    # Note: This works with string labels, no need for numeric conversion
    train_array = np.array(train_triples)
    test_array = np.array(test_triples)

    # For the graph operations, we need numeric indices
    # Create temporary mappings
    entity_to_idx = {}
    idx_to_entity = {}
    relation_to_idx = {}

    current_entity_idx = 0
    current_relation_idx = 0

    # Map entities and relations
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

    # Convert to numeric arrays
    train_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in train_triples
    ])

    test_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in test_triples
    ])

    # Create filter and apply
    filter_obj = ProximityFilter(train_numeric, test_numeric)

    filtered_numeric = filter_obj.filter_by_n_hop_and_degree(
        n_hops=n_hops,
        min_degree=min_degree,
        preserve_test_entity_edges=preserve_test_entity_edges
    )

    # Convert back to string labels
    filtered_triples = []
    for h_idx, r_idx, t_idx in filtered_numeric:
        h = idx_to_entity[h_idx]
        r = list(relation_to_idx.keys())[list(relation_to_idx.values()).index(r_idx)]
        t = idx_to_entity[t_idx]
        filtered_triples.append([h, r, t])

    # Write filtered triples
    logger.info(f"Writing {len(filtered_triples)} filtered triples to {output_path}")
    with open(output_path, 'w') as f:
        for h, r, t in filtered_triples:
            f.write(f"{h}\t{r}\t{t}\n")

    # Print statistics
    stats = filter_obj.get_statistics(filtered_numeric)

    logger.info("\nFiltering Statistics:")
    logger.info(f"  Original training triples: {len(train_triples)}")
    logger.info(f"  Filtered training triples: {stats['num_triples']}")
    logger.info(f"  Reduction: {(1 - stats['num_triples']/len(train_triples))*100:.1f}%")
    logger.info(f"  Entities in filtered graph: {stats['num_entities']}")
    logger.info(f"  Relations in filtered graph: {stats['num_relations']}")
    logger.info(f"  Average degree: {stats['avg_degree']:.2f}")
    logger.info(f"  Degree range: [{stats['min_degree']}, {stats['max_degree']}]")

    logger.info(f"\nFiltered training saved to: {output_path}")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Filter training triples by proximity to test triples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 2-hop neighborhood, remove degree-1 edges
  python filter_training_by_proximity.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 2 \\
      --min-degree 2

  # 3-hop neighborhood, more aggressive filtering
  python filter_training_by_proximity.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 3 \\
      --min-degree 3

  # Keep all edges touching test entities
  python filter_training_by_proximity.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 2 \\
      --preserve-test-edges
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
    parser.add_argument('--preserve-test-edges', action='store_true',
                        help='Always preserve edges containing test entities')

    args = parser.parse_args()

    filter_training_file(
        train_path=args.train,
        test_path=args.test,
        output_path=args.output,
        n_hops=args.n_hops,
        min_degree=args.min_degree,
        preserve_test_entity_edges=args.preserve_test_edges
    )
