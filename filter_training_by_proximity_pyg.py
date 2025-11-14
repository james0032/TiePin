"""
Filter training triples based on proximity to test triples using PyTorch Geometric.

This module uses PyG's efficient graph data structures and utilities for:
1. N-hop neighborhood extraction
2. Subgraph extraction
3. Degree-based filtering

Much faster and more memory-efficient than custom implementation.
"""

import logging
from typing import Dict, List, Set, Tuple, Optional
from pathlib import Path
import pickle
import hashlib
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, degree, to_undirected
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


class PerformanceTracker:
    """Track performance metrics for different operations."""

    def __init__(self):
        self.timings = {}
        self.start_times = {}

    def start(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()

    def end(self, operation: str):
        """End timing an operation."""
        if operation in self.start_times:
            elapsed = time.time() - self.start_times[operation]
            self.timings[operation] = elapsed
            del self.start_times[operation]
            return elapsed
        return 0

    def report(self):
        """Print performance report."""
        logger.info("\n" + "="*60)
        logger.info("PERFORMANCE BREAKDOWN")
        logger.info("="*60)

        total_time = sum(self.timings.values())

        # Sort by time (descending)
        sorted_timings = sorted(self.timings.items(), key=lambda x: x[1], reverse=True)

        for operation, elapsed in sorted_timings:
            percentage = (elapsed / total_time * 100) if total_time > 0 else 0
            logger.info(f"  {operation:40s}: {elapsed:8.2f}s ({percentage:5.1f}%)")

        logger.info("-"*60)
        logger.info(f"  {'TOTAL':40s}: {total_time:8.2f}s (100.0%)")
        logger.info("="*60 + "\n")


class ProximityFilterPyG:
    """Filter training triples using PyTorch Geometric.

    This implementation uses PyG's optimized graph operations for
    fast neighborhood extraction and filtering.

    Supports caching: Save/load the graph object to avoid rebuilding.
    """

    def __init__(self, training_triples: np.ndarray, cache_path: Optional[str] = None):
        """Initialize proximity filter with training data.

        Args:
            training_triples: Array of shape (N, 3) with [head, relation, tail]
            cache_path: Optional path to cache the graph object. If provided:
                       - Loads from cache if exists and matches training data
                       - Saves to cache after building if doesn't exist
        """
        self.training_triples = training_triples
        self.cache_path = cache_path

        # Try to load from cache
        cache_loaded = False
        if cache_path:
            logger.info(f"Attempting to load graph from cache: {cache_path}")
            cache_loaded = self._load_from_cache()

        if cache_loaded:
            logger.info(f"✓ Successfully loaded graph from cache: {cache_path}")
        else:
            if cache_path:
                logger.info(f"Cache not available, building graph from scratch...")
            # Build PyG graph from scratch
            self._build_pyg_graph()
            logger.info(f"✓ Built new graph with {len(training_triples)} training triples")

            # Save to cache if requested
            if cache_path:
                self._save_to_cache()
                logger.info(f"✓ Saved graph to cache: {cache_path}")

    def _build_pyg_graph(self):
        """Build PyTorch Geometric graph from training triples."""
        # Extract edges (undirected for neighborhood purposes)
        heads = torch.LongTensor(self.training_triples[:, 0])
        tails = torch.LongTensor(self.training_triples[:, 2])

        # Create edge index [2, num_edges] - undirected
        edge_index = torch.stack([heads, tails], dim=0)

        # Make undirected (add reverse edges)
        self.edge_index = to_undirected(edge_index)

        # Store relation info separately (PyG edge_index is just source/target)
        self.edge_relations = torch.LongTensor(self.training_triples[:, 1])

        # Compute node degrees
        num_nodes = max(self.edge_index.max().item() + 1,
                       self.training_triples[:, [0, 2]].max() + 1)
        self.node_degrees = degree(self.edge_index[0], num_nodes=num_nodes, dtype=torch.long)

        # Create mapping from edge to triple index
        self._create_edge_to_triple_mapping()

        logger.info(f"Built PyG graph with {num_nodes} nodes, {self.edge_index.shape[1]} edges")

    def _create_edge_to_triple_mapping(self):
        """Create mapping from (h, t) edge to original triple indices."""
        self.edge_to_triples = {}

        for idx, (h, r, t) in enumerate(self.training_triples):
            h, t = int(h), int(t)
            # Store both directions since graph is undirected
            if (h, t) not in self.edge_to_triples:
                self.edge_to_triples[(h, t)] = []
            if (t, h) not in self.edge_to_triples:
                self.edge_to_triples[(t, h)] = []

            self.edge_to_triples[(h, t)].append(idx)
            self.edge_to_triples[(t, h)].append(idx)

    def _compute_data_hash(self) -> str:
        """Compute hash of training data for cache validation.

        Returns:
            MD5 hash string of the training triples
        """
        # Create hash from training triples content
        data_bytes = self.training_triples.tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def _save_to_cache(self):
        """Save graph object to cache file."""
        try:
            cache_dir = Path(self.cache_path).parent
            cache_dir.mkdir(parents=True, exist_ok=True)

            cache_data = {
                'edge_index': self.edge_index,
                'edge_relations': self.edge_relations,
                'node_degrees': self.node_degrees,
                'edge_to_triples': self.edge_to_triples,
                'data_hash': self._compute_data_hash(),
                'num_triples': len(self.training_triples)
            }

            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            logger.info(f"Graph cached successfully ({len(self.training_triples)} triples)")

        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _load_from_cache(self) -> bool:
        """Load graph object from cache file.

        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            cache_path_obj = Path(self.cache_path)
            if not cache_path_obj.exists():
                logger.info(f"  Cache file does not exist: {self.cache_path}")
                return False

            logger.info(f"  Cache file found, loading...")
            file_size_mb = cache_path_obj.stat().st_size / (1024 * 1024)
            logger.info(f"  Cache file size: {file_size_mb:.1f} MB")

            with open(self.cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            logger.info(f"  Cache contains {cache_data.get('num_triples', 'unknown')} triples")

            # Validate cache matches current data
            current_hash = self._compute_data_hash()
            cached_hash = cache_data.get('data_hash', '')

            logger.info(f"  Current data hash: {current_hash[:8]}...")
            logger.info(f"  Cached data hash:  {cached_hash[:8]}...")

            if current_hash != cached_hash:
                logger.warning(f"  ✗ Cache invalidated: training data hash mismatch")
                logger.warning(f"     Current training triples: {len(self.training_triples)}")
                logger.warning(f"     Cached training triples:  {cache_data.get('num_triples', 'unknown')}")
                return False

            # Load cached data
            self.edge_index = cache_data['edge_index']
            self.edge_relations = cache_data['edge_relations']
            self.node_degrees = cache_data['node_degrees']
            self.edge_to_triples = cache_data['edge_to_triples']

            logger.info(f"  ✓ Cache validated and loaded ({cache_data['num_triples']} triples)")
            return True

        except Exception as e:
            logger.warning(f"  ✗ Failed to load cache: {type(e).__name__}: {e}")
            return False

    @classmethod
    def from_cache_or_build(
        cls,
        training_triples: np.ndarray,
        cache_path: str
    ) -> 'ProximityFilterPyG':
        """Factory method to load from cache or build new graph.

        Args:
            training_triples: Training triples array
            cache_path: Path to cache file

        Returns:
            ProximityFilterPyG instance
        """
        return cls(training_triples, cache_path=cache_path)

    def _compute_hop_distances(self, test_entities: List[int], max_hops: int) -> torch.Tensor:
        """Compute shortest hop distance from test entities to all nodes.

        Args:
            test_entities: List of test entity node IDs
            max_hops: Maximum number of hops to compute

        Returns:
            Tensor of shape (num_nodes,) with hop distances.
            Distance is -1 for unreachable nodes.
        """
        num_nodes = len(self.node_degrees)
        distances = torch.full((num_nodes,), -1, dtype=torch.long)

        # Initialize test entities with distance 0
        current_layer = set(test_entities)
        for node in current_layer:
            distances[node] = 0

        # BFS to compute distances
        for hop in range(1, max_hops + 1):
            next_layer = set()

            for node in current_layer:
                # Find all neighbors
                neighbors = self.edge_index[1][self.edge_index[0] == node]
                for neighbor in neighbors:
                    neighbor = neighbor.item()
                    if distances[neighbor] == -1:  # Not visited yet
                        distances[neighbor] = hop
                        next_layer.add(neighbor)

            if len(next_layer) == 0:
                break

            current_layer = next_layer

        return distances

    def filter_for_single_test_triple(
        self,
        test_triple: Tuple[int, int, int],
        n_hops: int = 2,
        min_degree: int = 2,
        preserve_test_entity_edges: bool = True,
        strict_hop_constraint: bool = False
    ) -> np.ndarray:
        """Filter training triples for a single test triple using PyG.

        Args:
            test_triple: Single test triple (head, relation, tail)
            n_hops: Number of hops from test entities
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: If True, always keep edges with test entities
            strict_hop_constraint: If True, enforce that BOTH endpoints of each edge
                                  are within n_hops (prevents distant shortcuts)

        Returns:
            Filtered training triples array
        """
        test_h, test_r, test_t = test_triple
        test_h, test_t = int(test_h), int(test_t)

        logger.info(f"Filtering for test triple: ({test_h}, {test_r}, {test_t})")
        logger.info(f"Parameters: n_hops={n_hops}, min_degree={min_degree}, "
                   f"strict_hop_constraint={strict_hop_constraint}")

        # Get n-hop neighborhoods from head (drug) and tail (disease) separately
        logger.info("Computing n-hop neighborhood from head entity (drug)...")
        head_nodes, head_edge_index, _, _ = k_hop_subgraph(
            node_idx=[test_h],
            num_hops=n_hops,
            edge_index=self.edge_index,
            relabel_nodes=False,
            num_nodes=len(self.node_degrees)
        )

        logger.info("Computing n-hop neighborhood from tail entity (disease)...")
        tail_nodes, tail_edge_index, _, _ = k_hop_subgraph(
            node_idx=[test_t],
            num_hops=n_hops,
            edge_index=self.edge_index,
            relabel_nodes=False,
            num_nodes=len(self.node_degrees)
        )

        # INTERSECTION: Keep only nodes reachable from BOTH head and tail
        head_nodes_set = set(head_nodes.tolist())
        tail_nodes_set = set(tail_nodes.tolist())
        intersect_nodes_set = head_nodes_set & tail_nodes_set

        # IMPORTANT: Always include the test entities themselves in the intersection
        # This ensures edges connected to test entities are preserved
        intersect_nodes_set.add(test_h)
        intersect_nodes_set.add(test_t)

        logger.info(f"Found {len(head_nodes_set)} nodes in {n_hops}-hop neighborhood of head (drug)")
        logger.info(f"Found {len(tail_nodes_set)} nodes in {n_hops}-hop neighborhood of tail (disease)")
        logger.info(f"Intersection: {len(intersect_nodes_set)} nodes reachable from BOTH head and tail (including test entities)")

        # Use the union of edge indices for degree computation, but filter by intersection
        # Combine both edge sets for subgraph
        combined_edge_index = torch.cat([head_edge_index, tail_edge_index], dim=1)
        # Remove duplicates
        combined_edge_index = torch.unique(combined_edge_index, dim=1)
        subset_edge_index = combined_edge_index

        # Compute hop distances if strict mode is enabled
        if strict_hop_constraint:
            hop_distances = self._compute_hop_distances([test_h, test_t], n_hops)
            logger.info(f"Computed hop distances in strict mode")

        # Compute degrees in subgraph
        subgraph_degrees = degree(subset_edge_index[0],
                                  num_nodes=len(self.node_degrees),
                                  dtype=torch.long)

        # Filter edges by degree AND intersection constraint
        filtered_triple_indices = set()

        # Iterate over edges in subgraph
        for i in range(subset_edge_index.shape[1]):
            src = subset_edge_index[0, i].item()
            dst = subset_edge_index[1, i].item()

            # Skip if this is a reverse edge we've already processed
            if src > dst:
                continue

            # INTERSECTION CONSTRAINT: Both endpoints must be in intersection set
            # This ensures edges are on potential paths between drug and disease
            if src not in intersect_nodes_set or dst not in intersect_nodes_set:
                continue

            # Check filtering rules
            should_keep = False

            # Rule 1: Preserve edges with test entities (evaluated first)
            if preserve_test_entity_edges:
                if src == test_h or src == test_t or dst == test_h or dst == test_t:
                    should_keep = True

            # Rule 2: Check degree threshold
            if not should_keep:
                src_degree = subgraph_degrees[src].item()
                dst_degree = subgraph_degrees[dst].item()

                if src_degree >= min_degree or dst_degree >= min_degree:
                    should_keep = True

            # STRICT HOP CONSTRAINT: Apply after degree/preserve rules
            # This ensures that even preserved edges must meet hop constraint
            if should_keep and strict_hop_constraint:
                src_dist = hop_distances[src].item()
                dst_dist = hop_distances[dst].item()

                # Skip if either endpoint is beyond n_hops or unreachable
                if src_dist < 0 or src_dist > n_hops or dst_dist < 0 or dst_dist > n_hops:
                    should_keep = False

            # Add corresponding triple indices
            if should_keep:
                # Find original triples for this edge
                if (src, dst) in self.edge_to_triples:
                    filtered_triple_indices.update(self.edge_to_triples[(src, dst)])
                if (dst, src) in self.edge_to_triples:
                    filtered_triple_indices.update(self.edge_to_triples[(dst, src)])

        # Convert to array and sort by original index
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
        strict_hop_constraint: bool = False
    ) -> np.ndarray:
        """Filter training triples for multiple test triples using PyG.

        More efficient than filtering for each test triple individually.

        Args:
            test_triples: Array of shape (M, 3) with [head, relation, tail]
            n_hops: Number of hops from test entities
            min_degree: Minimum degree threshold
            preserve_test_entity_edges: If True, always keep edges with test entities
            strict_hop_constraint: If True, enforce that BOTH endpoints of each edge
                                  are within n_hops (prevents distant shortcuts)

        Returns:
            Filtered training triples array
        """
        logger.info(f"Filtering for {len(test_triples)} test triples")
        logger.info(f"Parameters: n_hops={n_hops}, min_degree={min_degree}, "
                   f"strict_hop_constraint={strict_hop_constraint}")

        # Extract test entities - separate heads (drugs) and tails (diseases)
        head_entities = set()
        tail_entities = set()
        for h, r, t in test_triples:
            head_entities.add(int(h))
            tail_entities.add(int(t))

        head_entity_list = list(head_entities)
        tail_entity_list = list(tail_entities)
        logger.info(f"Test head entities (drugs): {len(head_entity_list)}")
        logger.info(f"Test tail entities (diseases): {len(tail_entity_list)}")

        # Get n-hop neighborhoods from heads (drugs) and tails (diseases) separately
        logger.info("Computing n-hop neighborhood from head entities (drugs)...")
        head_nodes, head_edge_index, _, _ = k_hop_subgraph(
            node_idx=head_entity_list,
            num_hops=n_hops,
            edge_index=self.edge_index,
            relabel_nodes=False,
            num_nodes=len(self.node_degrees)
        )

        logger.info("Computing n-hop neighborhood from tail entities (diseases)...")
        tail_nodes, tail_edge_index, _, _ = k_hop_subgraph(
            node_idx=tail_entity_list,
            num_hops=n_hops,
            edge_index=self.edge_index,
            relabel_nodes=False,
            num_nodes=len(self.node_degrees)
        )

        # INTERSECTION: Keep only nodes reachable from BOTH heads and tails
        head_nodes_set = set(head_nodes.tolist())
        tail_nodes_set = set(tail_nodes.tolist())
        intersect_nodes_set = head_nodes_set & tail_nodes_set

        # IMPORTANT: Always include all test entities in the intersection
        # This ensures edges connected to test entities are preserved
        intersect_nodes_set.update(head_entities)
        intersect_nodes_set.update(tail_entities)

        logger.info(f"Found {len(head_nodes_set)} nodes in {n_hops}-hop neighborhood of heads (drugs)")
        logger.info(f"Found {len(tail_nodes_set)} nodes in {n_hops}-hop neighborhood of tails (diseases)")
        logger.info(f"Intersection: {len(intersect_nodes_set)} nodes reachable from BOTH heads and tails (including test entities)")

        # Use the union of edge indices for degree computation, but filter by intersection
        # Combine both edge sets for subgraph
        combined_edge_index = torch.cat([head_edge_index, tail_edge_index], dim=1)
        # Remove duplicates
        combined_edge_index = torch.unique(combined_edge_index, dim=1)
        subset_edge_index = combined_edge_index

        # For backward compatibility, also compute test_entities set
        test_entities = head_entities | tail_entities

        # Compute hop distances if strict mode is enabled
        if strict_hop_constraint:
            hop_distances = self._compute_hop_distances(list(test_entities), n_hops)
            logger.info(f"Computed hop distances in strict mode")

        # Compute degrees in subgraph
        subgraph_degrees = degree(subset_edge_index[0],
                                  num_nodes=len(self.node_degrees),
                                  dtype=torch.long)

        # Filter edges by degree AND intersection constraint
        filtered_triple_indices = set()

        for i in range(subset_edge_index.shape[1]):
            src = subset_edge_index[0, i].item()
            dst = subset_edge_index[1, i].item()

            # Skip reverse edges
            if src > dst:
                continue

            # INTERSECTION CONSTRAINT: Both endpoints must be in intersection set
            # This ensures edges are on potential paths between drugs and diseases
            if src not in intersect_nodes_set or dst not in intersect_nodes_set:
                continue

            should_keep = False

            # Rule 1: Preserve edges with test entities
            if preserve_test_entity_edges:
                if src in test_entities or dst in test_entities:
                    should_keep = True

            # Rule 2: Check degree threshold
            if not should_keep:
                src_degree = subgraph_degrees[src].item()
                dst_degree = subgraph_degrees[dst].item()

                if src_degree >= min_degree or dst_degree >= min_degree:
                    should_keep = True

            # STRICT HOP CONSTRAINT: Apply after degree/preserve rules
            # This ensures that even preserved edges must meet hop constraint
            if should_keep and strict_hop_constraint:
                src_dist = hop_distances[src].item()
                dst_dist = hop_distances[dst].item()

                # Skip if either endpoint is beyond n_hops or unreachable
                if src_dist < 0 or src_dist > n_hops or dst_dist < 0 or dst_dist > n_hops:
                    should_keep = False

            if should_keep:
                if (src, dst) in self.edge_to_triples:
                    filtered_triple_indices.update(self.edge_to_triples[(src, dst)])
                if (dst, src) in self.edge_to_triples:
                    filtered_triple_indices.update(self.edge_to_triples[(dst, src)])

        filtered_indices = sorted(list(filtered_triple_indices))
        filtered_triples = self.training_triples[filtered_indices]

        reduction_pct = (1 - len(filtered_triples) / len(self.training_triples)) * 100
        logger.info(f"Filtered: {len(self.training_triples)} → {len(filtered_triples)} "
                   f"({reduction_pct:.1f}% reduction)")

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

        # Compute degree statistics for filtered subgraph
        filtered_heads = torch.LongTensor(filtered_triples[:, 0])
        filtered_tails = torch.LongTensor(filtered_triples[:, 2])
        filtered_edge_index = torch.stack([filtered_heads, filtered_tails], dim=0)
        filtered_edge_index = to_undirected(filtered_edge_index)

        num_nodes = max(filtered_edge_index.max().item() + 1, len(self.node_degrees))
        filtered_degrees = degree(filtered_edge_index[0], num_nodes=num_nodes, dtype=torch.long)

        # Get non-zero degrees
        nonzero_degrees = filtered_degrees[filtered_degrees > 0]

        stats = {
            'num_triples': len(filtered_triples),
            'num_entities': len(entities),
            'num_relations': len(relations),
            'avg_degree': nonzero_degrees.float().mean().item() if len(nonzero_degrees) > 0 else 0,
            'max_degree': nonzero_degrees.max().item() if len(nonzero_degrees) > 0 else 0,
            'min_degree': nonzero_degrees.min().item() if len(nonzero_degrees) > 0 else 0,
        }

        return stats


def filter_training_file(
    train_path: str,
    test_path: str,
    output_path: str,
    n_hops: int = 2,
    min_degree: int = 2,
    preserve_test_entity_edges: bool = True,
    use_single_triple_mode: bool = False,
    cache_path: Optional[str] = None,
    strict_hop_constraint: bool = False
):
    """Filter training file based on proximity to test triples using PyG.

    Args:
        train_path: Path to training triples file
        test_path: Path to test triples file
        output_path: Path to save filtered training triples
        n_hops: Number of hops from test triples
        min_degree: Minimum degree threshold
        preserve_test_entity_edges: Keep edges with test entities
        use_single_triple_mode: If True, analyze first test triple only
        cache_path: Optional path to cache the graph object for reuse
        strict_hop_constraint: If True, enforce strict n-hop constraint
    """
    perf = PerformanceTracker()
    perf.start("TOTAL_EXECUTION")

    # Step 1: Load training triples
    perf.start("1. Load training triples from disk")
    logger.info(f"Loading training triples from {train_path}")
    train_triples = []
    with open(train_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                train_triples.append(parts[:3])
    perf.end("1. Load training triples from disk")

    # Step 2: Load test triples
    perf.start("2. Load test triples from disk")
    logger.info(f"Loading test triples from {test_path}")
    test_triples = []
    with open(test_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                test_triples.append(parts[:3])
    logger.info(f"Loaded {len(train_triples)} training, {len(test_triples)} test triples")
    perf.end("2. Load test triples from disk")

    # Step 3: Create entity/relation mappings
    perf.start("3. Build entity/relation ID mappings")
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

    logger.info(f"Created mappings: {len(entity_to_idx)} entities, {len(relation_to_idx)} relations")
    perf.end("3. Build entity/relation ID mappings")

    # Step 4: Convert to numeric arrays
    perf.start("4. Convert triples to numeric format")
    train_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in train_triples
    ])

    test_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in test_triples
    ])
    perf.end("4. Convert triples to numeric format")

    # Step 5: Create PyG filter (load cache or build graph)
    perf.start("5. Initialize ProximityFilterPyG (cache/build)")
    filter_obj = ProximityFilterPyG(train_numeric, cache_path=cache_path)
    perf.end("5. Initialize ProximityFilterPyG (cache/build)")

    # Step 6: Filter triples
    perf.start("6. Filter triples (k-hop subgraph + degree)")
    if use_single_triple_mode and len(test_numeric) > 0:
        logger.info("Using single test triple mode (first triple only)")
        test_triple = tuple(test_numeric[0])
        filtered_numeric = filter_obj.filter_for_single_test_triple(
            test_triple=test_triple,
            n_hops=n_hops,
            min_degree=min_degree,
            preserve_test_entity_edges=preserve_test_entity_edges,
            strict_hop_constraint=strict_hop_constraint
        )
    else:
        logger.info("Using multiple test triples mode")
        filtered_numeric = filter_obj.filter_for_multiple_test_triples(
            test_triples=test_numeric,
            n_hops=n_hops,
            min_degree=min_degree,
            preserve_test_entity_edges=preserve_test_entity_edges,
            strict_hop_constraint=strict_hop_constraint
        )
    perf.end("6. Filter triples (k-hop subgraph + degree)")

    # Step 7: Convert back to string labels
    perf.start("7. Convert filtered triples back to strings")
    # Create inverse relation mapping for faster lookup
    idx_to_relation = {v: k for k, v in relation_to_idx.items()}

    filtered_triples = []
    for h_idx, r_idx, t_idx in filtered_numeric:
        h = idx_to_entity[h_idx]
        r = idx_to_relation[r_idx]
        t = idx_to_entity[t_idx]
        filtered_triples.append([h, r, t])
    perf.end("7. Convert filtered triples back to strings")

    # Step 8: Write filtered triples
    perf.start("8. Write filtered triples to disk")
    logger.info(f"Writing {len(filtered_triples)} filtered triples to {output_path}")

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for h, r, t in filtered_triples:
            f.write(f"{h}\t{r}\t{t}\n")
    perf.end("8. Write filtered triples to disk")

    # Step 9: Compute statistics
    perf.start("9. Compute statistics")
    stats = filter_obj.get_statistics(filtered_numeric)
    perf.end("9. Compute statistics")

    perf.end("TOTAL_EXECUTION")

    # Print statistics
    logger.info("\nFiltering Statistics:")
    logger.info(f"  Original training triples: {len(train_triples)}")
    logger.info(f"  Filtered training triples: {stats['num_triples']}")
    logger.info(f"  Reduction: {(1 - stats['num_triples']/len(train_triples))*100:.1f}%")
    logger.info(f"  Entities in filtered graph: {stats['num_entities']}")
    logger.info(f"  Relations in filtered graph: {stats['num_relations']}")
    logger.info(f"  Average degree: {stats['avg_degree']:.2f}")
    logger.info(f"  Degree range: [{stats['min_degree']}, {stats['max_degree']}]")

    logger.info(f"\nFiltered training saved to: {output_path}")

    # Print performance report
    perf.report()


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description='Filter training triples using PyTorch Geometric (faster!)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard filtering
  python filter_training_by_proximity_pyg.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 2 \\
      --min-degree 2

  # Single test triple mode
  python filter_training_by_proximity_pyg.py \\
      --train train.txt \\
      --test test.txt \\
      --output train_filtered.txt \\
      --n-hops 2 \\
      --single-triple

Benefits of PyG version:
  - 5-10x faster than custom implementation
  - Lower memory usage
  - Optimized C++ backend
  - Industry-standard library
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
    # Mutually exclusive group for preserve-test-edges
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
    parser.add_argument('--cache', type=str,
                        help='Path to cache the graph object (speeds up repeated runs)')
    parser.add_argument('--strict-hop-constraint', action='store_true',
                        help='Enforce strict n-hop constraint: both endpoints of each edge '
                             'must be within n_hops (prevents distant shortcuts)')
    parser.add_argument('--profile', action='store_true',
                        help='Enable detailed profiling with cProfile (outputs to profile.stats)')

    args = parser.parse_args()

    if args.profile:
        import cProfile
        import pstats

        logger.info("=" * 60)
        logger.info("PROFILING ENABLED - Running with cProfile")
        logger.info("=" * 60)

        profiler = cProfile.Profile()
        profiler.enable()

        filter_training_file(
            train_path=args.train,
            test_path=args.test,
            output_path=args.output,
            n_hops=args.n_hops,
            min_degree=args.min_degree,
            preserve_test_entity_edges=args.preserve_test_edges,
            use_single_triple_mode=args.single_triple,
            cache_path=args.cache,
            strict_hop_constraint=args.strict_hop_constraint
        )

        profiler.disable()

        # Save detailed stats
        profiler.dump_stats('profile.stats')
        logger.info("\n" + "=" * 60)
        logger.info("Profiling results saved to: profile.stats")
        logger.info("To view: python -m pstats profile.stats")
        logger.info("=" * 60)

        # Print top 30 time-consuming functions
        logger.info("\nTop 30 functions by cumulative time:")
        logger.info("-" * 60)
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(30)

    else:
        filter_training_file(
            train_path=args.train,
            test_path=args.test,
            output_path=args.output,
            n_hops=args.n_hops,
            min_degree=args.min_degree,
            preserve_test_entity_edges=args.preserve_test_edges,
            use_single_triple_mode=args.single_triple,
            cache_path=args.cache,
            strict_hop_constraint=args.strict_hop_constraint
        )
