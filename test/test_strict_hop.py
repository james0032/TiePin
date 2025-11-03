#!/usr/bin/env python3
"""
Test case to verify strict hop constraint works correctly.

This creates a simple graph structure where we can verify that:
1. Without strict mode: edges can create paths longer than n_hops
2. With strict mode: all edges have both endpoints within n_hops
"""

import numpy as np
import logging
from filter_training_by_proximity_pyg import ProximityFilterPyG

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def create_test_graph():
    """
    Create a test graph that demonstrates the difference between strict and non-strict modes.

    The key is to create a densely connected subgraph where all nodes are within
    n_hops, but we want to filter by degree. In non-strict mode with the old
    implementation, the degree filtering happens on the subgraph, which can
    include edges between nodes that are both far from test entities.

    Graph structure:

        Test: 0 --- 1 --- 2 --- 3
                    |     |     |
                    4 --- 5 --- 6

    With n_hops=1 from test entities [0, 1]:
    - Nodes 0, 1: 0 hops (test entities)
    - Nodes 2, 4: 1 hop
    - Nodes 3, 5, 6: 2 hops

    The OLD non-strict behavior:
    - k_hop_subgraph(n_hops=1) returns nodes: {0, 1, 2, 4}
    - But these nodes might have high degree
    - So ALL edges between them are kept, including (2, 4)
    - This is correct! Both 2 and 4 are within 1 hop.

    To see a real difference, we need to use min_degree filtering.
    Actually, let me reconsider the test...

    The issue the user reported is that with n_hops=2, they see paths
    that are 5-6 hops long in the FILTERED graph. This happens because:
    - Node A is 2 hops from test
    - Node B is also 2 hops from test (different direction)
    - Edge (A, B) is included
    - This creates a path: test -> ... -> A -> B -> ... -> test (4+ hops)

    Let's create that scenario:
    """

    edges = [
        # Path 1 from node 0: 0 - 1 - 2
        (0, 0, 1),
        (1, 0, 2),

        # Path 2 from node 0: 0 - 10 - 20
        (0, 0, 10),
        (10, 0, 20),

        # Shortcut between nodes that are both 2 hops away
        # This edge creates a 4-hop path: 0 -> 1 -> 2 -> 20 -> 10 -> 0
        (2, 0, 20),
    ]

    # Add reverse edges
    reverse_edges = [(t, r, h) for h, r, t in edges]
    all_edges = edges + reverse_edges

    training_triples = np.array(all_edges)

    return training_triples


def analyze_hop_distances(filtered_triples, filter_obj, test_entities, n_hops):
    """Analyze hop distances in filtered graph from ORIGINAL graph perspective.

    The key insight: we need to check hop distances from test entities in the
    ORIGINAL graph, not recompute them in the filtered graph.
    """

    # Compute hop distances in original graph
    hop_distances = filter_obj._compute_hop_distances(test_entities, max_hops=10)

    # Find max distance per edge
    max_edge_distance = 0
    edges_by_max_dist = {}

    for h, r, t in filtered_triples:
        h, t = int(h), int(t)
        h_dist = hop_distances[h].item()
        t_dist = hop_distances[t].item()

        # Maximum distance of either endpoint
        edge_max_dist = max(h_dist, t_dist)
        if edge_max_dist > max_edge_distance:
            max_edge_distance = edge_max_dist

        if edge_max_dist not in edges_by_max_dist:
            edges_by_max_dist[edge_max_dist] = []
        edges_by_max_dist[edge_max_dist].append((h, t, h_dist, t_dist))

    # Display results
    logger.info(f"\n  Edges grouped by maximum endpoint distance:")
    for dist in sorted(edges_by_max_dist.keys()):
        count = len(edges_by_max_dist[dist])
        logger.info(f"    {dist} hops: {count} edges")
        if dist > n_hops:  # Show problematic edges
            for h, t, h_dist, t_dist in edges_by_max_dist[dist][:3]:
                logger.info(f"      ({h}, {t}): distances=({h_dist}, {t_dist})")

    logger.info(f"\n  Maximum hop distance of any edge endpoint: {max_edge_distance}")

    return max_edge_distance


def test_strict_vs_non_strict():
    """Test the difference between strict and non-strict hop constraints."""

    logger.info("=" * 70)
    logger.info("Testing Strict vs Non-Strict Hop Constraint")
    logger.info("=" * 70)

    # Create test graph
    training_triples = create_test_graph()
    logger.info(f"\nCreated test graph with {len(training_triples)} edges")

    # Test triple: we'll use node 0 as the test entity
    test_triple = (0, 0, 1)  # An edge in the graph

    # Create filter object
    filter_obj = ProximityFilterPyG(training_triples)

    # Test with n_hops=2
    n_hops = 2
    min_degree = 1  # Low threshold to allow more edges

    logger.info("\n" + "=" * 70)
    logger.info(f"TEST 1: NON-STRICT mode (n_hops={n_hops}, min_degree={min_degree})")
    logger.info("=" * 70)

    filtered_non_strict = filter_obj.filter_for_single_test_triple(
        test_triple=test_triple,
        n_hops=n_hops,
        min_degree=min_degree,
        preserve_test_entity_edges=True,
        strict_hop_constraint=False
    )

    logger.info(f"\nFiltered to {len(filtered_non_strict)} edges")
    max_dist_non_strict = analyze_hop_distances(filtered_non_strict, filter_obj, [0, 1], n_hops)

    logger.info("\n" + "=" * 70)
    logger.info(f"TEST 2: STRICT mode (n_hops={n_hops}, min_degree={min_degree})")
    logger.info("=" * 70)

    filtered_strict = filter_obj.filter_for_single_test_triple(
        test_triple=test_triple,
        n_hops=n_hops,
        min_degree=min_degree,
        preserve_test_entity_edges=True,
        strict_hop_constraint=True
    )

    logger.info(f"\nFiltered to {len(filtered_strict)} edges")
    max_dist_strict = analyze_hop_distances(filtered_strict, filter_obj, [0, 1], n_hops)

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Original graph: {len(training_triples)} edges")
    logger.info(f"Non-strict mode: {len(filtered_non_strict)} edges, "
               f"max endpoint distance: {max_dist_non_strict} hops")
    logger.info(f"Strict mode:     {len(filtered_strict)} edges, "
               f"max endpoint distance: {max_dist_strict} hops")
    logger.info(f"\nWith n_hops={n_hops}:")
    logger.info(f"  Non-strict: {'PASS' if max_dist_non_strict <= n_hops else 'FAIL (allows distant edges)'}")
    logger.info(f"  Strict:     {'PASS' if max_dist_strict <= n_hops else 'FAIL'}")

    # Verify strict mode enforces constraint
    if max_dist_strict <= n_hops and max_dist_non_strict > n_hops:
        logger.info("\n✓ SUCCESS: Strict mode correctly enforces n-hop constraint!")
    elif max_dist_strict <= n_hops and max_dist_non_strict <= n_hops:
        logger.info("\n⚠ NOTE: Both modes stayed within constraint (graph may be too small)")
    else:
        logger.info("\n✗ FAILURE: Strict mode did not enforce constraint properly")

    logger.info("=" * 70)


if __name__ == '__main__':
    test_strict_vs_non_strict()
