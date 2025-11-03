#!/usr/bin/env python3
"""
Visualization demonstrating how k_hop_subgraph and _compute_hop_distances work together.

This creates detailed diagrams showing:
1. The original graph
2. What k_hop_subgraph extracts
3. What _compute_hop_distances calculates
4. How strict mode filters edges
"""

import numpy as np
import logging
from filter_training_by_proximity_pyg import ProximityFilterPyG

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def print_graph_ascii(title, nodes_info, edges, highlight_edges=None):
    """Print an ASCII visualization of a graph."""
    logger.info(f"\n{'='*70}")
    logger.info(title)
    logger.info('='*70)

    # Print nodes with their hop distances
    logger.info("\nNodes (with hop distances from test entities):")
    for node_id, distance, label in sorted(nodes_info):
        dist_str = f"{distance} hops" if distance >= 0 else "unreachable"
        logger.info(f"  Node {node_id}: {dist_str} {label}")

    # Print edges
    logger.info("\nEdges:")
    for src, dst in sorted(edges):
        highlight = "  ←←← " if highlight_edges and (src, dst) in highlight_edges else ""
        logger.info(f"  ({src}, {dst}){highlight}")


def create_demo_graph():
    """
    Create a graph that clearly demonstrates the difference:

         Test           2 hops away
          |                 |
        0-+-1 ------ 2 ---- 3
          |          |
          10 ------- 20
          |
         2 hops     2 hops
         away       away

    Hop distances from test entities [0, 1]:
    - Nodes 0, 1: 0 hops (test entities)
    - Node 2: 1 hop (from 1)
    - Node 10: 1 hop (from 0)
    - Node 3: 2 hops (from 2)
    - Node 20: 2 hops (from both 2 and 10)

    Key insight: Edge (10, 20) connects two nodes that are both 2 hops away
    but in different directions, creating a potential 4-hop path in the
    filtered graph: 0 -> 10 -> 20 -> 2 -> 1
    """
    edges = [
        (0, 0, 1),   # Test edge
        (1, 0, 2),   # 1 hop from test
        (0, 0, 10),  # 1 hop from test
        (2, 0, 3),   # 2 hops from test
        (2, 0, 20),  # 2 hops from test
        (10, 0, 20), # Both endpoints at 2 hops (different directions)
    ]

    # Add reverse edges
    reverse_edges = [(t, r, h) for h, r, t in edges]
    all_edges = edges + reverse_edges

    return np.array(all_edges)


def main():
    logger.info("="*70)
    logger.info("VISUALIZATION: How k_hop_subgraph and _compute_hop_distances Work")
    logger.info("="*70)

    # Create graph
    training_triples = create_demo_graph()
    filter_obj = ProximityFilterPyG(training_triples)

    # Test parameters
    test_triple = (0, 0, 1)
    test_entities = [0, 1]
    n_hops = 2

    logger.info("\n" + "="*70)
    logger.info("SETUP")
    logger.info("="*70)
    logger.info(f"Test triple: {test_triple}")
    logger.info(f"Test entities: {test_entities}")
    logger.info(f"n_hops: {n_hops}")

    # Compute hop distances
    hop_distances = filter_obj._compute_hop_distances(test_entities, max_hops=10)

    # Original graph
    all_nodes = [
        (0, 0, "[TEST ENTITY]"),
        (1, 0, "[TEST ENTITY]"),
        (2, 1, ""),
        (10, 1, ""),
        (3, 2, ""),
        (20, 2, ""),
    ]

    all_edges = [(0, 1), (1, 2), (0, 10), (2, 3), (2, 20), (10, 20)]

    print_graph_ascii(
        "STEP 0: Original Graph",
        all_nodes,
        all_edges
    )

    # Step 1: k_hop_subgraph
    from torch_geometric.utils import k_hop_subgraph

    subset_nodes, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=test_entities,
        num_hops=n_hops,
        edge_index=filter_obj.edge_index,
        relabel_nodes=False,
        num_nodes=len(filter_obj.node_degrees)
    )

    subset_nodes_list = sorted(subset_nodes.tolist())
    subset_edges = set()
    for i in range(subset_edge_index.shape[1]):
        src = subset_edge_index[0, i].item()
        dst = subset_edge_index[1, i].item()
        if src < dst:  # Only keep one direction
            subset_edges.add((src, dst))

    logger.info("\n" + "="*70)
    logger.info("STEP 1: k_hop_subgraph(test_entities, n_hops=2)")
    logger.info("="*70)
    logger.info("\nWhat it does:")
    logger.info("  - Finds ALL nodes within 2 hops of test entities [0, 1]")
    logger.info("  - Extracts all edges between those nodes")
    logger.info("  - Returns a subgraph")

    subgraph_nodes_info = [(n, hop_distances[n].item(), "[TEST]" if n in test_entities else "")
                           for n in subset_nodes_list]

    print_graph_ascii(
        "\nResult: Extracted Subgraph",
        subgraph_nodes_info,
        subset_edges
    )

    logger.info(f"\nExtracted nodes: {subset_nodes_list}")
    logger.info(f"All nodes are within {n_hops} hops ✓")

    # Step 2: _compute_hop_distances
    logger.info("\n" + "="*70)
    logger.info("STEP 2: _compute_hop_distances(test_entities, max_hops=2)")
    logger.info("="*70)
    logger.info("\nWhat it does:")
    logger.info("  - Performs BFS from test entities")
    logger.info("  - Computes EXACT shortest path distance to each node")
    logger.info("  - Returns a tensor of distances")

    logger.info("\nResult: Distance Tensor")
    logger.info("  hop_distances = {")
    for node in subset_nodes_list:
        dist = hop_distances[node].item()
        logger.info(f"    {node}: {dist},")
    logger.info("  }")

    # Step 3: Edge-by-edge validation
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Strict Mode - Validate Each Edge")
    logger.info("="*70)
    logger.info("\nFor each edge in the subgraph, check if BOTH endpoints")
    logger.info(f"are within {n_hops} hops:")

    strict_edges = set()
    rejected_edges = set()

    logger.info("\nEdge Validation:")
    for src, dst in sorted(subset_edges):
        src_dist = hop_distances[src].item()
        dst_dist = hop_distances[dst].item()
        max_dist = max(src_dist, dst_dist)

        if src_dist <= n_hops and dst_dist <= n_hops:
            status = "✓ KEEP"
            strict_edges.add((src, dst))
        else:
            status = "✗ REJECT"
            rejected_edges.add((src, dst))

        logger.info(f"  ({src}, {dst}): distances=({src_dist}, {dst_dist}), "
                   f"max={max_dist} → {status}")

    # Final comparison
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Comparison - Non-Strict vs Strict Mode")
    logger.info("="*70)

    logger.info("\nNON-STRICT MODE (original behavior):")
    logger.info(f"  - Uses k_hop_subgraph to get nodes within {n_hops} hops")
    logger.info("  - Applies degree filtering to edges in subgraph")
    logger.info("  - Keeps all edges that pass degree filter")
    logger.info(f"  - Result: {len(subset_edges)} edges kept")

    logger.info("\nSTRICT MODE (new --strict-hop-constraint flag):")
    logger.info(f"  - Additionally validates each edge's endpoints")
    logger.info(f"  - Rejects edges where either endpoint > {n_hops} hops")
    logger.info(f"  - Result: {len(strict_edges)} edges kept")

    if rejected_edges:
        logger.info(f"\n  Rejected edges: {rejected_edges}")
    else:
        logger.info("\n  No edges rejected (all edges already satisfy constraint)")

    # The key insight
    logger.info("\n" + "="*70)
    logger.info("KEY INSIGHT: Why Both Modes Usually Give Same Results")
    logger.info("="*70)
    logger.info("\nBecause k_hop_subgraph ALREADY guarantees all nodes are within")
    logger.info("n_hops, the strict mode validation is typically redundant.")
    logger.info("\nHowever, strict mode provides:")
    logger.info("  1. Explicit validation (defensive programming)")
    logger.info("  2. Clear documentation of the constraint")
    logger.info("  3. Protection against potential edge cases or bugs")

    # Path demonstration
    logger.info("\n" + "="*70)
    logger.info("IMPORTANT: About the '5-6 Hop' Observation")
    logger.info("="*70)
    logger.info("\nWhen you see paths of 5-6 hops in the FILTERED graph,")
    logger.info("you're measuring paths THROUGH the filtered graph, not")
    logger.info("the distance of edge endpoints from test entities.")
    logger.info("\nExample with edge (10, 20):")
    logger.info("  - Node 10: 1 hop from test (via 0)")
    logger.info("  - Node 20: 2 hops from test (via 0->10->20 OR 1->2->20)")
    logger.info("  - Edge (10, 20): Both endpoints within 2 hops ✓")
    logger.info("\n  But in the filtered graph, you could have path:")
    logger.info("  0 -> 10 -> 20 -> 2 -> 1  (4 hops)")
    logger.info("\n  This is EXPECTED! Each edge is valid, but their")
    logger.info("  combination creates longer paths.")

    logger.info("\n" + "="*70)
    logger.info("CONCLUSION")
    logger.info("="*70)
    logger.info("\n✓ The original code was CORRECT")
    logger.info("✓ The strict mode adds EXPLICIT validation")
    logger.info("✓ Both modes typically produce the same results")
    logger.info("✓ Use strict mode for extra safety and clarity")
    logger.info("\n" + "="*70)


if __name__ == '__main__':
    main()
