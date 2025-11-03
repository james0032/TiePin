#!/usr/bin/env python3
"""Debug script with more detailed output."""

import numpy as np
import logging
from filter_training_by_proximity_pyg import ProximityFilterPyG
import torch

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Create simple test graph
edges = [
    (0, 0, 1),
    (1, 0, 2),
    (2, 0, 3),  # This edge: node 2 is at 1 hop, node 3 is at 2 hops
]

# Add reverse edges
reverse_edges = [(t, r, h) for h, r, t in edges]
all_edges = edges + reverse_edges

training_triples = np.array(all_edges)

logger.info("Linear graph: 0 - 1 - 2 - 3")
logger.info(f"Total edges (with reverse): {len(training_triples)}\n")

# Create filter
filter_obj = ProximityFilterPyG(training_triples)

# Test triple
test_triple = (0, 0, 1)
test_h, test_t = 0, 1
n_hops = 1  # Use n_hops=1 to make it very strict

logger.info(f"Test triple: {test_triple}")
logger.info(f"n_hops: {n_hops}\n")

# Compute hop distances
hop_distances = filter_obj._compute_hop_distances([test_h, test_t], n_hops)

logger.info("Hop distances from test entities [0, 1]:")
for i in range(min(10, len(hop_distances))):
    dist = hop_distances[i].item()
    logger.info(f"  Node {i}: {dist} hops {'(TEST ENTITY)' if i in [0, 1] else ''}")

logger.info(f"\n{'='*60}")
logger.info("Testing STRICT mode with preserve_test_entity_edges=True")
logger.info(f"{'='*60}\n")

# Get the subgraph
from torch_geometric.utils import k_hop_subgraph, degree

subset_nodes, subset_edge_index, mapping, edge_mask = k_hop_subgraph(
    node_idx=[test_h, test_t],
    num_hops=n_hops,
    edge_index=filter_obj.edge_index,
    relabel_nodes=False,
    num_nodes=len(filter_obj.node_degrees)
)

logger.info(f"Nodes in {n_hops}-hop neighborhood: {sorted(subset_nodes.tolist())}\n")

logger.info("Edges in subgraph (before filtering):")
for i in range(subset_edge_index.shape[1]):
    src = subset_edge_index[0, i].item()
    dst = subset_edge_index[1, i].item()
    src_dist = hop_distances[src].item()
    dst_dist = hop_distances[dst].item()

    is_test_edge = (src == test_h or src == test_t or dst == test_h or dst == test_t)
    within_hops = (src_dist >= 0 and src_dist <= n_hops and dst_dist >= 0 and dst_dist <= n_hops)

    logger.info(f"  ({src}, {dst}): distances=({src_dist}, {dst_dist}) "
               f"test_edge={is_test_edge} within_hops={within_hops}")

logger.info("\nExpected behavior in strict mode:")
logger.info("  - Edge (0, 1): KEEP (test edge, both at 0 hops)")
logger.info("  - Edge (1, 2): KEEP (one test entity, both within 1 hop)")
logger.info(f"  - Edge (2, 3): REJECT (node 3 is at {hop_distances[3].item()} hops > {n_hops})")

# Now actually filter
filtered = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=n_hops,
    min_degree=1,
    preserve_test_entity_edges=True,
    strict_hop_constraint=True
)

logger.info(f"\n{'='*60}")
logger.info("Filtered result:")
logger.info(f"{'='*60}\n")

logger.info(f"Number of filtered edges: {len(filtered)}")
logger.info("Filtered triples:")
for h, r, t in filtered:
    h_dist = hop_distances[h].item()
    t_dist = hop_distances[t].item()
    logger.info(f"  ({h}, {r}, {t}): distances=({h_dist}, {t_dist})")

# Check if it worked
max_dist = max(max(hop_distances[h].item(), hop_distances[t].item()) for h, r, t in filtered)
logger.info(f"\nMaximum hop distance in filtered graph: {max_dist}")
if max_dist <= n_hops:
    logger.info("✓ SUCCESS: Strict mode working correctly!")
else:
    logger.info(f"✗ FAILURE: Found edges beyond {n_hops} hops")
