#!/usr/bin/env python3
"""Debug the original failing test case."""

import numpy as np
import logging
from filter_training_by_proximity_pyg import ProximityFilterPyG

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Recreate the test graph from test_strict_hop.py
edges = [
    # Main chain
    (0, 0, 1),
    (1, 0, 2),
    (2, 0, 3),
    (3, 0, 4),
    (4, 0, 5),

    # Branch from node 1
    (1, 0, 6),
    (6, 0, 8),

    # Branch from node 3
    (3, 0, 7),
    (7, 0, 9),

    # Cross-edge
    (8, 0, 9),
    (9, 0, 10),
]

# Add reverse edges
reverse_edges = [(t, r, h) for h, r, t in edges]
all_edges = edges + reverse_edges
training_triples = np.array(all_edges)

logger.info("Graph structure:")
logger.info("  0 - 1 - 2 - 3 - 4 - 5")
logger.info("      |       |")
logger.info("      6       7")
logger.info("      |       |")
logger.info("      8 ----- 9 - 10\n")

# Create filter
filter_obj = ProximityFilterPyG(training_triples)

# Test parameters
test_triple = (0, 0, 1)
n_hops = 2

logger.info(f"Test triple: {test_triple}")
logger.info(f"n_hops: {n_hops}\n")

# Compute hop distances
hop_distances = filter_obj._compute_hop_distances([0, 1], n_hops)

logger.info("Hop distances from test entities [0, 1]:")
for i in range(11):
    dist = hop_distances[i].item()
    logger.info(f"  Node {i}: {dist} hops")

logger.info(f"\n{'='*60}")
logger.info("Filtering with STRICT mode")
logger.info(f"{'='*60}\n")

# Filter
filtered = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=n_hops,
    min_degree=1,
    preserve_test_entity_edges=True,
    strict_hop_constraint=True
)

logger.info(f"Filtered to {len(filtered)} edges\n")

# Analyze distances in filtered graph
logger.info("Edges in filtered graph (grouped by max distance):")
by_max_dist = {}
for h, r, t in filtered:
    h_dist = hop_distances[h].item()
    t_dist = hop_distances[t].item()
    max_dist = max(h_dist, t_dist)

    if max_dist not in by_max_dist:
        by_max_dist[max_dist] = []
    by_max_dist[max_dist].append((h, t, h_dist, t_dist))

for max_dist in sorted(by_max_dist.keys()):
    logger.info(f"\n  Max distance {max_dist} hops:")
    for h, t, h_dist, t_dist in by_max_dist[max_dist]:
        logger.info(f"    ({h}, {t}): distances=({h_dist}, {t_dist})")

max_dist_overall = max(by_max_dist.keys()) if by_max_dist else 0
logger.info(f"\nOverall maximum distance: {max_dist_overall} hops")
if max_dist_overall <= n_hops:
    logger.info(f"✓ SUCCESS: All edges within {n_hops} hops")
else:
    logger.info(f"✗ FAILURE: Found edges with endpoints > {n_hops} hops")
