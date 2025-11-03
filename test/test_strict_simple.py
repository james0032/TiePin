#!/usr/bin/env python3
"""
Simple test demonstrating strict vs non-strict hop constraint.

The key difference: strict mode ensures EVERY edge has both endpoints
within n_hops, while non-strict mode extracts the n-hop neighborhood
then applies degree filtering which might keep all edges in that neighborhood.
"""

import numpy as np
import logging
from filter_training_by_proximity_pyg import ProximityFilterPyG

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Simple chain: 0 - 1 - 2 - 3 - 4
edges = [
    (0, 0, 1),
    (1, 0, 2),
    (2, 0, 3),
    (3, 0, 4),
]
reverse_edges = [(t, r, h) for h, r, t in edges]
training_triples = np.array(edges + reverse_edges)

logger.info("Graph: 0 - 1 - 2 - 3 - 4")
logger.info("Test triple: (0, 0, 1)")
logger.info("=" * 60 + "\n")

filter_obj = ProximityFilterPyG(training_triples)
test_triple = (0, 0, 1)

# Test with n_hops=1, which should only include nodes {0, 1, 2}
n_hops = 1

logger.info(f"TEST: n_hops={n_hops}, min_degree=1\n")

# Compute expected hop distances
hop_distances = filter_obj._compute_hop_distances([0, 1], 10)
logger.info("Hop distances from test entities [0, 1]:")
for i in range(5):
    logger.info(f"  Node {i}: {hop_distances[i].item()} hops")

logger.info(f"\nWith n_hops={n_hops}, we expect to keep:")
logger.info("  - Nodes: 0, 1, 2 (all within 1 hop)")
logger.info("  - Edges: (0,1), (1,2)")
logger.info("  - Edge (2,3) should be EXCLUDED (node 3 is at 2 hops)")

# Test both modes
logger.info("\n" + "=" * 60)
logger.info("NON-STRICT MODE")
logger.info("=" * 60)

filtered_non_strict = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=n_hops,
    min_degree=1,
    preserve_test_entity_edges=True,
    strict_hop_constraint=False
)

logger.info(f"\nFiltered edges ({len(filtered_non_strict)} total):")
for h, r, t in filtered_non_strict:
    h_dist = hop_distances[h].item()
    t_dist = hop_distances[t].item()
    max_dist = max(h_dist, t_dist)
    status = "✓" if max_dist <= n_hops else "✗"
    logger.info(f"  {status} ({h}, {t}): distances=({h_dist}, {t_dist}), max={max_dist}")

logger.info("\n" + "=" * 60)
logger.info("STRICT MODE")
logger.info("=" * 60)

filtered_strict = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=n_hops,
    min_degree=1,
    preserve_test_entity_edges=True,
    strict_hop_constraint=True
)

logger.info(f"\nFiltered edges ({len(filtered_strict)} total):")
for h, r, t in filtered_strict:
    h_dist = hop_distances[h].item()
    t_dist = hop_distances[t].item()
    max_dist = max(h_dist, t_dist)
    status = "✓" if max_dist <= n_hops else "✗"
    logger.info(f"  {status} ({h}, {t}): distances=({h_dist}, {t_dist}), max={max_dist}")

# Summary
logger.info("\n" + "=" * 60)
logger.info("SUMMARY")
logger.info("=" * 60)

max_non_strict = max((max(hop_distances[h].item(), hop_distances[t].item())
                      for h, r, t in filtered_non_strict), default=0)
max_strict = max((max(hop_distances[h].item(), hop_distances[t].item())
                  for h, r, t in filtered_strict), default=0)

logger.info(f"Non-strict: {len(filtered_non_strict)} edges, max endpoint dist: {max_non_strict}")
logger.info(f"Strict:     {len(filtered_strict)} edges, max endpoint dist: {max_strict}")

if max_strict <= n_hops < max_non_strict:
    logger.info(f"\n✓ SUCCESS: Strict mode enforces {n_hops}-hop constraint!")
elif max_strict <= n_hops and max_non_strict <= n_hops:
    logger.info(f"\n✓ BOTH PASS: Both modes respect the {n_hops}-hop constraint")
    logger.info("   (This is expected! k_hop_subgraph already limits nodes)")
else:
    logger.info(f"\n✗ FAILURE: Constraint not properly enforced")

logger.info("\nNote: The 'strict' mode provides extra safety by explicitly")
logger.info("      checking each edge, preventing any edge with endpoints")
logger.info(f"      beyond {n_hops} hops from being included.")
