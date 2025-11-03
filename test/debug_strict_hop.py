#!/usr/bin/env python3
"""Debug script to see what's happening with hop distances."""

import numpy as np
import logging
from filter_training_by_proximity_pyg import ProximityFilterPyG

logging.basicConfig(level=logging.DEBUG, format='%(message)s')
logger = logging.getLogger(__name__)

# Create simple linear graph: 0 - 1 - 2 - 3 - 4 - 5
edges = [
    (0, 0, 1),
    (1, 0, 2),
    (2, 0, 3),
    (3, 0, 4),
    (4, 0, 5),
]

# Add reverse edges
reverse_edges = [(t, r, h) for h, r, t in edges]
all_edges = edges + reverse_edges

training_triples = np.array(all_edges)

logger.info("Linear graph: 0 - 1 - 2 - 3 - 4 - 5")
logger.info(f"Total edges: {len(training_triples)}")

# Create filter
filter_obj = ProximityFilterPyG(training_triples)

# Compute hop distances manually
test_entities = [0, 1]  # From test triple (0, 0, 1)
n_hops = 2

logger.info(f"\nComputing hop distances from test entities {test_entities}...")
hop_distances = filter_obj._compute_hop_distances(test_entities, n_hops)

logger.info("\nHop distances:")
for i in range(6):
    dist = hop_distances[i].item()
    logger.info(f"  Node {i}: {dist} hops")

# Now filter with strict mode
logger.info(f"\n{'='*60}")
logger.info("Filtering with strict_hop_constraint=True, n_hops=2")
logger.info(f"{'='*60}")

test_triple = (0, 0, 1)
filtered = filter_obj.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=n_hops,
    min_degree=1,
    preserve_test_entity_edges=True,
    strict_hop_constraint=True
)

logger.info(f"\nFiltered edges: {len(filtered)}")
logger.info("Filtered triples:")
for h, r, t in filtered:
    h_dist = hop_distances[h].item()
    t_dist = hop_distances[t].item()
    logger.info(f"  ({h}, {r}, {t}) - distances: {h_dist}, {t_dist}")
