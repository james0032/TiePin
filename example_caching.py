"""
Example demonstrating graph caching for faster repeated filtering.

Shows dramatic speedup when analyzing multiple test triples with same training data.
"""

import numpy as np
import time
from pathlib import Path
from filter_training_by_proximity_pyg import ProximityFilterPyG

print("=" * 80)
print("Graph Caching Demo - Speed Comparison")
print("=" * 80)

# Create example training data (simulating large dataset)
np.random.seed(42)
num_train = 10000
num_entities = 1000
num_relations = 50

training_triples = np.array([
    [np.random.randint(0, num_entities),
     np.random.randint(0, num_relations),
     np.random.randint(0, num_entities)]
    for _ in range(num_train)
])

# Example test triples
test_triples = [
    (0, 0, 1),
    (2, 1, 3),
    (5, 2, 10),
    (15, 3, 20),
    (25, 4, 30),
]

print(f"\nDataset: {num_train:,} training triples, {len(test_triples)} test triples")
print(f"Entities: {num_entities}, Relations: {num_relations}")

# ============================================================================
print("\n" + "=" * 80)
print("Test 1: WITHOUT Caching (Building graph each time)")
print("=" * 80)

start_time = time.time()

for i, test_triple in enumerate(test_triples):
    iter_start = time.time()

    # Build graph from scratch each time
    filter_obj = ProximityFilterPyG(training_triples, cache_path=None)

    # Filter
    filtered = filter_obj.filter_for_single_test_triple(
        test_triple=test_triple,
        n_hops=2,
        min_degree=2
    )

    iter_time = time.time() - iter_start
    print(f"  Test triple {i+1}: {len(filtered):4d} triples ({iter_time:.2f}s)")

total_time_no_cache = time.time() - start_time
print(f"\nTotal time WITHOUT caching: {total_time_no_cache:.2f}s")

# ============================================================================
print("\n" + "=" * 80)
print("Test 2: WITH Caching (Building graph once, reusing)")
print("=" * 80)

cache_path = "/tmp/train_graph_cache.pkl"
start_time = time.time()

# First iteration: Build and cache
print("\n  Building and caching graph...")
build_start = time.time()
filter_obj = ProximityFilterPyG(training_triples, cache_path=cache_path)
build_time = time.time() - build_start
print(f"  Graph built and cached: {build_time:.2f}s")

# Filter all test triples (graph already built)
print("\n  Filtering test triples (using cached graph)...")
for i, test_triple in enumerate(test_triples):
    iter_start = time.time()

    filtered = filter_obj.filter_for_single_test_triple(
        test_triple=test_triple,
        n_hops=2,
        min_degree=2
    )

    iter_time = time.time() - iter_start
    print(f"  Test triple {i+1}: {len(filtered):4d} triples ({iter_time:.3f}s)")

total_time_with_cache = time.time() - start_time
print(f"\nTotal time WITH caching: {total_time_with_cache:.2f}s")

# ============================================================================
print("\n" + "=" * 80)
print("Test 3: Reloading from Cache (Next run)")
print("=" * 80)

print("\n  Simulating a new run - loading from cache...")
start_time = time.time()

# This will load from cache instead of building
load_start = time.time()
filter_obj_reloaded = ProximityFilterPyG(training_triples, cache_path=cache_path)
load_time = time.time() - load_start
print(f"  Graph loaded from cache: {load_time:.2f}s")

# Filter one test triple
test_triple = test_triples[0]
filter_start = time.time()
filtered = filter_obj_reloaded.filter_for_single_test_triple(
    test_triple=test_triple,
    n_hops=2,
    min_degree=2
)
filter_time = time.time() - filter_start

total_time_reload = time.time() - start_time
print(f"  Filtering time: {filter_time:.3f}s")
print(f"\nTotal time (load + filter): {total_time_reload:.2f}s")

# ============================================================================
print("\n" + "=" * 80)
print("Performance Summary")
print("=" * 80)

speedup_vs_no_cache = total_time_no_cache / total_time_with_cache
speedup_reload = build_time / load_time

print(f"\n1. WITHOUT caching (build graph 5 times):")
print(f"   Time: {total_time_no_cache:.2f}s")
print(f"   Avg per triple: {total_time_no_cache / len(test_triples):.2f}s")

print(f"\n2. WITH caching (build once, use 5 times):")
print(f"   Time: {total_time_with_cache:.2f}s")
print(f"   Build time: {build_time:.2f}s")
print(f"   Avg filter time: {(total_time_with_cache - build_time) / len(test_triples):.3f}s")
print(f"   Speedup: {speedup_vs_no_cache:.1f}x faster!")

print(f"\n3. Reload from cache:")
print(f"   Load time: {load_time:.2f}s")
print(f"   Build time: {build_time:.2f}s")
print(f"   Speedup: {speedup_reload:.1f}x faster than building!")

# ============================================================================
print("\n" + "=" * 80)
print("When to Use Caching")
print("=" * 80)

print("""
✓ Use caching when:
  • Analyzing multiple test triples with same training data
  • Running experiments with different hyperparameters (n_hops, min_degree)
  • Iterating on analysis (same train.txt, different test triples)
  • Training data is large (>10K triples)

✗ Don't use caching when:
  • Training data changes between runs
  • Only filtering once
  • Training data is very small (<1K triples)
  • Storage space is limited

Cache automatically invalidates when training data changes!
""")

# ============================================================================
print("=" * 80)
print("Usage Examples")
print("=" * 80)

print("""
1. Command line with cache:

   # First run - builds and caches graph
   python filter_training_by_proximity_pyg.py \\
       --train train.txt \\
       --test test.txt \\
       --output filtered_1.txt \\
       --cache /path/to/train_graph.pkl \\
       --n-hops 2

   # Second run - loads from cache (much faster!)
   python filter_training_by_proximity_pyg.py \\
       --train train.txt \\
       --test test2.txt \\
       --output filtered_2.txt \\
       --cache /path/to/train_graph.pkl \\  # Same cache file
       --n-hops 2

2. Python API with cache:

   from filter_training_by_proximity_pyg import ProximityFilterPyG

   # Create filter with caching
   filter_obj = ProximityFilterPyG(
       training_triples,
       cache_path='train_graph.pkl'  # Will cache on first run
   )

   # Analyze multiple test triples (graph is already built)
   for test_triple in test_triples:
       filtered = filter_obj.filter_for_single_test_triple(
           test_triple=test_triple,
           n_hops=2
       )
       # Process filtered data...

3. Factory method:

   # Convenient factory method
   filter_obj = ProximityFilterPyG.from_cache_or_build(
       training_triples=train_data,
       cache_path='train_graph.pkl'
   )

4. Integration with TracIn:

   from filter_training_by_proximity_pyg import ProximityFilterPyG
   from tracin_optimized import TracInAnalyzerOptimized

   # Build/load cached graph once
   filter_obj = ProximityFilterPyG(
       training_triples,
       cache_path='train_graph.pkl'
   )

   # Analyze many test triples efficiently
   for test_triple in all_test_triples:
       # Filter training data (fast - graph already built!)
       filtered_train = filter_obj.filter_for_single_test_triple(
           test_triple=test_triple,
           n_hops=2
       )

       # Run TracIn on filtered data
       analyzer = TracInAnalyzerOptimized(...)
       influences = analyzer.compute_influences_sampled(
           test_triple=test_triple,
           training_triples=filtered_train,
           sample_rate=0.2
       )
""")

print("=" * 80)
print("\nCache file location:", cache_path)
print("Cache file exists:", Path(cache_path).exists())
if Path(cache_path).exists():
    import os
    cache_size = os.path.getsize(cache_path) / 1024 / 1024
    print(f"Cache file size: {cache_size:.2f} MB")
print("=" * 80)
