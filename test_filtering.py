"""
Quick test script to verify filtering implementations work correctly.

Creates a small synthetic graph and tests all filtering modes.
Use this to ensure implementations are working before running on real data.
"""

import numpy as np
import tempfile
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


def create_synthetic_data():
    """
    Create a small synthetic knowledge graph for testing.

    Graph structure:
        Drug1 --[treats]--> Gene1 --[regulates]--> Gene2 --[associated_with]--> Disease1
        Drug1 --[inhibits]--> Protein1 --[interacts]--> Gene2
        Drug2 --[treats]--> Gene3 --[causes]--> Disease2

    This creates:
    - 2-hop path from Drug1 to Disease1 (via Gene1 -> Gene2)
    - 3-hop path from Drug1 to Disease1 (via Protein1 -> Gene2)
    - Disconnected path: Drug2 to Disease2
    """

    # Training triples
    train_triples = [
        # Path 1: Drug1 -> Gene1 -> Gene2 -> Disease1
        ("Drug1", "treats", "Gene1"),
        ("Gene1", "regulates", "Gene2"),
        ("Gene2", "associated_with", "Disease1"),

        # Path 2: Drug1 -> Protein1 -> Gene2
        ("Drug1", "inhibits", "Protein1"),
        ("Protein1", "interacts", "Gene2"),

        # Additional edges for degree filtering
        ("Gene1", "interacts", "Protein2"),
        ("Gene1", "interacts", "Protein3"),
        ("Gene2", "interacts", "Protein4"),

        # Separate component
        ("Drug2", "treats", "Gene3"),
        ("Gene3", "causes", "Disease2"),

        # Distant edges (should be filtered out)
        ("Protein5", "interacts", "Protein6"),
        ("Gene4", "similar_to", "Gene5"),
    ]

    # Test triples (what we're trying to predict)
    test_triples = [
        ("Drug1", "treats", "Disease1"),  # 3-hop path exists
        ("Drug2", "treats", "Disease2"),  # 2-hop path exists
    ]

    return train_triples, test_triples


def save_triples(triples, filepath):
    """Save triples to TSV file."""
    with open(filepath, 'w') as f:
        for h, r, t in triples:
            f.write(f"{h}\t{r}\t{t}\n")


def load_triples(filepath):
    """Load triples from TSV file."""
    triples = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                triples.append(tuple(parts[:3]))
    return triples


def test_filtering_implementation(impl_name, filter_func, train_file, test_file, n_hops=2):
    """
    Test a filtering implementation.

    Args:
        impl_name: Name of implementation (e.g., "NetworkX")
        filter_func: Function that takes train_file, test_file, output_file, n_hops
        train_file: Path to training triples
        test_file: Path to test triples
        n_hops: Number of hops for filtering
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {impl_name} Implementation")
    logger.info(f"{'='*60}")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        output_file = tmp.name

    try:
        # Run filter
        filter_func(train_file, test_file, output_file, n_hops)

        # Load results
        filtered_triples = load_triples(output_file)

        logger.info(f"\n{impl_name} Results:")
        logger.info(f"  Filtered triples: {len(filtered_triples)}")

        if filtered_triples:
            logger.info(f"  Sample triples:")
            for triple in filtered_triples[:5]:
                logger.info(f"    {triple}")

        return filtered_triples

    except Exception as e:
        logger.error(f"❌ {impl_name} test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

    finally:
        # Clean up
        if os.path.exists(output_file):
            os.remove(output_file)


def test_networkx(train_file, test_file, output_file, n_hops):
    """Test NetworkX implementation."""
    from filter_training_networkx import filter_training_file

    filter_training_file(
        train_path=train_file,
        test_path=test_file,
        output_path=output_file,
        n_hops=n_hops,
        min_degree=2,
        preserve_test_entity_edges=True,
        path_filtering=False
    )


def test_pyg(train_file, test_file, output_file, n_hops):
    """Test PyG implementation."""
    from filter_training_by_proximity_pyg import filter_training_file

    filter_training_file(
        train_path=train_file,
        test_path=test_file,
        output_path=output_file,
        n_hops=n_hops,
        min_degree=2,
        preserve_test_entity_edges=True,
        path_filtering=False
    )


def test_igraph(train_file, test_file, output_file, n_hops):
    """Test igraph implementation."""
    from filter_training_igraph import load_triples_from_file, create_entity_mappings, IGraphProximityFilter
    import numpy as np
    from pathlib import Path

    # Load and process data
    train_triples = load_triples_from_file(train_file)
    test_triples = load_triples_from_file(test_file)

    entity_to_idx, idx_to_entity, relation_to_idx = create_entity_mappings(train_triples, test_triples)

    train_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in train_triples
    ])

    test_numeric = np.array([
        [entity_to_idx[h], relation_to_idx[r], entity_to_idx[t]]
        for h, r, t in test_triples
    ])

    # Filter
    filter_obj = IGraphProximityFilter(train_numeric)
    filtered_numeric, _ = filter_obj.filter_for_test_triples(
        test_numeric,
        n_hops=n_hops,
        min_degree=2,
        preserve_test_entity_edges=True,
        path_filtering=False
    )

    # Convert back and save
    idx_to_relation = {v: k for k, v in relation_to_idx.items()}

    output_dir = Path(output_file).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        for h_idx, r_idx, t_idx in filtered_numeric:
            h = idx_to_entity[h_idx]
            r = idx_to_relation[r_idx]
            t = idx_to_entity[t_idx]
            f.write(f"{h}\t{r}\t{t}\n")


def compare_results(results_dict):
    """Compare results from different implementations."""
    logger.info(f"\n{'='*60}")
    logger.info("Comparison Results")
    logger.info(f"{'='*60}")

    # Convert to sets for comparison
    result_sets = {name: set(triples) for name, triples in results_dict.items() if triples is not None}

    if len(result_sets) < 2:
        logger.warning("Need at least 2 implementations to compare")
        return

    # Compare all pairs
    implementations = list(result_sets.keys())
    all_match = True

    for i in range(len(implementations)):
        for j in range(i + 1, len(implementations)):
            impl1 = implementations[i]
            impl2 = implementations[j]

            set1 = result_sets[impl1]
            set2 = result_sets[impl2]

            if set1 == set2:
                logger.info(f"✓ {impl1} and {impl2} produce IDENTICAL results ({len(set1)} triples)")
            else:
                all_match = False
                logger.warning(f"✗ {impl1} and {impl2} produce DIFFERENT results!")
                logger.warning(f"  {impl1}: {len(set1)} triples")
                logger.warning(f"  {impl2}: {len(set2)} triples")
                logger.warning(f"  In both: {len(set1 & set2)}")
                logger.warning(f"  Only in {impl1}: {len(set1 - set2)}")
                logger.warning(f"  Only in {impl2}: {len(set2 - set1)}")

                if len(set1 - set2) > 0:
                    logger.warning(f"\n  Sample triples only in {impl1}:")
                    for triple in list(set1 - set2)[:3]:
                        logger.warning(f"    {triple}")

                if len(set2 - set1) > 0:
                    logger.warning(f"\n  Sample triples only in {impl2}:")
                    for triple in list(set2 - set1)[:3]:
                        logger.warning(f"    {triple}")

    if all_match:
        logger.info(f"\n✓ All implementations agree! Safe to use any of them.")
    else:
        logger.warning(f"\n⚠️  Implementations disagree! Investigate before using.")


def main():
    """Run tests on all implementations."""
    logger.info("="*60)
    logger.info("Knowledge Graph Filtering - Implementation Tests")
    logger.info("="*60)

    # Create synthetic data
    logger.info("\nCreating synthetic test data...")
    train_triples, test_triples = create_synthetic_data()

    logger.info(f"Created {len(train_triples)} training triples")
    logger.info(f"Created {len(test_triples)} test triples")

    # Save to temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_file = os.path.join(tmpdir, 'train.txt')
        test_file = os.path.join(tmpdir, 'test.txt')

        save_triples(train_triples, train_file)
        save_triples(test_triples, test_file)

        logger.info(f"\nSaved to temporary files:")
        logger.info(f"  Train: {train_file}")
        logger.info(f"  Test: {test_file}")

        # Test each implementation
        results = {}

        # Test NetworkX
        try:
            nx_result = test_filtering_implementation(
                "NetworkX", test_networkx, train_file, test_file, n_hops=2
            )
            results['NetworkX'] = nx_result
        except ImportError:
            logger.warning("NetworkX implementation not available")

        # Test PyG
        try:
            pyg_result = test_filtering_implementation(
                "PyG", test_pyg, train_file, test_file, n_hops=2
            )
            results['PyG'] = pyg_result
        except ImportError:
            logger.warning("PyG implementation not available (install torch-geometric)")

        # Test igraph
        try:
            ig_result = test_filtering_implementation(
                "igraph", test_igraph, train_file, test_file, n_hops=2
            )
            results['igraph'] = ig_result
        except ImportError:
            logger.warning("igraph implementation not available (install igraph)")

        # Compare results
        if len(results) > 0:
            compare_results(results)
        else:
            logger.error("No implementations available to test!")

    logger.info(f"\n{'='*60}")
    logger.info("Testing Complete")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
