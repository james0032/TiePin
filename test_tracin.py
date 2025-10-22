"""
Unit tests for TracIn implementation.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory

from tracin import TracInAnalyzer


class TestTracInAnalyzer(unittest.TestCase):
    """Test TracInAnalyzer class."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are reused across tests."""
        # Create small synthetic dataset
        cls.num_entities = 50
        cls.num_relations = 10
        cls.embedding_dim = 32
        cls.output_channels = 8

        # Generate synthetic triples
        np.random.seed(42)
        cls.train_triples = np.array([
            [np.random.randint(0, cls.num_entities),
             np.random.randint(0, cls.num_relations),
             np.random.randint(0, cls.num_entities)]
            for _ in range(100)
        ], dtype=np.int64)

        cls.test_triples = np.array([
            [np.random.randint(0, cls.num_entities),
             np.random.randint(0, cls.num_relations),
             np.random.randint(0, cls.num_entities)]
            for _ in range(10)
        ], dtype=np.int64)

        # Create entity and relation mappings
        cls.entity_to_id = {f'entity_{i}': i for i in range(cls.num_entities)}
        cls.relation_to_id = {f'relation_{i}': i for i in range(cls.num_relations)}

        # Save to temporary files
        cls.temp_dir = tempfile.mkdtemp()
        cls.train_path = Path(cls.temp_dir) / 'train.txt'
        cls.test_path = Path(cls.temp_dir) / 'test.txt'
        cls.entity_path = Path(cls.temp_dir) / 'entity_to_id.tsv'
        cls.relation_path = Path(cls.temp_dir) / 'relation_to_id.tsv'

        # Write train triples
        with open(cls.train_path, 'w') as f:
            for h, r, t in cls.train_triples:
                f.write(f'entity_{h}\trelation_{r}\tentity_{t}\n')

        # Write test triples
        with open(cls.test_path, 'w') as f:
            for h, r, t in cls.test_triples:
                f.write(f'entity_{h}\trelation_{r}\tentity_{t}\n')

        # Write entity mapping
        with open(cls.entity_path, 'w') as f:
            for entity, idx in cls.entity_to_id.items():
                f.write(f'{entity}\t{idx}\n')

        # Write relation mapping
        with open(cls.relation_path, 'w') as f:
            for relation, idx in cls.relation_to_id.items():
                f.write(f'{relation}\t{idx}\n')

        # Create triples factory
        cls.train_factory = TriplesFactory.from_path(
            path=str(cls.train_path),
            entity_to_id=cls.entity_to_id,
            relation_to_id=cls.relation_to_id
        )

        cls.test_factory = TriplesFactory.from_path(
            path=str(cls.test_path),
            entity_to_id=cls.entity_to_id,
            relation_to_id=cls.relation_to_id
        )

        # Create and initialize a simple model
        cls.model = ConvE(
            triples_factory=cls.train_factory,
            embedding_dim=cls.embedding_dim,
            output_channels=cls.output_channels
        )

    def test_initialization(self):
        """Test TracInAnalyzer initialization."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')
        self.assertIsNotNone(analyzer)
        self.assertEqual(analyzer.loss_fn, 'bce')
        self.assertEqual(analyzer.device, 'cpu')

    def test_compute_gradient(self):
        """Test gradient computation for a single triple."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        # Get a test triple
        h, r, t = 5, 2, 10

        # Compute gradient
        grad = analyzer.compute_gradient(h, r, t, label=1.0)

        # Check that gradients were computed
        self.assertIsInstance(grad, dict)
        self.assertGreater(len(grad), 0)

        # Check that gradients have correct shapes
        for name, gradient in grad.items():
            self.assertIsInstance(gradient, torch.Tensor)
            self.assertFalse(torch.isnan(gradient).any(), f"NaN found in gradient {name}")
            self.assertFalse(torch.isinf(gradient).any(), f"Inf found in gradient {name}")

    def test_compute_gradient_eval_mode(self):
        """Test that compute_gradient keeps model in eval mode."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        # Compute gradient
        h, r, t = 5, 2, 10
        grad = analyzer.compute_gradient(h, r, t, label=1.0)

        # Check model is still in eval mode
        self.assertFalse(self.model.training)

    def test_compute_influence(self):
        """Test influence computation between two triples."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        train_triple = (5, 2, 10)
        test_triple = (3, 1, 8)

        # Compute influence
        influence = analyzer.compute_influence(
            train_triple=train_triple,
            test_triple=test_triple,
            learning_rate=0.001
        )

        # Check that influence is a scalar
        self.assertIsInstance(influence, float)
        self.assertFalse(np.isnan(influence))
        self.assertFalse(np.isinf(influence))

    def test_compute_batch_individual_gradients(self):
        """Test batched gradient computation."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        # Create a small batch
        batch_size = 5
        triples_batch = torch.LongTensor(self.train_triples[:batch_size])

        # Compute batch gradients
        batch_gradients = analyzer.compute_batch_individual_gradients(triples_batch)

        # Check results
        self.assertEqual(len(batch_gradients), batch_size)
        for grad_dict in batch_gradients:
            self.assertIsInstance(grad_dict, dict)
            self.assertGreater(len(grad_dict), 0)
            for name, gradient in grad_dict.items():
                self.assertIsInstance(gradient, torch.Tensor)
                self.assertFalse(torch.isnan(gradient).any())

    def test_compute_influences_for_test_triple_small(self):
        """Test computing influences for a test triple with small dataset."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        # Use only first 20 training triples for speed
        small_train_factory = TriplesFactory(
            mapped_triples=self.train_factory.mapped_triples[:20],
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id
        )

        test_triple = tuple(int(x) for x in self.test_factory.mapped_triples[0])

        # Compute influences
        influences = analyzer.compute_influences_for_test_triple(
            test_triple=test_triple,
            training_triples=small_train_factory,
            learning_rate=0.001,
            top_k=5,
            batch_size=8
        )

        # Check results
        self.assertIsInstance(influences, list)
        self.assertLessEqual(len(influences), 5)  # top_k=5

        for inf in influences:
            self.assertIn('train_head', inf)
            self.assertIn('train_relation', inf)
            self.assertIn('train_tail', inf)
            self.assertIn('influence', inf)
            self.assertIsInstance(inf['influence'], float)
            self.assertFalse(np.isnan(inf['influence']))

    def test_compute_influences_different_batch_sizes(self):
        """Test that different batch sizes give same results."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        # Use small dataset
        small_train_factory = TriplesFactory(
            mapped_triples=self.train_factory.mapped_triples[:10],
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id
        )

        test_triple = tuple(int(x) for x in self.test_factory.mapped_triples[0])

        # Compute with batch_size=1
        influences_bs1 = analyzer.compute_influences_for_test_triple(
            test_triple=test_triple,
            training_triples=small_train_factory,
            learning_rate=0.001,
            top_k=None,
            batch_size=1
        )

        # Compute with batch_size=5
        influences_bs5 = analyzer.compute_influences_for_test_triple(
            test_triple=test_triple,
            training_triples=small_train_factory,
            learning_rate=0.001,
            top_k=None,
            batch_size=5
        )

        # Results should be identical
        self.assertEqual(len(influences_bs1), len(influences_bs5))

        for inf1, inf5 in zip(influences_bs1, influences_bs5):
            self.assertEqual(inf1['train_head'], inf5['train_head'])
            self.assertEqual(inf1['train_relation'], inf5['train_relation'])
            self.assertEqual(inf1['train_tail'], inf5['train_tail'])
            # Allow small numerical differences
            self.assertAlmostEqual(inf1['influence'], inf5['influence'], places=5)

    def test_compute_self_influence(self):
        """Test self-influence computation."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        # Use small dataset
        small_train_factory = TriplesFactory(
            mapped_triples=self.train_factory.mapped_triples[:10],
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id
        )

        # Compute self-influences
        influences = analyzer.compute_self_influence(
            training_triples=small_train_factory,
            learning_rate=0.001
        )

        # Check results
        self.assertEqual(len(influences), 10)
        for inf in influences:
            self.assertIn('head', inf)
            self.assertIn('relation', inf)
            self.assertIn('tail', inf)
            self.assertIn('self_influence', inf)
            self.assertGreater(inf['self_influence'], 0)  # Self-influence should be positive

    def test_analyze_test_set(self):
        """Test analyzing a test set."""
        analyzer = TracInAnalyzer(model=self.model, device='cpu')

        # Use very small datasets
        small_train_factory = TriplesFactory(
            mapped_triples=self.train_factory.mapped_triples[:10],
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id
        )

        small_test_factory = TriplesFactory(
            mapped_triples=self.test_factory.mapped_triples[:2],
            entity_to_id=self.entity_to_id,
            relation_to_id=self.relation_to_id
        )

        # Analyze test set
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            output_path = f.name

        analysis = analyzer.analyze_test_set(
            test_triples=small_test_factory,
            training_triples=small_train_factory,
            learning_rate=0.001,
            top_k=3,
            output_path=output_path,
            batch_size=5
        )

        # Check results
        self.assertIn('num_test_triples', analysis)
        self.assertEqual(analysis['num_test_triples'], 2)
        self.assertIn('results', analysis)
        self.assertEqual(len(analysis['results']), 2)

        # Check that output file was created
        self.assertTrue(Path(output_path).exists())

        # Read and validate JSON
        with open(output_path, 'r') as f:
            saved_analysis = json.load(f)
        self.assertEqual(saved_analysis['num_test_triples'], 2)

        # Cleanup
        Path(output_path).unlink()


class TestTracInGPUCompatibility(unittest.TestCase):
    """Test GPU compatibility (only runs if CUDA available)."""

    def setUp(self):
        """Skip tests if CUDA not available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

    def test_gpu_initialization(self):
        """Test analyzer initialization on GPU."""
        # Create simple model
        entity_to_id = {f'e{i}': i for i in range(10)}
        relation_to_id = {f'r{i}': i for i in range(3)}

        triples = np.array([[0, 0, 1], [1, 1, 2], [2, 0, 3]], dtype=np.int64)

        factory = TriplesFactory(
            mapped_triples=torch.from_numpy(triples),
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id
        )

        model = ConvE(triples_factory=factory, embedding_dim=16, output_channels=4)

        analyzer = TracInAnalyzer(model=model, device='cuda')
        self.assertEqual(analyzer.device, 'cuda')

        # Test gradient computation on GPU
        grad = analyzer.compute_gradient(0, 0, 1, label=1.0)
        self.assertIsInstance(grad, dict)
        self.assertGreater(len(grad), 0)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
