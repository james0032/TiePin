"""
Integration tests for run_tracin.py script.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import numpy as np
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory

from run_tracin import run_tracin_analysis


class TestRunTracInAnalysis(unittest.TestCase):
    """Test run_tracin_analysis function."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        # Create small synthetic dataset
        cls.num_entities = 30
        cls.num_relations = 5
        cls.embedding_dim = 32
        cls.output_channels = 8

        # Generate synthetic triples
        np.random.seed(42)
        cls.train_triples = []
        for _ in range(50):
            h = np.random.randint(0, cls.num_entities)
            r = np.random.randint(0, cls.num_relations)
            t = np.random.randint(0, cls.num_entities)
            cls.train_triples.append([h, r, t])

        cls.test_triples = []
        for _ in range(5):
            h = np.random.randint(0, cls.num_entities)
            r = np.random.randint(0, cls.num_relations)
            t = np.random.randint(0, cls.num_entities)
            cls.test_triples.append([h, r, t])

        # Create entity and relation mappings
        cls.entity_to_id = {f'entity_{i}': i for i in range(cls.num_entities)}
        cls.relation_to_id = {f'relation_{i}': i for i in range(cls.num_relations)}

        # Create temporary directory
        cls.temp_dir = tempfile.mkdtemp()
        cls.train_path = Path(cls.temp_dir) / 'train.txt'
        cls.test_path = Path(cls.temp_dir) / 'test.txt'
        cls.entity_path = Path(cls.temp_dir) / 'entity_to_id.tsv'
        cls.relation_path = Path(cls.temp_dir) / 'relation_to_id.tsv'
        cls.model_path = Path(cls.temp_dir) / 'model.pt'

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

        # Create and save a model
        train_factory = TriplesFactory.from_path(
            path=str(cls.train_path),
            entity_to_id=cls.entity_to_id,
            relation_to_id=cls.relation_to_id
        )

        model = ConvE(
            triples_factory=train_factory,
            embedding_dim=cls.embedding_dim,
            output_channels=cls.output_channels
        )

        # Save model
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'config': {
                'embedding_dim': cls.embedding_dim,
                'output_channels': cls.output_channels
            }
        }
        torch.save(checkpoint, cls.model_path)

    def test_single_mode(self):
        """Test run_tracin_analysis in single mode."""
        output_path = Path(self.temp_dir) / 'single_output.json'

        run_tracin_analysis(
            model_path=str(self.model_path),
            train_path=str(self.train_path),
            test_path=str(self.test_path),
            entity_to_id_path=str(self.entity_path),
            relation_to_id_path=str(self.relation_path),
            output_path=str(output_path),
            mode='single',
            test_triple_indices=[0, 1],
            top_k=5,
            learning_rate=0.001,
            embedding_dim=self.embedding_dim,
            output_channels=self.output_channels,
            device='cpu',
            batch_size=10
        )

        # Check output file exists
        self.assertTrue(output_path.exists())

        # Load and validate JSON
        with open(output_path, 'r') as f:
            results = json.load(f)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 2)  # Two test indices

        for result in results:
            self.assertIn('test_triple', result)
            self.assertIn('test_triple_index', result)
            self.assertIn('influences', result)
            self.assertIn('self_influence', result)

            # Check influences
            influences = result['influences']
            self.assertLessEqual(len(influences), 5)  # top_k=5

            for inf in influences:
                self.assertIn('train_head', inf)
                self.assertIn('train_relation', inf)
                self.assertIn('train_tail', inf)
                self.assertIn('influence', inf)

    def test_self_mode(self):
        """Test run_tracin_analysis in self mode."""
        output_path = Path(self.temp_dir) / 'self_output.json'

        run_tracin_analysis(
            model_path=str(self.model_path),
            train_path=str(self.train_path),
            test_path=None,  # Not needed for self mode
            entity_to_id_path=str(self.entity_path),
            relation_to_id_path=str(self.relation_path),
            output_path=str(output_path),
            mode='self',
            learning_rate=0.001,
            embedding_dim=self.embedding_dim,
            output_channels=self.output_channels,
            device='cpu'
        )

        # Check output file exists
        self.assertTrue(output_path.exists())

        # Load and validate JSON
        with open(output_path, 'r') as f:
            results = json.load(f)

        self.assertIsInstance(results, list)
        self.assertEqual(len(results), len(self.train_triples))

        for result in results:
            self.assertIn('head', result)
            self.assertIn('relation', result)
            self.assertIn('tail', result)
            self.assertIn('self_influence', result)
            self.assertGreater(result['self_influence'], 0)

    def test_test_mode_with_max_triples(self):
        """Test run_tracin_analysis in test mode with max_test_triples."""
        output_path = Path(self.temp_dir) / 'test_output.json'

        run_tracin_analysis(
            model_path=str(self.model_path),
            train_path=str(self.train_path),
            test_path=str(self.test_path),
            entity_to_id_path=str(self.entity_path),
            relation_to_id_path=str(self.relation_path),
            output_path=str(output_path),
            mode='test',
            max_test_triples=2,
            top_k=3,
            learning_rate=0.001,
            embedding_dim=self.embedding_dim,
            output_channels=self.output_channels,
            device='cpu',
            batch_size=10
        )

        # Check output file exists
        self.assertTrue(output_path.exists())

        # Load and validate JSON
        with open(output_path, 'r') as f:
            analysis = json.load(f)

        self.assertIn('num_test_triples', analysis)
        self.assertEqual(analysis['num_test_triples'], 2)
        self.assertIn('results', analysis)
        self.assertEqual(len(analysis['results']), 2)

    def test_test_mode_output_per_triple(self):
        """Test run_tracin_analysis in test mode with output_per_triple."""
        output_dir = Path(self.temp_dir) / 'per_triple_output'
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / 'results.json'

        run_tracin_analysis(
            model_path=str(self.model_path),
            train_path=str(self.train_path),
            test_path=str(self.test_path),
            entity_to_id_path=str(self.entity_path),
            relation_to_id_path=str(self.relation_path),
            output_path=str(output_path),
            mode='test',
            max_test_triples=2,
            top_k=3,
            learning_rate=0.001,
            embedding_dim=self.embedding_dim,
            output_channels=self.output_channels,
            device='cpu',
            output_per_triple=True,
            batch_size=10
        )

        # Check individual output files exist
        for i in range(2):
            triple_output = output_dir / f'tracin_test_{i}.json'
            self.assertTrue(triple_output.exists(), f"Missing {triple_output}")

            # Load and validate
            with open(triple_output, 'r') as f:
                result = json.load(f)

            self.assertIn('test_index', result)
            self.assertIn('test_head', result)
            self.assertIn('test_relation', result)
            self.assertIn('test_tail', result)
            self.assertIn('self_influence', result)
            self.assertIn('top_influences', result)

    def test_with_optional_mappings(self):
        """Test run_tracin_analysis with optional edge_map and node_name_dict."""
        # Create edge map
        edge_map_path = Path(self.temp_dir) / 'edge_map.json'
        edge_map = {}
        for i in range(self.num_relations):
            key = json.dumps({'predicate': f'predicate_{i}'})
            edge_map[key] = f'relation_{i}'

        with open(edge_map_path, 'w') as f:
            json.dump(edge_map, f)

        # Create node name dict
        node_name_path = Path(self.temp_dir) / 'node_name_dict.txt'
        with open(node_name_path, 'w') as f:
            for i in range(self.num_entities):
                f.write(f'Entity Name {i}\t{i}\n')

        output_path = Path(self.temp_dir) / 'with_names_output.json'

        run_tracin_analysis(
            model_path=str(self.model_path),
            train_path=str(self.train_path),
            test_path=str(self.test_path),
            entity_to_id_path=str(self.entity_path),
            relation_to_id_path=str(self.relation_path),
            output_path=str(output_path),
            edge_map_path=str(edge_map_path),
            node_name_dict_path=str(node_name_path),
            mode='single',
            test_triple_indices=[0],
            top_k=3,
            learning_rate=0.001,
            embedding_dim=self.embedding_dim,
            output_channels=self.output_channels,
            device='cpu',
            batch_size=10
        )

        # Check output includes names
        self.assertTrue(output_path.exists())

        with open(output_path, 'r') as f:
            results = json.load(f)

        result = results[0]
        # Check that labels and names are included
        self.assertIn('test_head_label', result)
        self.assertIn('test_head_name', result)

    def test_different_batch_sizes(self):
        """Test that different batch sizes produce consistent results."""
        output_path_bs5 = Path(self.temp_dir) / 'bs5_output.json'
        output_path_bs10 = Path(self.temp_dir) / 'bs10_output.json'

        # Run with batch_size=5
        run_tracin_analysis(
            model_path=str(self.model_path),
            train_path=str(self.train_path),
            test_path=str(self.test_path),
            entity_to_id_path=str(self.entity_path),
            relation_to_id_path=str(self.relation_path),
            output_path=str(output_path_bs5),
            mode='single',
            test_triple_indices=[0],
            top_k=5,
            learning_rate=0.001,
            embedding_dim=self.embedding_dim,
            output_channels=self.output_channels,
            device='cpu',
            batch_size=5
        )

        # Run with batch_size=10
        run_tracin_analysis(
            model_path=str(self.model_path),
            train_path=str(self.train_path),
            test_path=str(self.test_path),
            entity_to_id_path=str(self.entity_path),
            relation_to_id_path=str(self.relation_path),
            output_path=str(output_path_bs10),
            mode='single',
            test_triple_indices=[0],
            top_k=5,
            learning_rate=0.001,
            embedding_dim=self.embedding_dim,
            output_channels=self.output_channels,
            device='cpu',
            batch_size=10
        )

        # Load both results
        with open(output_path_bs5, 'r') as f:
            results_bs5 = json.load(f)

        with open(output_path_bs10, 'r') as f:
            results_bs10 = json.load(f)

        # Compare results (should be very similar)
        influences_bs5 = results_bs5[0]['influences']
        influences_bs10 = results_bs10[0]['influences']

        self.assertEqual(len(influences_bs5), len(influences_bs10))

        for inf5, inf10 in zip(influences_bs5, influences_bs10):
            self.assertEqual(inf5['train_head'], inf10['train_head'])
            self.assertEqual(inf5['train_relation'], inf10['train_relation'])
            self.assertEqual(inf5['train_tail'], inf10['train_tail'])
            # Allow small numerical differences
            self.assertAlmostEqual(inf5['influence'], inf10['influence'], places=5)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
