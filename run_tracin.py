"""
Standalone script for running TracIn analysis on a trained model.

This script provides a convenient interface for analyzing training data
influence on test predictions.
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory

from tracin import TracInAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_tracin_analysis(
    model_path: str,
    train_path: str,
    test_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
    output_path: str,
    edge_map_path: str = None,
    node_name_dict_path: str = None,
    mode: str = 'test',
    test_triple_indices: list = None,
    max_test_triples: int = None,
    top_k: int = 10,
    learning_rate: float = 0.001,
    embedding_dim: int = 200,
    output_channels: int = 32,
    device: str = 'cpu',
    output_per_triple: bool = False,
    batch_size: int = 256,
    use_last_layers_only: bool = False,
    last_layer_names: list = None,
    num_last_layers: int = 2,
    csv_output: str = None
):
    """Run TracIn analysis.

    Args:
        model_path: Path to trained model file (.pt)
        train_path: Path to training triples
        test_path: Path to test triples
        entity_to_id_path: Path to entity_to_id.tsv
        relation_to_id_path: Path to relation_to_id.tsv
        output_path: Path to save results (JSON)
        edge_map_path: Path to edge_map.json for predicate names (optional)
        node_name_dict_path: Path to node_name_dict.txt for entity names (optional)
        mode: Analysis mode ('test', 'self', or 'single')
        test_triple_indices: Specific test triple indices to analyze (for single mode)
        max_test_triples: Maximum number of test triples to analyze
        top_k: Number of top influential triples to return
        learning_rate: Learning rate used during training
        embedding_dim: Model embedding dimension
        output_channels: Model output channels
        device: Device to run on
        output_per_triple: If True, save separate file for each test triple
        batch_size: Batch size for processing training triples
        use_last_layers_only: If True, only compute gradients for last layers (MUCH faster!)
        last_layer_names: Specific layer names to track (optional, auto-detects if None)
        num_last_layers: Number of last layers to track when auto-detecting (default: 2)
        csv_output: Optional path to save results in CSV format with exact header format
    """
    logger.info("Loading model and data...")

    # Load entity and relation mappings
    entity_to_id = {}
    with open(entity_to_id_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity, idx = parts
                entity_to_id[entity] = int(idx)
    logger.info(f"Loaded {len(entity_to_id)} entities")

    relation_to_id = {}
    with open(relation_to_id_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation, idx = parts
                relation_to_id[relation] = int(idx)
    logger.info(f"Loaded {len(relation_to_id)} relations")

    # Load edge map to get predicate names
    idx_to_predicate = {}
    if edge_map_path and Path(edge_map_path).exists():
        logger.info(f"Loading edge map from {edge_map_path}")
        with open(edge_map_path, 'r') as f:
            edge_map = json.load(f)

        # Parse edge map: extract predicate from JSON key and map from predicate:N to predicate name
        # edge_map format: {"json_string": "predicate:0", ...}
        for json_key, predicate_id in edge_map.items():
            try:
                # Parse the JSON key to extract predicate
                pred_details = json.loads(json_key)
                predicate_name = pred_details.get('predicate', '')

                # Check if predicate_id is in relation_to_id keys (like "predicate:27")
                if predicate_id in relation_to_id:
                    # Get the index for this predicate_id
                    rel_idx = relation_to_id[predicate_id]
                    idx_to_predicate[rel_idx] = predicate_name
            except json.JSONDecodeError:
                continue

        logger.info(f"  Loaded {len(idx_to_predicate)} predicate names")

    # Load entity names
    idx_to_entity_name = {}
    if node_name_dict_path and Path(node_name_dict_path).exists():
        logger.info(f"Loading entity names from {node_name_dict_path}")
        with open(node_name_dict_path, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    name, idx = parts
                    idx_to_entity_name[int(idx)] = name
        logger.info(f"  Loaded {len(idx_to_entity_name)} entity names")

    # Create reverse mappings for CURIE labels
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    # Load training triples
    train_triples = TriplesFactory.from_path(
        path=train_path,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )
    logger.info(f"Loaded {train_triples.num_triples} training triples")

    # Load model
    logger.info(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('config', {}).get('embedding_dim', embedding_dim)
        output_channels = checkpoint.get('config', {}).get('output_channels', output_channels)
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        logger.error("Unknown checkpoint format")
        return

    model = ConvE(
        triples_factory=train_triples,
        embedding_dim=embedding_dim,
        output_channels=output_channels
    )
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Model loaded successfully")

    # Create analyzer
    analyzer = TracInAnalyzer(
        model=model,
        device=device,
        use_last_layers_only=use_last_layers_only,
        last_layer_names=last_layer_names,
        num_last_layers=num_last_layers
    )

    # Log which parameters are being tracked
    if use_last_layers_only:
        logger.info(f"Using LAST {num_last_layers} LAYER(S) mode for faster computation")
        if analyzer.tracked_params:
            logger.info(f"Tracking {len(analyzer.tracked_params)} parameter(s): {analyzer.tracked_params}")
            # Calculate and log reduction
            tracked_size = sum(
                param.numel() for name, param in model.named_parameters()
                if name in analyzer.tracked_params
            )
            total_size = sum(p.numel() for p in model.parameters())
            reduction = 100 * (1 - tracked_size / total_size)
            logger.info(f"Parameter reduction: {reduction:.1f}% ({tracked_size:,} / {total_size:,} params)")
    else:
        total_params = len(list(model.named_parameters()))
        logger.info(f"Using ALL PARAMETERS mode - tracking {total_params} parameters")

    # Helper function to add labels to a triple
    def add_labels_to_triple(triple_dict, prefix=''):
        """Add CURIE labels, entity names, and predicate names to a triple dictionary."""
        if prefix:
            h_key, r_key, t_key = f'{prefix}_head', f'{prefix}_relation', f'{prefix}_tail'
        else:
            h_key, r_key, t_key = 'head', 'relation', 'tail'

        h_idx, r_idx, t_idx = triple_dict[h_key], triple_dict[r_key], triple_dict[t_key]

        # Add CURIE labels
        triple_dict[f'{h_key}_label'] = id_to_entity.get(h_idx, f'UNKNOWN_{h_idx}')
        triple_dict[f'{r_key}_label'] = id_to_relation.get(r_idx, f'UNKNOWN_{r_idx}')
        triple_dict[f'{t_key}_label'] = id_to_entity.get(t_idx, f'UNKNOWN_{t_idx}')

        # Add entity names
        if idx_to_entity_name:
            triple_dict[f'{h_key}_name'] = idx_to_entity_name.get(h_idx, triple_dict[f'{h_key}_label'])
            triple_dict[f'{t_key}_name'] = idx_to_entity_name.get(t_idx, triple_dict[f'{t_key}_label'])

        # Add predicate name
        if idx_to_predicate:
            triple_dict[f'{r_key}_name'] = idx_to_predicate.get(r_idx, triple_dict[f'{r_key}_label'])

        return triple_dict

    if mode == 'self':
        # Compute self-influence for training set
        logger.info("Computing self-influences for training set...")
        logger.info("This will take a while...")

        influences = analyzer.compute_self_influence(
            training_triples=train_triples,
            learning_rate=learning_rate,
            output_path=output_path
        )

        # Print top-10
        logger.info("\nTop-10 training examples by self-influence:")
        for i, inf in enumerate(influences[:10]):
            logger.info(f"  {i+1}. ({inf['head']}, {inf['relation']}, {inf['tail']})")
            logger.info(f"     Self-influence: {inf['self_influence']:.6f}")

    elif mode == 'test':
        # Analyze test set
        test_triples = TriplesFactory.from_path(
            path=test_path,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id
        )
        logger.info(f"Loaded {test_triples.num_triples} test triples")

        logger.info("Analyzing influence on test predictions...")
        logger.info("WARNING: This is computationally expensive!")

        if max_test_triples:
            logger.info(f"Limiting analysis to first {max_test_triples} test triples")

        if output_per_triple:
            # Save separate file for each test triple
            logger.info("Will save separate file for each test triple")
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            test_triple_list = [(int(h), int(r), int(t)) for h, r, t in test_triples.mapped_triples]
            if max_test_triples:
                test_triple_list = test_triple_list[:max_test_triples]

            for test_idx, test_triple in enumerate(test_triple_list):
                logger.info(f"\n[{test_idx+1}/{len(test_triple_list)}] Analyzing test triple: {test_triple}")

                # Compute influences from training triples
                influences = analyzer.compute_influences_for_test_triple(
                    test_triple=test_triple,
                    training_triples=train_triples,
                    learning_rate=learning_rate,
                    top_k=top_k,
                    batch_size=batch_size
                )

                # Compute self-influence for the test triple
                logger.info(f"  Computing self-influence...")
                test_h, test_r, test_t = test_triple
                grad = analyzer.compute_gradient(test_h, test_r, test_t, label=1.0)

                # Compute squared L2 norm of gradient (self-influence)
                self_influence = 0.0
                for name in grad:
                    grad_flat = grad[name].flatten()
                    self_influence += torch.dot(grad_flat, grad_flat).item()
                self_influence *= learning_rate

                # Add labels to test triple
                result = {
                    'test_index': test_idx,
                    'test_head': test_triple[0],
                    'test_relation': test_triple[1],
                    'test_tail': test_triple[2]
                }
                result = add_labels_to_triple(result, prefix='test')

                # Add labels to training influences
                for inf in influences:
                    add_labels_to_triple(inf, prefix='train')

                result['self_influence'] = self_influence
                result['top_influences'] = influences
                result['top_k'] = top_k
                result['learning_rate'] = learning_rate

                # Save to separate JSON file
                triple_output = output_dir / f"tracin_test_{test_idx}.json"
                with open(triple_output, 'w') as f:
                    json.dump(result, f, indent=2)

                logger.info(f"  Self-influence: {self_influence:.6f}")
                logger.info(f"  Saved JSON to {triple_output}")

                # Save CSV if requested
                if csv_output:
                    csv_dir = Path(csv_output).parent
                    csv_stem = Path(csv_output).stem
                    csv_ext = Path(csv_output).suffix
                    csv_file = csv_dir / f"{csv_stem}_test_{test_idx}{csv_ext}"

                    logger.info(f"  Saving CSV to {csv_file}")

                    # Prepare relation labels from idx_to_predicate if available
                    relation_labels = {}
                    if idx_to_predicate:
                        for rel_idx, predicate_name in idx_to_predicate.items():
                            relation_labels[rel_idx] = json.dumps({"predicate": predicate_name})

                    analyzer.save_influences_to_csv(
                        test_triple=test_triple,
                        influences=influences,
                        output_path=str(csv_file),
                        id_to_entity=id_to_entity,
                        id_to_relation=id_to_relation,
                        entity_labels=idx_to_entity_name,
                        relation_labels=relation_labels if relation_labels else None
                    )

            logger.info(f"\nAnalyzed {len(test_triple_list)} test triples")
            logger.info(f"Results saved to {output_dir}/tracin_test_*.json")

        else:
            # Save all results to single file
            analysis = analyzer.analyze_test_set(
                test_triples=test_triples,
                training_triples=train_triples,
                learning_rate=learning_rate,
                top_k=top_k,
                max_test_triples=max_test_triples,
                output_path=output_path,
                batch_size=batch_size
            )

            logger.info(f"\nAnalyzed {analysis['num_test_triples']} test triples")
            logger.info(f"Results saved to {output_path}")

    elif mode == 'single':
        # Analyze specific test triples
        test_triples = TriplesFactory.from_path(
            path=test_path,
            entity_to_id=entity_to_id,
            relation_to_id=relation_to_id
        )

        if test_triple_indices is None:
            test_triple_indices = [0]  # Default to first test triple

        results = []
        for idx in test_triple_indices:
            if idx >= test_triples.num_triples:
                logger.warning(f"Test triple index {idx} out of range (max: {test_triples.num_triples-1})")
                continue

            test_triple = tuple(int(x) for x in test_triples.mapped_triples[idx])
            logger.info(f"\nAnalyzing test triple {idx}: {test_triple}")

            influences = analyzer.compute_influences_for_test_triple(
                test_triple=test_triple,
                training_triples=train_triples,
                learning_rate=learning_rate,
                top_k=top_k,
                batch_size=batch_size
            )

            # Compute self-influence for the test triple
            logger.info(f"Computing self-influence...")
            test_h, test_r, test_t = test_triple
            grad = analyzer.compute_gradient(test_h, test_r, test_t, label=1.0)

            # Compute squared L2 norm of gradient (self-influence)
            self_influence = 0.0
            for name in grad:
                grad_flat = grad[name].flatten()
                self_influence += torch.dot(grad_flat, grad_flat).item()
            self_influence *= learning_rate

            logger.info(f"Self-influence: {self_influence:.6f}")
            logger.info(f"Top-{min(5, top_k)} influential training triples:")
            for i, inf in enumerate(influences[:5]):
                logger.info(f"  {i+1}. ({inf['train_head']}, {inf['train_relation']}, {inf['train_tail']})")
                logger.info(f"     Influence: {inf['influence']:.6f}")

            # Create result with labels
            result_dict = {
                'test_triple': test_triple,
                'test_triple_index': idx,
                'test_head': test_triple[0],
                'test_relation': test_triple[1],
                'test_tail': test_triple[2],
                'self_influence': self_influence
            }
            result_dict = add_labels_to_triple(result_dict, prefix='test')

            # Add labels to training influences
            for inf in influences:
                add_labels_to_triple(inf, prefix='train')

            result_dict['influences'] = influences
            results.append(result_dict)

            # Save CSV output if requested
            if csv_output:
                # Determine CSV filename for this test triple
                if len(test_triple_indices) == 1:
                    csv_file = csv_output
                else:
                    # Multiple test triples - create separate CSV for each
                    csv_dir = Path(csv_output).parent
                    csv_stem = Path(csv_output).stem
                    csv_ext = Path(csv_output).suffix
                    csv_file = csv_dir / f"{csv_stem}_test_{idx}{csv_ext}"

                logger.info(f"Saving CSV output to {csv_file}")

                # Prepare relation labels from idx_to_predicate if available
                relation_labels = {}
                if idx_to_predicate:
                    # Convert idx_to_predicate to JSON strings for extraction
                    for rel_idx, predicate_name in idx_to_predicate.items():
                        # Create a minimal JSON string with just the predicate field
                        relation_labels[rel_idx] = json.dumps({"predicate": predicate_name})

                analyzer.save_influences_to_csv(
                    test_triple=test_triple,
                    influences=influences,
                    output_path=str(csv_file),
                    id_to_entity=id_to_entity,
                    id_to_relation=id_to_relation,
                    entity_labels=idx_to_entity_name,
                    relation_labels=relation_labels if relation_labels else None
                )

        # Save JSON results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nJSON results saved to {output_path}")

    else:
        raise ValueError(f"Unknown mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run TracIn analysis on trained ConvE model'
    )

    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model file (.pt)'
    )
    parser.add_argument(
        '--train', type=str, required=True,
        help='Path to training triples file'
    )
    parser.add_argument(
        '--test', type=str,
        help='Path to test triples file (required for test/single modes)'
    )
    parser.add_argument(
        '--entity-to-id', type=str, required=True,
        help='Path to entity_to_id.tsv'
    )
    parser.add_argument(
        '--relation-to-id', type=str, required=True,
        help='Path to relation_to_id.tsv'
    )
    parser.add_argument(
        '--edge-map', type=str,
        help='Path to edge_map.json for predicate names (optional)'
    )
    parser.add_argument(
        '--node-name-dict', type=str,
        help='Path to node_name_dict.txt for entity names (optional)'
    )
    parser.add_argument(
        '--output', '-o', type=str, required=True,
        help='Output path for results (JSON)'
    )
    parser.add_argument(
        '--csv-output', type=str,
        help='Optional: Output path for CSV format results with labels'
    )

    parser.add_argument(
        '--mode', type=str, default='test',
        choices=['test', 'self', 'single'],
        help='Analysis mode: test (full test set), self (training self-influence), single (specific triples)'
    )

    parser.add_argument(
        '--test-indices', type=int, nargs='+',
        help='Test triple indices to analyze (for single mode)'
    )
    parser.add_argument(
        '--max-test-triples', type=int,
        help='Maximum number of test triples to analyze (for speed)'
    )
    parser.add_argument(
        '--top-k', type=int, default=None,
        help='Number of top influential triples to return per test triple (default: None = all influences)'
    )
    parser.add_argument(
        '--learning-rate', type=float, default=0.001,
        help='Learning rate used during training'
    )
    parser.add_argument(
        '--embedding-dim', type=int, default=200,
        help='Model embedding dimension'
    )
    parser.add_argument(
        '--output-channels', type=int, default=32,
        help='Model output channels'
    )
    parser.add_argument(
        '--device', type=str, default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run on'
    )
    parser.add_argument(
        '--output-per-triple', action='store_true',
        help='Save separate JSON file for each test triple (test mode only)'
    )
    parser.add_argument(
        '--batch-size', type=int, default=256,
        help='Batch size for processing training triples (larger = faster on GPU)'
    )
    parser.add_argument(
        '--use-last-layers-only', action='store_true',
        help='Only compute gradients for last layers (MUCH faster, following original TracIn paper)'
    )
    parser.add_argument(
        '--num-last-layers', type=int, default=2,
        help='Number of last layers to track (default: 2). Options: 1 (fastest), 2-3 (recommended), 5+ (slower)'
    )
    parser.add_argument(
        '--last-layer-names', type=str, nargs='+',
        help='Specific layer names to track (optional, auto-detects if not provided)'
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Validate arguments
    if args.mode in ['test', 'single'] and not args.test:
        raise ValueError(f"--test is required for mode '{args.mode}'")

    if args.mode == 'single' and not args.test_indices:
        logger.info("No test indices specified, will analyze first test triple")

    # Run analysis
    run_tracin_analysis(
        model_path=args.model_path,
        train_path=args.train,
        test_path=args.test,
        entity_to_id_path=args.entity_to_id,
        relation_to_id_path=args.relation_to_id,
        output_path=args.output,
        edge_map_path=args.edge_map,
        node_name_dict_path=args.node_name_dict,
        mode=args.mode,
        test_triple_indices=args.test_indices,
        max_test_triples=args.max_test_triples,
        top_k=args.top_k,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        output_channels=args.output_channels,
        device=args.device,
        output_per_triple=args.output_per_triple,
        batch_size=args.batch_size,
        use_last_layers_only=args.use_last_layers_only,
        last_layer_names=args.last_layer_names,
        csv_output=args.csv_output
    )

    logger.info("\nTracIn analysis completed!")


if __name__ == '__main__':
    main()
