"""
Convert TracIn results to CSV format with IDs and labels.

This script loads TracIn analysis results and converts them to CSV format
with both entity/relation CURIEs and human-readable labels.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict

import torch
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory

from tracin import TracInAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_labels(label_path: str) -> Dict[int, str]:
    """Load entity or relation labels from file.

    Format: Either tab-separated (label\tindex) or newline-separated (label)
            or JSON file (for edge_map.json format)

    Args:
        label_path: Path to label file

    Returns:
        Dictionary mapping index to label (stores full string, extraction happens later)
    """
    import json

    labels = {}

    if not Path(label_path).exists():
        logger.warning(f"Label file not found: {label_path}")
        return labels

    # Check if it's a JSON file (like edge_map.json)
    if label_path.endswith('.json'):
        with open(label_path, 'r') as f:
            edge_map = json.load(f)
            # edge_map format: {"json_string": "predicate:0", ...}
            # We need to create a reverse mapping: predicate_id -> json_string
            for json_str, predicate_id in edge_map.items():
                # Extract the number from "predicate:X"
                if predicate_id.startswith('predicate:'):
                    idx = int(predicate_id.split(':')[1])
                    # Store the full JSON string (will be parsed by _extract_predicate_from_json)
                    labels[idx] = json_str
        return labels

    # Otherwise, load as tab-separated or newline-separated
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                # Format: label\tindex
                label, idx = parts
                labels[int(idx)] = label
            elif len(parts) == 1 and line.strip():
                # Format: just labels (use line number as index)
                idx = len(labels)
                labels[idx] = parts[0]

    return labels


def run_tracin_and_save_csv(
    model_path: str,
    train_path: str,
    test_triple_file: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
    output_csv: str,
    entity_labels_path: str = None,
    relation_labels_path: str = None,
    learning_rate: float = 0.001,
    top_k: int = 100,
    embedding_dim: int = 200,
    output_channels: int = 32,
    device: str = 'cpu',
    use_last_layers_only: bool = True,
    num_last_layers: int = 2
):
    """Run TracIn analysis and save results to CSV.

    Args:
        model_path: Path to trained model (.pt)
        train_path: Path to training triples
        test_triple_file: Path to file with single test triple
        entity_to_id_path: Path to entity_to_id.tsv
        relation_to_id_path: Path to relation_to_id.tsv
        output_csv: Path to output CSV file
        entity_labels_path: Optional path to entity labels file
        relation_labels_path: Optional path to relation labels file
        learning_rate: Learning rate used during training
        top_k: Number of top influences to return
        embedding_dim: Model embedding dimension
        output_channels: Model output channels
        device: Device to run on
        use_last_layers_only: Use last layers only for speed
        num_last_layers: Number of last layers to track
    """
    logger.info("=" * 80)
    logger.info("TracIn CSV Export")
    logger.info("=" * 80)

    # Load entity and relation mappings
    logger.info("\nLoading entity/relation mappings...")
    entity_to_id = {}
    with open(entity_to_id_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity, idx = parts
                entity_to_id[entity] = int(idx)

    relation_to_id = {}
    with open(relation_to_id_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation, idx = parts
                relation_to_id[relation] = int(idx)

    logger.info(f"  Entities: {len(entity_to_id)}")
    logger.info(f"  Relations: {len(relation_to_id)}")

    # Create reverse mappings
    id_to_entity = {v: k for k, v in entity_to_id.items()}
    id_to_relation = {v: k for k, v in relation_to_id.items()}

    # Load labels if provided
    entity_labels = None
    relation_labels = None

    if entity_labels_path:
        logger.info(f"\nLoading entity labels from {entity_labels_path}")
        entity_labels = load_labels(entity_labels_path)
        logger.info(f"  Loaded {len(entity_labels)} entity labels")

    if relation_labels_path:
        logger.info(f"\nLoading relation labels from {relation_labels_path}")
        relation_labels = load_labels(relation_labels_path)
        logger.info(f"  Loaded {len(relation_labels)} relation labels")

    # Load training triples
    logger.info(f"\nLoading training triples from {train_path}")
    train_triples = TriplesFactory.from_path(
        path=train_path,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )
    logger.info(f"  Training triples: {train_triples.num_triples}")

    # Load test triple
    logger.info(f"\nLoading test triple from {test_triple_file}")
    with open(test_triple_file, 'r') as f:
        line = f.readline().strip()
        parts = line.split('\t')
        if len(parts) != 3:
            raise ValueError(f"Invalid test triple format: {line}")

        test_h_str, test_r_str, test_t_str = parts
        test_triple = (
            entity_to_id[test_h_str],
            relation_to_id[test_r_str],
            entity_to_id[test_t_str]
        )

    logger.info(f"  Test triple: ({test_h_str}, {test_r_str}, {test_t_str})")
    logger.info(f"  Test triple (indices): {test_triple}")

    # Load model
    logger.info(f"\nLoading model from {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise ValueError("Unknown checkpoint format")

    model = ConvE(
        triples_factory=train_triples,
        embedding_dim=embedding_dim,
        output_channels=output_channels
    )
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("  Model loaded successfully")

    # Create TracIn analyzer
    logger.info(f"\nCreating TracIn analyzer...")
    logger.info(f"  Device: {device}")
    logger.info(f"  Last layers only: {use_last_layers_only}")
    if use_last_layers_only:
        logger.info(f"  Number of layers: {num_last_layers}")

    analyzer = TracInAnalyzer(
        model=model,
        device=device,
        use_last_layers_only=use_last_layers_only,
        num_last_layers=num_last_layers
    )

    # Compute influences
    logger.info(f"\nComputing TracIn influences (top-{top_k})...")
    influences = analyzer.compute_influences_for_test_triple(
        test_triple=test_triple,
        training_triples=train_triples,
        learning_rate=learning_rate,
        top_k=top_k
    )

    logger.info(f"  Computed {len(influences)} influences")

    # Save to CSV
    logger.info(f"\nSaving results to CSV: {output_csv}")
    analyzer.save_influences_to_csv(
        test_triple=test_triple,
        influences=influences,
        output_path=output_csv,
        id_to_entity=id_to_entity,
        id_to_relation=id_to_relation,
        entity_labels=entity_labels,
        relation_labels=relation_labels
    )

    logger.info("\n" + "=" * 80)
    logger.info("TracIn CSV export completed!")
    logger.info("=" * 80)
    logger.info(f"\nOutput file: {output_csv}")
    logger.info(f"Number of influences: {len(influences)}")


def main():
    parser = argparse.ArgumentParser(
        description='Run TracIn analysis and export to CSV with labels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python tracin_to_csv.py \\
      --model-path trained_model.pt \\
      --train train.txt \\
      --test-triple test_triple.txt \\
      --entity-to-id entity_to_id.tsv \\
      --relation-to-id relation_to_id.tsv \\
      --entity-labels node_name_dict.txt \\
      --relation-labels relation_labels.txt \\
      --output results/tracin_influences.csv \\
      --top-k 100

Output format (CSV):
  TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,
  TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,
  TracInScore
        """
    )

    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to training triples')
    parser.add_argument('--test-triple', type=str, required=True,
                        help='Path to file with single test triple')
    parser.add_argument('--entity-to-id', type=str, required=True,
                        help='Path to entity_to_id.tsv')
    parser.add_argument('--relation-to-id', type=str, required=True,
                        help='Path to relation_to_id.tsv')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output CSV file path')

    # Optional label files
    parser.add_argument('--entity-labels', type=str,
                        help='Path to entity labels file (optional)')
    parser.add_argument('--relation-labels', type=str,
                        help='Path to relation labels file (optional)')

    # TracIn parameters
    parser.add_argument('--top-k', type=int, default=100,
                        help='Number of top influences to return (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate used during training (default: 0.001)')

    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=200,
                        help='Model embedding dimension (default: 200)')
    parser.add_argument('--output-channels', type=int, default=32,
                        help='Model output channels (default: 32)')

    # Optimization parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run on')
    parser.add_argument('--use-last-layers-only', action='store_true', default=True,
                        help='Use last layers only for speed (default: True)')
    parser.add_argument('--num-last-layers', type=int, default=2,
                        help='Number of last layers to track (default: 2)')

    args = parser.parse_args()

    run_tracin_and_save_csv(
        model_path=args.model_path,
        train_path=args.train,
        test_triple_file=args.test_triple,
        entity_to_id_path=args.entity_to_id,
        relation_to_id_path=args.relation_to_id,
        output_csv=args.output,
        entity_labels_path=args.entity_labels,
        relation_labels_path=args.relation_labels,
        learning_rate=args.learning_rate,
        top_k=args.top_k,
        embedding_dim=args.embedding_dim,
        output_channels=args.output_channels,
        device=args.device,
        use_last_layers_only=args.use_last_layers_only,
        num_last_layers=args.num_last_layers
    )


if __name__ == '__main__':
    main()
