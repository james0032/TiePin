#!/usr/bin/env python3
"""
Score test triples without computing rankings.

This script gets ConvE scores for test triples WITHOUT computing expensive rankings.
Use this when you only need the model's confidence score for each triple.

Usage:
    python score_only.py --model-dir output/trained_model --test data/test.txt --output test_scores.json
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory

from evaluate import DetailedEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_node_names(node_name_dict_path: str) -> Dict[int, str]:
    """Load entity index to name mapping from node_name_dict.txt.

    Args:
        node_name_dict_path: Path to node_name_dict.txt

    Returns:
        Dictionary mapping entity index to name
    """
    idx_to_name = {}

    if not Path(node_name_dict_path).exists():
        logger.warning(f"Node name dictionary not found: {node_name_dict_path}")
        return idx_to_name

    with open(node_name_dict_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 2:
                name, idx = parts
                idx_to_name[int(idx)] = name

    logger.info(f"Loaded {len(idx_to_name)} entity names")
    return idx_to_name


def main():
    parser = argparse.ArgumentParser(
        description='Score test triples without computing rankings (fast!)'
    )
    parser.add_argument(
        '--model-dir',
        type=str,
        required=True,
        help='Directory containing trained_model.pkl'
    )
    parser.add_argument(
        '--test',
        type=str,
        required=True,
        help='Path to test.txt'
    )
    parser.add_argument(
        '--entity-to-id',
        type=str,
        help='Path to entity_to_id.tsv (if not in same dir as test.txt)'
    )
    parser.add_argument(
        '--relation-to-id',
        type=str,
        help='Path to relation_to_id.tsv (if not in same dir as test.txt)'
    )
    parser.add_argument(
        '--node-name-dict',
        type=str,
        help='Path to node_name_dict.txt for entity names (if not in same dir as test.txt)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='test_scores.json',
        help='Output path for scores (default: test_scores.json)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (cuda/cpu, default: auto-detect)'
    )
    parser.add_argument(
        '--use-sigmoid',
        action='store_true',
        help='Apply sigmoid to convert scores to probabilities [0, 1] (default: False, returns raw logits)'
    )
    parser.add_argument(
        '--top-n',
        type=int,
        default=None,
        help='Output top N highest scoring triples to a separate TSV file (format: head\\tpredicate\\ttail, no header)'
    )

    args = parser.parse_args()

    # Determine model path
    model_path = Path(args.model_dir)

    # If args.model_dir is a file, use it directly
    if model_path.is_file():
        model_file = model_path
    # If it's a directory, look for best_model.pt or final_model.pt
    elif model_path.is_dir():
        if (model_path / 'best_model.pt').exists():
            model_file = model_path / 'best_model.pt'
            logger.info(f"Found best_model.pt in {args.model_dir}")
        elif (model_path / 'final_model.pt').exists():
            model_file = model_path / 'final_model.pt'
            logger.info(f"Found final_model.pt in {args.model_dir}")
        elif (model_path / 'trained_model.pkl').exists():
            model_file = model_path / 'trained_model.pkl'
            logger.info(f"Found trained_model.pkl in {args.model_dir}")
        else:
            logger.error(f"No model file found in {args.model_dir}")
            logger.error(f"Looking for: best_model.pt, final_model.pt, or trained_model.pkl")
            return
    else:
        logger.error(f"Model path does not exist: {args.model_dir}")
        return

    logger.info(f"Loading model from {model_file}...")

    # Load test triples
    logger.info(f"Loading test triples from {args.test}...")
    test_dir = Path(args.test).parent

    # Determine entity/relation mapping paths
    if args.entity_to_id:
        entity_map_path = args.entity_to_id
    else:
        entity_map_path = test_dir / 'entity_to_id.tsv'

    if args.relation_to_id:
        relation_map_path = args.relation_to_id
    else:
        relation_map_path = test_dir / 'relation_to_id.tsv'

    if args.node_name_dict:
        node_name_dict_path = args.node_name_dict
    else:
        node_name_dict_path = test_dir / 'node_name_dict.txt'

    # Load entity and relation mappings from TSV files
    logger.info(f"Loading entity mappings from {entity_map_path}")
    entity_to_id = {}
    with open(entity_map_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity, idx = parts
                entity_to_id[entity] = int(idx)  # Convert to int!
    logger.info(f"  Loaded {len(entity_to_id)} entities")

    logger.info(f"Loading relation mappings from {relation_map_path}")
    relation_to_id = {}
    with open(relation_map_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation, idx = parts
                relation_to_id[relation] = int(idx)  # Convert to int!
    logger.info(f"  Loaded {len(relation_to_id)} relations")

    # Load test triples using the SAME entity/relation mappings from training
    test_triples = TriplesFactory.from_path(
        path=args.test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )
    logger.info(f"Loaded {test_triples.num_triples} test triples")
    logger.info(f"Number of entities: {test_triples.num_entities}")
    logger.info(f"Number of relations: {test_triples.num_relations}")

    # Load entity names
    idx_to_name = load_node_names(str(node_name_dict_path))

    # Load the model checkpoint
    checkpoint = torch.load(model_file, map_location='cpu')

    # Debug: what type is the checkpoint?
    logger.info(f"Checkpoint type: {type(checkpoint)}")
    if isinstance(checkpoint, dict):
        logger.info(f"Checkpoint top-level keys: {list(checkpoint.keys())[:20]}")

    # Determine if this is a state_dict only or a full checkpoint
    state_dict = None
    embedding_dim = None
    output_channels = None
    embedding_height = None
    embedding_width = None

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint with metadata
        state_dict = checkpoint['model_state_dict']
        logger.info(f"Extracted model_state_dict with {len(state_dict)} keys")

        # Try to get config if it exists
        config = checkpoint.get('config', {})
        if config:
            embedding_dim = config.get('embedding_dim', None)
            output_channels = config.get('output_channels', None)
            embedding_height = config.get('embedding_height', None)
            embedding_width = config.get('embedding_width', None)
            logger.info(f"Loaded checkpoint with config: embedding_dim={embedding_dim}, output_channels={output_channels}")
        else:
            logger.info("No config found in checkpoint, will infer from state_dict...")
    elif isinstance(checkpoint, dict):
        # Just a state_dict - infer parameters from tensor shapes
        state_dict = checkpoint
        logger.info("Checkpoint is a plain state_dict")
    else:
        logger.error(f"Unknown checkpoint format")
        return

    # If we don't have model parameters yet, infer them from state_dict
    if embedding_dim is None or output_channels is None:
        logger.info("=" * 80)
        logger.info("Inferring model parameters from state_dict...")
        logger.info("=" * 80)

        # Debug: print ALL keys to see what we're working with
        logger.info(f"Total keys in state_dict: {len(state_dict)}")
        logger.info("All state_dict keys:")
        for key in sorted(state_dict.keys()):
            value = state_dict[key]
            if hasattr(value, 'shape'):
                logger.info(f"  {key}: {value.shape}")
            else:
                logger.info(f"  {key}: {type(value)}")

        # Try different possible key names for entity embeddings
        entity_embedding_keys = [
            'entity_representations.0._embeddings.weight',
            'entity_embeddings.weight',
            'entity_embedding.weight',
        ]

        for key in entity_embedding_keys:
            if key in state_dict:
                value = state_dict[key]
                if hasattr(value, 'shape') and len(value.shape) >= 2:
                    embedding_dim = value.shape[1]
                    logger.info(f"  ✓ Inferred embedding_dim={embedding_dim} from {key}")
                    break

        # If still not found, search for any key containing 'entity' and 'embedding'
        if embedding_dim is None:
            for key in state_dict.keys():
                if 'entity' in key.lower() and 'embedding' in key.lower() and 'weight' in key:
                    value = state_dict[key]
                    if hasattr(value, 'shape') and len(value.shape) >= 2:
                        embedding_dim = value.shape[1]
                        logger.info(f"  ✓ Inferred embedding_dim={embedding_dim} from {key}")
                        break

        # Infer output_channels from convolution layer
        # Check hr2d.2.weight which has shape [output_channels, 1, kernel_h, kernel_w]
        if 'interaction.hr2d.2.weight' in state_dict:
            value = state_dict['interaction.hr2d.2.weight']
            if hasattr(value, 'shape') and len(value.shape) >= 1:
                output_channels = value.shape[0]
                logger.info(f"  ✓ Inferred output_channels={output_channels} from interaction.hr2d.2.weight")
        elif 'interaction.hr1d.0.weight' in state_dict:
            value = state_dict['interaction.hr1d.0.weight']
            if hasattr(value, 'shape') and len(value.shape) >= 1:
                output_channels = value.shape[0]
                logger.info(f"  ✓ Inferred output_channels={output_channels} from interaction.hr1d.0.weight")

        # Try to infer embedding_height and embedding_width from hr1d layer size
        # In ConvE, entity and relation embeddings are STACKED (not concatenated)
        # So the reshaped input is [embedding_height, embedding_width * 2]
        # where embedding_height * embedding_width = embedding_dim
        # The hr1d.0.weight has shape [output_dim, input_size]
        # input_size = (h - kernel_h + 1) * (w*2 - kernel_w + 1) * output_channels
        if 'interaction.hr1d.0.weight' in state_dict and embedding_dim and output_channels:
            value = state_dict['interaction.hr1d.0.weight']
            if hasattr(value, 'shape') and len(value.shape) >= 2:
                hr1d_input_size = value.shape[1]
                logger.info(f"  hr1d input size: {hr1d_input_size}")

            # Assuming kernel size 3x3 (default in ConvE)
            kernel_h, kernel_w = 3, 3
            # hr1d_input_size = (h - 2) * (w*2 - 2) * output_channels
            conv_output_size = hr1d_input_size // output_channels
            logger.info(f"  Convolution output size after conv: {conv_output_size}")

            # Try to find h, w such that h*w = embedding_dim and (h-2)*(w*2-2) = conv_output_size
            logger.info(f"  Searching for h, w where:")
            logger.info(f"    h * w = {embedding_dim}")
            logger.info(f"    (h-2) * (w*2-2) = {conv_output_size}")

            for h in range(3, 100):  # h must be at least 3 for kernel 3x3
                if embedding_dim % h == 0:
                    w = embedding_dim // h
                    if (h - 2) * (w * 2 - 2) == conv_output_size:
                        embedding_height = h
                        embedding_width = w
                        logger.info(f"  ✓ Inferred embedding_height={h}, embedding_width={w}")
                        logger.info(f"    Verification: {h}*{w}={h*w} (embedding_dim={embedding_dim})")
                        logger.info(f"    Verification: ({h}-2)*({w}*2-2)=({h-2})*{w*2-2}={conv_output_size}")
                        break

            if embedding_height is None:
                logger.warning(f"Could not infer embedding dimensions from hr1d size")
                logger.warning(f"  Need: h*w={embedding_dim} and (h-2)*(w*2-2)={conv_output_size}")

        # Fallback: try common configurations
        if embedding_height is None and embedding_dim:
            logger.info("  Using fallback common configurations...")
            common_configs = [
                (10, 20),  # Common for embedding_dim=200
                (20, 10),  # Alternative for embedding_dim=200
                (8, 4),    # Common for embedding_dim=32
                (4, 8),    # Alternative for embedding_dim=32
                (16, 16),  # For embedding_dim=256
                (14, 14),  # For embedding_dim=196
            ]
            for h, w in common_configs:
                if h * w == embedding_dim:
                    embedding_height = h
                    embedding_width = w
                    logger.info(f"  ✓ Using fallback: embedding_height={h}, embedding_width={w}")
                    break

    # Validate that we have the required parameters
    if embedding_dim is None or output_channels is None:
        logger.error("Could not determine model architecture parameters")
        logger.error(f"embedding_dim={embedding_dim}, output_channels={output_channels}")
        logger.error("Available keys in checkpoint:")
        for key in list(state_dict.keys())[:10]:
            value = state_dict[key]
            if hasattr(value, 'shape'):
                logger.error(f"  {key}: {value.shape}")
            else:
                logger.error(f"  {key}: {type(value)}")
        return

    # Create model with correct architecture
    logger.info(f"Creating ConvE model with:")
    logger.info(f"  embedding_dim={embedding_dim}")
    logger.info(f"  output_channels={output_channels}")

    model_kwargs = {
        'embedding_dim': embedding_dim,
        'output_channels': output_channels,
    }

    # Add embedding dimensions if we have them
    if embedding_height and embedding_width:
        model_kwargs['embedding_height'] = embedding_height
        model_kwargs['embedding_width'] = embedding_width
        logger.info(f"  embedding_height={embedding_height}")
        logger.info(f"  embedding_width={embedding_width}")

    model = ConvE(
        triples_factory=test_triples,
        **model_kwargs
    )

    # Load state dict into model
    try:
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded successfully")
    except RuntimeError as e:
        logger.error(f"Failed to load state_dict with current configuration: {e}")
        logger.info("Trying to find correct embedding_height and embedding_width by testing configurations...")

        # Extract the expected size from the error message
        import re
        match = re.search(r'copying a param with shape torch\.Size\(\[(\d+), (\d+)\]\)', str(e))
        if match:
            expected_hr1d_out = int(match.group(1))
            expected_hr1d_in = int(match.group(2))
            logger.info(f"  Checkpoint expects hr1d input size: {expected_hr1d_in}")

            # Calculate what conv output size this corresponds to
            # hr1d_input = conv_output * output_channels
            expected_conv_out = expected_hr1d_in // output_channels
            logger.info(f"  Expected conv output size: {expected_conv_out}")

            # Try all possible h, w combinations for this embedding_dim
            found = False
            for h in range(3, 100):
                if embedding_dim % h == 0:
                    w = embedding_dim // h
                    # Test this configuration
                    test_kwargs = {
                        'embedding_dim': embedding_dim,
                        'output_channels': output_channels,
                        'embedding_height': h,
                        'embedding_width': w,
                    }

                    test_model = ConvE(triples_factory=test_triples, **test_kwargs)

                    # Check if the hr1d layer size matches
                    if 'interaction.hr1d.0.weight' in test_model.state_dict():
                        value = test_model.state_dict()['interaction.hr1d.0.weight']
                        if hasattr(value, 'shape') and len(value.shape) >= 2:
                            test_hr1d_size = value.shape[1]
                            if test_hr1d_size == expected_hr1d_in:
                                logger.info(f"  ✓ Found matching configuration: h={h}, w={w}")
                                logger.info(f"    hr1d input size: {test_hr1d_size}")

                            # Use this configuration
                            model = test_model
                            model.load_state_dict(state_dict)
                            model.eval()
                            found = True
                            break

            if not found:
                logger.error("Could not find matching embedding configuration")
                return
        else:
            logger.error("Could not parse error message to find expected sizes")
            return

    # Create evaluator
    evaluator = DetailedEvaluator(
        model=model,
        filter_triples=False,  # No filtering needed for score-only
        device=args.device,
        use_sigmoid=args.use_sigmoid
    )

    score_type = "probabilities (0-1)" if args.use_sigmoid else "logits (can be negative)"
    logger.info(f"Scoring mode: {score_type}")

    # Score all test triples WITHOUT computing rankings
    logger.info("Scoring test triples (this will be FAST - no ranking computation)...")
    results = evaluator.score_dataset(
        test_triples=test_triples,
        output_path=None,  # We'll save ourselves with entity names
        include_labels=True  # Get CURIE labels
    )

    # Add entity names to results
    if idx_to_name:
        logger.info("Adding entity names to results...")
        for result in results:
            result['head_name'] = idx_to_name.get(result['head_id'], result.get('head_label', 'UNKNOWN'))
            result['tail_name'] = idx_to_name.get(result['tail_id'], result.get('tail_label', 'UNKNOWN'))
    else:
        # If no node_name_dict, use CURIE as name
        logger.warning("No node_name_dict found - using CURIE IDs as names")
        for result in results:
            result['head_name'] = result.get('head_label', 'UNKNOWN')
            result['tail_name'] = result.get('tail_label', 'UNKNOWN')

    # Save results with entity names
    logger.info(f"Saving results to {args.output}...")

    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    # Save as CSV with all columns
    csv_path = args.output.replace('.json', '.csv')
    df = pd.DataFrame(results)

    # Define column order: index, CURIE, name for head/tail, plus relation and score
    column_order = [
        'head_id', 'head_label', 'head_name',
        'relation_id', 'relation_label',
        'tail_id', 'tail_label', 'tail_name',
        'score'
    ]

    # Only keep columns that exist
    column_order = [col for col in column_order if col in df.columns]
    df = df[column_order]
    df.to_csv(csv_path, index=False)

    # Create ranked version sorted by score (descending)
    ranked_csv_path = args.output.replace('.json', '_ranked.csv')
    df_ranked = df.sort_values('score', ascending=False).reset_index(drop=True)
    # Add rank column
    df_ranked.insert(0, 'rank', range(1, len(df_ranked) + 1))
    df_ranked.to_csv(ranked_csv_path, index=False)

    # Create ranked JSON
    ranked_json_path = args.output.replace('.json', '_ranked.json')
    ranked_results = df_ranked.to_dict('records')
    with open(ranked_json_path, 'w') as f:
        json.dump(ranked_results, f, indent=2)

    # Create top N TSV file if requested
    top_n_path = None
    if args.top_n is not None:
        top_n_path = args.output.replace('.json', f'_top{args.top_n}.txt')
        top_n_df = df_ranked.head(args.top_n)

        # Write in TSV format: head_label, relation_label, tail_label (no header)
        with open(top_n_path, 'w') as f:
            for _, row in top_n_df.iterrows():
                f.write(f"{row['head_label']}\t{row['relation_label']}\t{row['tail_label']}\n")

        logger.info(f"✓ Top {args.top_n} triples saved to: {top_n_path}")

    logger.info(f"✓ Done! Scored {len(results)} triples")
    logger.info(f"✓ Results saved to:")
    logger.info(f"  - Original order JSON: {args.output}")
    logger.info(f"  - Original order CSV: {csv_path}")
    logger.info(f"  - Ranked by score JSON: {ranked_json_path}")
    logger.info(f"  - Ranked by score CSV: {ranked_csv_path}")
    if top_n_path:
        logger.info(f"  - Top {args.top_n} TSV (no header): {top_n_path}")

    # Show some example scores from original order
    logger.info("\nExample scores (first 5 triples in original order):")
    for i, result in enumerate(results[:5]):
        if 'head_name' in result:
            logger.info(f"  {i+1}. {result['head_name']} --[{result['relation_label']}]--> {result['tail_name']}")
            logger.info(f"      Score: {result['score']:.4f}")
            logger.info(f"      IDs: h={result['head_id']} ({result['head_label']}), r={result['relation_id']}, t={result['tail_id']} ({result['tail_label']})")
        elif 'head_label' in result:
            logger.info(f"  {i+1}. {result['head_label']} --[{result['relation_label']}]--> {result['tail_label']}")
            logger.info(f"      Score: {result['score']:.4f} (IDs: h={result['head_id']}, r={result['relation_id']}, t={result['tail_id']})")
        else:
            logger.info(f"  {i+1}. (h={result['head_id']}, r={result['relation_id']}, t={result['tail_id']}) → score={result['score']:.4f}")

    # Show top 5 highest scoring triples
    logger.info("\nTop 5 highest scoring triples:")
    for i, result in enumerate(ranked_results[:5]):
        if 'head_name' in result:
            logger.info(f"  {result['rank']}. {result['head_name']} --[{result['relation_label']}]--> {result['tail_name']}")
            logger.info(f"      Score: {result['score']:.4f}")
        elif 'head_label' in result:
            logger.info(f"  {result['rank']}. {result['head_label']} --[{result['relation_label']}]--> {result['tail_label']}")
            logger.info(f"      Score: {result['score']:.4f}")
        else:
            logger.info(f"  {result['rank']}. (h={result['head_id']}, r={result['relation_id']}, t={result['tail_id']}) → score={result['score']:.4f}")


if __name__ == '__main__':
    main()
