#!/usr/bin/env python3
"""
Fast TracIn runner with optimization strategies.

This is a convenience wrapper around run_tracin.py that uses optimized
configurations for faster analysis on large datasets.

Usage:
    # Ultra-fast mode (recommended for exploration)
    python run_tracin_fast.py \\
        --model-path model.pt \\
        --train train.txt \\
        --test test.txt \\
        --entity-to-id entity_to_id.tsv \\
        --relation-to-id relation_to_id.tsv \\
        --output results.json \\
        --sample-rate 0.1 \\
        --mode single

    # With projection for even more speed
    python run_tracin_fast.py \\
        --model-path model.pt \\
        ... \\
        --use-projection \\
        --projection-dim 256 \\
        --sample-rate 0.1
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory

from tracin_optimized import TracInAnalyzerOptimized

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description='Fast TracIn analysis with optimizations',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ultra-fast mode with 10% sampling
  python run_tracin_fast.py --model-path model.pt --train train.txt \\
      --test test.txt --entity-to-id entity_to_id.tsv \\
      --relation-to-id relation_to_id.tsv --output results.json \\
      --sample-rate 0.1 --mode single

  # Maximum speed with projection
  python run_tracin_fast.py --model-path model.pt ... \\
      --use-projection --projection-dim 128 --sample-rate 0.05

  # Balanced mode with stratification
  python run_tracin_fast.py --model-path model.pt ... \\
      --sample-rate 0.2 --stratify-by relation
        """
    )

    # Required arguments
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to training triples')
    parser.add_argument('--test', type=str, required=True,
                        help='Path to test triples')
    parser.add_argument('--entity-to-id', type=str, required=True,
                        help='Path to entity_to_id.tsv')
    parser.add_argument('--relation-to-id', type=str, required=True,
                        help='Path to relation_to_id.tsv')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output path for results (JSON)')

    # Mode
    parser.add_argument('--mode', type=str, default='single',
                        choices=['single', 'test'],
                        help='Analysis mode')
    parser.add_argument('--test-indices', type=int, nargs='+',
                        help='Test triple indices (for single mode)')
    parser.add_argument('--max-test-triples', type=int,
                        help='Max test triples to analyze (for test mode)')

    # Model parameters
    parser.add_argument('--embedding-dim', type=int, default=200,
                        help='Model embedding dimension')
    parser.add_argument('--output-channels', type=int, default=32,
                        help='Model output channels')

    # Optimization parameters
    parser.add_argument('--sample-rate', type=float, default=0.1,
                        help='Fraction of training data to sample (0.01-1.0, default: 0.1)')
    parser.add_argument('--stratify-by', type=str, choices=['relation', 'head', 'tail', None],
                        help='Stratification strategy for sampling')
    parser.add_argument('--use-projection', action='store_true',
                        help='Use random projection for gradients (faster)')
    parser.add_argument('--projection-dim', type=int, default=256,
                        help='Target dimension for random projection (default: 256)')
    parser.add_argument('--projection-type', type=str, default='gaussian',
                        choices=['gaussian', 'sparse'],
                        help='Type of random projection')

    # Standard parameters
    parser.add_argument('--top-k', type=int, default=10,
                        help='Number of top influences to return')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate used during training')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        choices=['cpu', 'cuda'],
                        help='Device to run on')
    parser.add_argument('--num-last-layers', type=int, default=2,
                        help='Number of last layers to track (default: 2)')
    parser.add_argument('--seed', type=int,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Validate
    if args.sample_rate <= 0 or args.sample_rate > 1:
        parser.error("--sample-rate must be in range (0, 1]")

    # Show configuration
    logger.info("=" * 80)
    logger.info("Fast TracIn Analysis Configuration")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Optimizations:")
    logger.info(f"  • Last {args.num_last_layers} layer(s) only: Enabled")
    logger.info(f"  • Sampling rate: {args.sample_rate*100:.1f}% of training data")
    if args.stratify_by:
        logger.info(f"  • Stratification: By {args.stratify_by}")
    if args.use_projection:
        logger.info(f"  • Random projection: Enabled (dim={args.projection_dim}, type={args.projection_type})")
    else:
        logger.info(f"  • Random projection: Disabled")

    # Estimate speedup
    speedup = 50  # From last-layers
    speedup *= (1.0 / args.sample_rate)  # From sampling
    if args.use_projection:
        speedup *= 10  # From projection
    logger.info(f"\nEstimated speedup vs baseline: ~{speedup:.0f}x 🚀")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading data...")

    entity_to_id = {}
    with open(args.entity_to_id, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity, idx = parts
                entity_to_id[entity] = int(idx)
    logger.info(f"Loaded {len(entity_to_id)} entities")

    relation_to_id = {}
    with open(args.relation_to_id, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                relation, idx = parts
                relation_to_id[relation] = int(idx)
    logger.info(f"Loaded {len(relation_to_id)} relations")

    # Load training triples
    train_triples = TriplesFactory.from_path(
        path=args.train,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )
    logger.info(f"Loaded {train_triples.num_triples} training triples")

    # Load model
    logger.info(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        logger.error("Unknown checkpoint format")
        return

    model = ConvE(
        triples_factory=train_triples,
        embedding_dim=args.embedding_dim,
        output_channels=args.output_channels
    )
    model.load_state_dict(state_dict)
    model.eval()
    logger.info("Model loaded successfully")

    # Create optimized analyzer
    analyzer = TracInAnalyzerOptimized(
        model=model,
        device=args.device,
        use_last_layers_only=True,
        num_last_layers=args.num_last_layers,
        use_projection=args.use_projection,
        projection_dim=args.projection_dim,
        projection_type=args.projection_type
    )

    # Load test triples
    test_triples = TriplesFactory.from_path(
        path=args.test,
        entity_to_id=entity_to_id,
        relation_to_id=relation_to_id
    )
    logger.info(f"Loaded {test_triples.num_triples} test triples")

    # Run analysis
    if args.mode == 'single':
        # Analyze specific test triples
        if args.test_indices is None:
            test_indices = [0]
        else:
            test_indices = args.test_indices

        results = []
        for idx in test_indices:
            if idx >= test_triples.num_triples:
                logger.warning(f"Index {idx} out of range, skipping")
                continue

            test_triple = tuple(int(x) for x in test_triples.mapped_triples[idx])
            logger.info(f"\n{'='*80}")
            logger.info(f"Analyzing test triple {idx}: {test_triple}")
            logger.info('='*80)

            influences = analyzer.compute_influences_sampled(
                test_triple=test_triple,
                training_triples=train_triples,
                sample_rate=args.sample_rate,
                stratify_by=args.stratify_by,
                learning_rate=args.learning_rate,
                top_k=args.top_k,
                seed=args.seed
            )

            logger.info(f"\nTop-{min(5, args.top_k)} influential training triples:")
            for i, inf in enumerate(influences[:5], 1):
                logger.info(f"  {i}. ({inf['train_head']}, {inf['train_relation']}, {inf['train_tail']})")
                logger.info(f"     Influence: {inf['influence']:.6f}")

            results.append({
                'test_index': idx,
                'test_triple': test_triple,
                'influences': influences
            })

        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")

    elif args.mode == 'test':
        # Analyze test set
        test_triple_list = [(int(h), int(r), int(t)) for h, r, t in test_triples.mapped_triples]

        if args.max_test_triples:
            test_triple_list = test_triple_list[:args.max_test_triples]

        logger.info(f"\nAnalyzing {len(test_triple_list)} test triples...")

        results = []
        for i, test_triple in enumerate(test_triple_list):
            logger.info(f"\n[{i+1}/{len(test_triple_list)}] Test triple: {test_triple}")

            influences = analyzer.compute_influences_sampled(
                test_triple=test_triple,
                training_triples=train_triples,
                sample_rate=args.sample_rate,
                stratify_by=args.stratify_by,
                learning_rate=args.learning_rate,
                top_k=args.top_k,
                seed=args.seed
            )

            results.append({
                'test_index': i,
                'test_triple': test_triple,
                'influences': influences
            })

        # Save results
        output_data = {
            'num_test_triples': len(results),
            'sample_rate': args.sample_rate,
            'top_k': args.top_k,
            'results': results
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"\nResults saved to {args.output}")

    logger.info("\n" + "=" * 80)
    logger.info("Fast TracIn analysis completed! 🚀")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
