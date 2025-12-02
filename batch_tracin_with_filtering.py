#!/usr/bin/env python
"""
Batch TracIn Analysis with Training Data Filtering

This script:
1. Reads a list of test triples
2. For each triple, filters the training data by proximity
3. Runs TracIn analysis using the filtered training data
4. Generates CSV output with influence scores

This approach significantly speeds up TracIn by reducing training data size.
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_filename(text: str) -> str:
    """Sanitize text to be safe for filenames."""
    return text.replace(':', '_').replace('/', '_').replace(' ', '_')


def read_test_triples(test_file: str) -> List[Tuple[str, str, str]]:
    """Read test triples from file.

    Args:
        test_file: Path to test triples file (tab-separated)

    Returns:
        List of (head, relation, tail) tuples
    """
    triples = []
    with open(test_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) == 3:
                triples.append(tuple(parts))
    return triples


def create_single_triple_file(triple: Tuple[str, str, str], output_path: str):
    """Create a file with a single test triple.

    Args:
        triple: (head, relation, tail) tuple
        output_path: Path to write the triple file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(f"{triple[0]}\t{triple[1]}\t{triple[2]}\n")


def filter_training_data(
    train_file: str,
    test_triple_file: str,
    output_file: str,
    n_hops: int = 2,
    min_degree: int = 2,
    cache_path: str = None,
    preserve_test_edges: bool = True,
    strict_hop_constraint: bool = False,
    path_filtering: bool = False,
    max_total_path_length: int = None
) -> bool:
    """Filter training data by proximity to test triple using PyG.

    Args:
        train_file: Path to training triples
        test_triple_file: Path to single test triple file
        output_file: Path to write filtered training data
        n_hops: Number of hops for proximity filtering
        min_degree: Minimum degree threshold
        cache_path: Optional path to cached graph
        preserve_test_edges: Whether to preserve edges containing test entities
        strict_hop_constraint: Whether to enforce strict n-hop constraint
        path_filtering: Whether to only keep edges on paths between drug and disease
        max_total_path_length: Maximum total path length when path_filtering is enabled

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'filter_training_by_proximity_pyg.py',
        '--train', train_file,
        '--test', test_triple_file,
        '--output', output_file,
        '--n-hops', str(n_hops),
        '--min-degree', str(min_degree),
        '--single-triple'
    ]

    if cache_path:
        cmd.extend(['--cache', cache_path])

    if not preserve_test_edges:
        cmd.append('--no-preserve-test-edges')

    if strict_hop_constraint:
        cmd.append('--strict-hop-constraint')

    if path_filtering:
        cmd.append('--path-filtering')

    if max_total_path_length is not None:
        cmd.extend(['--max-total-path-length', str(max_total_path_length)])

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            # Let stdout/stderr pass through to console (don't capture)
            # This allows real-time log viewing
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
            text=True
        )
        logger.info("Filtering completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Filtering failed: {e}")
        return False


def filter_training_data_networkx(
    train_file: str,
    test_triple_file: str,
    output_file: str,
    n_hops: int = 2,
    min_degree: int = 2,
    preserve_test_edges: bool = True,
    strict_hop_constraint: bool = False,
    path_filtering: bool = False,
    max_total_path_length: int = None
) -> bool:
    """Filter training data by proximity to test triple using NetworkX.

    Args:
        train_file: Path to training triples
        test_triple_file: Path to single test triple file
        output_file: Path to write filtered training data
        n_hops: Number of hops for proximity filtering
        min_degree: Minimum degree threshold
        preserve_test_edges: Whether to preserve edges containing test entities
        strict_hop_constraint: Whether to enforce strict n-hop constraint
        path_filtering: Whether to only keep edges on paths between drug and disease
        max_total_path_length: Maximum total path length when path_filtering is enabled

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'filter_training_networkx.py',
        '--train', train_file,
        '--test', test_triple_file,
        '--output', output_file,
        '--n-hops', str(n_hops),
        '--min-degree', str(min_degree),
        '--single-triple'
    ]

    if not preserve_test_edges:
        cmd.append('--no-preserve-test-edges')

    if strict_hop_constraint:
        cmd.append('--strict-hop-constraint')

    if path_filtering:
        cmd.append('--path-filtering')

    if max_total_path_length is not None:
        cmd.extend(['--max-total-path-length', str(max_total_path_length)])

    logger.info(f"Running NetworkX filter: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            # Let stdout/stderr pass through to console (don't capture)
            # This allows real-time log viewing
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
            text=True
        )
        logger.info("NetworkX filtering completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"NetworkX filtering failed: {e}")
        return False


def run_tracin_analysis(
    model_path: str,
    train_file: str,
    test_triple_file: str,
    entity_to_id: str,
    relation_to_id: str,
    output_json: str,
    output_csv: str,
    edge_map: str = None,
    node_name_dict: str = None,
    top_k: int = None,
    device: str = 'cuda',
    use_last_layers: bool = False,
    num_last_layers: int = 2,
    batch_size: int = 512,
    use_mixed_precision: bool = False,
    use_gradient_checkpointing: bool = False,
    disable_memory_cleanup: bool = False,
    use_optimized_tracin: bool = False,
    use_vectorized_gradients: bool = True,
    cache_test_gradients: bool = True,
    use_torch_compile: bool = False,
    enable_multi_gpu: bool = False
) -> bool:
    """Run TracIn analysis on a single test triple.

    Args:
        model_path: Path to trained model
        train_file: Path to (filtered) training data
        test_triple_file: Path to single test triple file
        entity_to_id: Path to entity_to_id.tsv
        relation_to_id: Path to relation_to_id.tsv
        output_json: Path to write JSON output
        output_csv: Path to write CSV output
        edge_map: Optional path to edge_map.json
        node_name_dict: Optional path to node_name_dict.txt
        top_k: Number of top influences to compute (None = all influences)
        device: Device to use (cuda/cpu)
        use_last_layers: Whether to use last layers only (default: False = all layers)
        num_last_layers: Number of last layers to track when use_last_layers=True
        batch_size: Batch size for processing
        use_mixed_precision: Use FP16 mixed precision (2x memory + 2x speed)
        use_gradient_checkpointing: Use gradient checkpointing (2-3x memory reduction)
        disable_memory_cleanup: Disable automatic memory cleanup

    Returns:
        True if successful, False otherwise
    """
    cmd = [
        'python', 'run_tracin.py',
        '--model-path', model_path,
        '--train', train_file,
        '--test', test_triple_file,
        '--entity-to-id', entity_to_id,
        '--relation-to-id', relation_to_id,
        '--edge-map', edge_map,
        '--node-name-dict', node_name_dict,
        '--output', output_json,
        '--csv-output', output_csv,
        '--mode', 'single',
        '--test-indices', '0',
        '--device', device,
        '--batch-size', str(batch_size)
    ]

    # Only add top-k if specified (None means return all influences)
    if top_k is not None:
        cmd.extend(['--top-k', str(top_k)])

    if use_last_layers:
        cmd.extend(['--use-last-layers-only', '--num-last-layers', str(num_last_layers)])

    if edge_map:
        cmd.extend(['--edge-map', edge_map])

    if node_name_dict:
        cmd.extend(['--node-name-dict', node_name_dict])

    # Add Phase 1 optimization flags (basic)
    if use_mixed_precision:
        cmd.append('--use-mixed-precision')

    if use_gradient_checkpointing:
        cmd.append('--use-gradient-checkpointing')

    if disable_memory_cleanup:
        cmd.append('--disable-memory-cleanup')

    # Add Phase 2 optimization flags (advanced)
    if use_optimized_tracin:
        cmd.append('--use-optimized-tracin')

    if not use_vectorized_gradients:
        cmd.append('--disable-vectorized-gradients')

    if not cache_test_gradients:
        cmd.append('--disable-test-gradient-caching')

    if use_torch_compile:
        cmd.append('--use-torch-compile')

    if enable_multi_gpu:
        cmd.append('--enable-multi-gpu')

    logger.info(f"Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            # Let stdout/stderr pass through to console (don't capture)
            # This allows real-time log viewing
            stdout=None,  # Inherit stdout
            stderr=None,  # Inherit stderr
            text=True
        )
        logger.info("TracIn analysis completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"TracIn analysis failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch TracIn analysis with training data filtering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic usage with PyG filtering (default, fastest)
  python batch_tracin_with_filtering.py \\
      --test-triples examples/20251017_top_test_triples.txt \\
      --model-path model.pt \\
      --train train.txt \\
      --entity-to-id entity_to_id.tsv \\
      --relation-to-id relation_to_id.tsv \\
      --edge-map edge_map.json \\
      --node-name-dict node_name_dict.txt \\
      --output-dir results/batch_tracin \\
      --n-hops 2 \\
      --device cuda

  # Use NetworkX filtering (more transparent, easier to debug)
  python batch_tracin_with_filtering.py \\
      --test-triples examples/20251017_top_test_triples.txt \\
      --model-path model.pt \\
      --train train.txt \\
      --entity-to-id entity_to_id.tsv \\
      --relation-to-id relation_to_id.tsv \\
      --output-dir results/batch_tracin \\
      --filter-method networkx \\
      --n-hops 2 \\
      --path-filtering \\
      --device cuda

  # Resume from checkpoint (skip already processed triples)
  python batch_tracin_with_filtering.py \\
      --test-triples examples/20251017_top_test_triples.txt \\
      --model-path model.pt \\
      --train train.txt \\
      --entity-to-id entity_to_id.tsv \\
      --relation-to-id relation_to_id.tsv \\
      --output-dir results/batch_tracin \\
      --skip-existing \\
      --device cuda

  # Process specific range (e.g., triples 100-200)
  python batch_tracin_with_filtering.py \\
      --test-triples examples/20251017_top_test_triples.txt \\
      --model-path model.pt \\
      --train train.txt \\
      --entity-to-id entity_to_id.tsv \\
      --relation-to-id relation_to_id.tsv \\
      --output-dir results/batch_tracin \\
      --start-index 100 \\
      --max-triples 100 \\
      --skip-existing \\
      --device cuda
        """
    )

    # Required arguments
    parser.add_argument('--test-triples', type=str, required=True,
                        help='Path to test triples file')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to trained model (.pt)')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to training triples')
    parser.add_argument('--entity-to-id', type=str, required=True,
                        help='Path to entity_to_id.tsv')
    parser.add_argument('--relation-to-id', type=str, required=True,
                        help='Path to relation_to_id.tsv')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for results')

    # Optional arguments
    parser.add_argument('--edge-map', type=str,
                        help='Path to edge_map.json (optional)')
    parser.add_argument('--node-name-dict', type=str,
                        help='Path to node_name_dict.txt (optional)')

    # Filtering parameters
    parser.add_argument('--filter-method', type=str, default='pyg',
                        choices=['pyg', 'networkx'],
                        help='Filtering implementation to use: pyg (PyTorch Geometric, faster) or '
                             'networkx (more transparent, easier to debug) (default: pyg)')
    parser.add_argument('--n-hops', type=int, default=2,
                        help='Number of hops for proximity filtering (default: 2)')
    parser.add_argument('--min-degree', type=int, default=2,
                        help='Minimum degree threshold (default: 2)')
    parser.add_argument('--cache', type=str,
                        help='Path to cache graph (optional, speeds up PyG filtering only)')
    parser.add_argument('--no-preserve-test-edges', action='store_true',
                        help='Do not preserve edges containing test entities')
    parser.add_argument('--strict-hop-constraint', action='store_true',
                        help='Enforce strict n-hop constraint: both endpoints of each edge '
                             'must be within n_hops (prevents distant shortcuts)')
    parser.add_argument('--path-filtering', action='store_true',
                        help='Only keep edges on paths between drug and disease within n_hops+n_hops '
                             '(stricter than intersection filtering)')
    parser.add_argument('--max-total-path-length', type=int, default=None,
                        help='Maximum total path length (drug_dist + disease_dist). '
                             'Only used when --path-filtering is enabled. '
                             'Example: --max-total-path-length 3 limits to 3-hop paths.')

    # TracIn parameters
    parser.add_argument('--top-k', type=int, default=None,
                        help='Number of top influences (default: None = all influences)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use (default: cuda)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size (default: 512)')
    parser.add_argument('--use-last-layers-only', action='store_true',
                        help='Only compute gradients for last layers (MUCH faster, following original TracIn paper)')
    parser.add_argument('--num-last-layers', type=int, default=2,
                        help='Number of last layers to track when --use-last-layers-only is set (default: 2)')

    # Memory optimization arguments (Phase 1 - Basic)
    parser.add_argument('--use-mixed-precision', action='store_true',
                        help='Use FP16 mixed precision (2x memory + 2x speed). Recommended for GPUs with Tensor Cores.')
    parser.add_argument('--use-gradient-checkpointing', action='store_true',
                        help='Use gradient checkpointing (2-3x memory reduction, slight speed penalty)')
    parser.add_argument('--disable-memory-cleanup', action='store_true',
                        help='Disable automatic memory cleanup (tensor deletion and cache clearing)')

    # Advanced optimization arguments (Phase 2 - High Performance)
    parser.add_argument('--use-optimized-tracin', action='store_true',
                        help='Use tracin_optimized.py for 20-80x speedup (enables vectorized gradients + test caching by default)')
    parser.add_argument('--disable-vectorized-gradients', action='store_true',
                        help='Disable vectorized gradient computation (only applies if --use-optimized-tracin is set)')
    parser.add_argument('--disable-test-gradient-caching', action='store_true',
                        help='Disable test gradient caching (only applies if --use-optimized-tracin is set)')
    parser.add_argument('--use-torch-compile', action='store_true',
                        help='Use torch.compile for JIT compilation (1.5x speedup, requires PyTorch 2.0+)')
    parser.add_argument('--enable-multi-gpu', action='store_true',
                        help='Enable multi-GPU processing (experimental, 3-4x speedup with 4 GPUs)')

    # Execution control
    parser.add_argument('--start-index', type=int, default=0,
                        help='Start from this triple index (default: 0)')
    parser.add_argument('--max-triples', type=int,
                        help='Maximum number of triples to process (default: all)')
    parser.add_argument('--skip-filtering', action='store_true',
                        help='Skip filtering step (use existing filtered files)')
    parser.add_argument('--skip-tracin', action='store_true',
                        help='Skip TracIn step (only do filtering)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip triples that already have output CSV files (resume from checkpoint)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    filtered_dir = output_dir / 'filtered_training'
    filtered_dir.mkdir(exist_ok=True)

    temp_dir = output_dir / 'temp_triples'
    temp_dir.mkdir(exist_ok=True)

    # Read test triples
    logger.info(f"Reading test triples from {args.test_triples}")
    test_triples = read_test_triples(args.test_triples)
    logger.info(f"Found {len(test_triples)} test triples")

    # Apply start index and max triples
    if args.start_index > 0:
        test_triples = test_triples[args.start_index:]
        logger.info(f"Starting from index {args.start_index}")

    if args.max_triples:
        test_triples = test_triples[:args.max_triples]
        logger.info(f"Processing up to {args.max_triples} triples")

    # Process each test triple
    summary = {
        'total_triples': len(test_triples),
        'successful': 0,
        'failed_filtering': 0,
        'failed_tracin': 0,
        'skipped': 0,
        'filter_method': args.filter_method,
        'results': []
    }

    start_time = time.time()

    for idx, triple in enumerate(test_triples):
        head, rel, tail = triple
        triple_idx = args.start_index + idx

        logger.info("=" * 80)
        logger.info(f"Processing triple {triple_idx + 1}/{args.start_index + len(test_triples)}: {head} --[{rel}]--> {tail}")
        logger.info("=" * 80)

        # Create sanitized filename
        head_clean = sanitize_filename(head)
        tail_clean = sanitize_filename(tail)
        base_name = f"triple_{triple_idx:03d}_{head_clean}_{tail_clean}"

        # Paths
        temp_triple_file = temp_dir / f"{base_name}.txt"
        filtered_train_file = filtered_dir / f"{base_name}_filtered_train.txt"
        output_json = output_dir / f"{base_name}_tracin.json"
        output_csv = output_dir / f"{base_name}_tracin.csv"

        # Check if output already exists (checkpoint/resume functionality)
        if args.skip_existing and output_csv.exists():
            logger.info(f"✓ Output already exists: {output_csv}")
            logger.info(f"  Skipping triple {triple_idx} (use --no-skip-existing to force re-run)")
            summary['skipped'] += 1
            result = {
                'index': triple_idx,
                'triple': {'head': head, 'relation': rel, 'tail': tail},
                'base_name': base_name,
                'filtering_success': True,
                'tracin_success': True,
                'filtered_train_file': str(filtered_train_file),
                'output_csv': str(output_csv),
                'skipped': True,
                'reason': 'output_exists'
            }
            summary['results'].append(result)
            continue

        result = {
            'index': triple_idx,
            'triple': {'head': head, 'relation': rel, 'tail': tail},
            'base_name': base_name,
            'filtering_success': False,
            'tracin_success': False,
            'filtered_train_file': str(filtered_train_file),
            'output_csv': str(output_csv)
        }

        # Step 1: Create single triple file
        create_single_triple_file(triple, str(temp_triple_file))
        logger.info(f"Created test triple file: {temp_triple_file}")

        # Step 2: Filter training data
        if not args.skip_filtering:
            logger.info(f"Step 1/2: Filtering training data using {args.filter_method.upper()}...")

            # Choose filtering method
            if args.filter_method == 'networkx':
                filter_success = filter_training_data_networkx(
                    train_file=args.train,
                    test_triple_file=str(temp_triple_file),
                    output_file=str(filtered_train_file),
                    n_hops=args.n_hops,
                    min_degree=args.min_degree,
                    preserve_test_edges=not args.no_preserve_test_edges,
                    strict_hop_constraint=args.strict_hop_constraint,
                    path_filtering=args.path_filtering,
                    max_total_path_length=args.max_total_path_length
                )
            else:  # pyg
                filter_success = filter_training_data(
                    train_file=args.train,
                    test_triple_file=str(temp_triple_file),
                    output_file=str(filtered_train_file),
                    n_hops=args.n_hops,
                    min_degree=args.min_degree,
                    cache_path=args.cache,
                    preserve_test_edges=not args.no_preserve_test_edges,
                    strict_hop_constraint=args.strict_hop_constraint,
                    path_filtering=args.path_filtering,
                    max_total_path_length=args.max_total_path_length
                )

            result['filtering_success'] = filter_success
            result['filter_method'] = args.filter_method

            if not filter_success:
                logger.error(f"Filtering failed for triple {triple_idx}")
                summary['failed_filtering'] += 1
                summary['results'].append(result)
                continue
        else:
            logger.info("Skipping filtering (using existing filtered file)")
            if not filtered_train_file.exists():
                logger.error(f"Filtered file not found: {filtered_train_file}")
                summary['failed_filtering'] += 1
                summary['results'].append(result)
                continue
            result['filtering_success'] = True

        # Step 3: Run TracIn analysis
        if not args.skip_tracin:
            logger.info("Step 2/2: Running TracIn analysis...")
            tracin_success = run_tracin_analysis(
                model_path=args.model_path,
                train_file=str(filtered_train_file),
                test_triple_file=str(temp_triple_file),
                entity_to_id=args.entity_to_id,
                relation_to_id=args.relation_to_id,
                output_json=str(output_json),
                output_csv=str(output_csv),
                edge_map=args.edge_map,
                node_name_dict=args.node_name_dict,
                top_k=args.top_k,
                device=args.device,
                use_last_layers=args.use_last_layers_only,
                num_last_layers=args.num_last_layers,
                batch_size=args.batch_size,
                # Phase 1 optimizations
                use_mixed_precision=args.use_mixed_precision,
                use_gradient_checkpointing=args.use_gradient_checkpointing,
                disable_memory_cleanup=args.disable_memory_cleanup,
                # Phase 2 advanced optimizations
                use_optimized_tracin=args.use_optimized_tracin,
                use_vectorized_gradients=not args.disable_vectorized_gradients,
                cache_test_gradients=not args.disable_test_gradient_caching,
                use_torch_compile=args.use_torch_compile,
                enable_multi_gpu=args.enable_multi_gpu
            )

            result['tracin_success'] = tracin_success

            if tracin_success:
                summary['successful'] += 1
                logger.info(f"✓ Successfully completed triple {triple_idx}")
                logger.info(f"  CSV output: {output_csv}")
            else:
                logger.error(f"✗ TracIn failed for triple {triple_idx}")
                summary['failed_tracin'] += 1
        else:
            logger.info("Skipping TracIn analysis")
            summary['skipped'] += 1
            result['tracin_success'] = None

        summary['results'].append(result)

    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    summary['elapsed_time_seconds'] = elapsed_time
    summary['elapsed_time_formatted'] = f"{elapsed_time // 3600:.0f}h {(elapsed_time % 3600) // 60:.0f}m {elapsed_time % 60:.0f}s"

    # Save summary
    summary_file = output_dir / 'batch_tracin_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 80)
    logger.info("BATCH PROCESSING COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Filter method: {summary['filter_method'].upper()}")
    logger.info(f"Total triples in batch: {summary['total_triples']}")
    logger.info(f"Successful (new): {summary['successful']}")
    logger.info(f"Skipped (existing): {summary['skipped']}")
    logger.info(f"Failed (filtering): {summary['failed_filtering']}")
    logger.info(f"Failed (TracIn): {summary['failed_tracin']}")
    logger.info(f"Elapsed time: {summary['elapsed_time_formatted']}")
    logger.info(f"Summary saved to: {summary_file}")
    if args.skip_existing:
        logger.info(f"\n✓ Resume mode enabled (--skip-existing)")
        logger.info(f"  Total completed (new + existing): {summary['successful'] + summary['skipped']}")
    logger.info("=" * 80)

    # Print output file locations
    if summary['successful'] > 0:
        logger.info("\nOutput files:")
        logger.info(f"  Filtered training data: {filtered_dir}")
        logger.info(f"  TracIn CSV files: {output_dir}/*_tracin.csv")
        logger.info(f"  TracIn JSON files: {output_dir}/*_tracin.json")


if __name__ == '__main__':
    main()
