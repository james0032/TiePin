#!/usr/bin/env python3
"""
Convert train_pytorch.py checkpoints to PyKEEN-compatible format.

This script converts checkpoints saved by the custom PyTorch ConvE implementation
in train_pytorch.py to the format expected by PyKEEN's ConvE model, enabling
compatibility with tracin_optimized.py and other PyKEEN-based tools.

Usage:
    # Convert single checkpoint
    python convert_checkpoint.py \
        --input models/conve/checkpoints/checkpoint_epoch_50.pt \
        --output models/conve/checkpoints/checkpoint_epoch_50_pykeen.pt

    # Convert all checkpoints in a directory
    python convert_checkpoint.py \
        --input-dir models/conve/checkpoints \
        --output-dir models/conve/checkpoints_pykeen

    # Verify conversion (compare outputs)
    python convert_checkpoint.py \
        --input checkpoint.pt \
        --output checkpoint_pykeen.pt \
        --verify
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_pytorch_to_pykeen_state_dict(pytorch_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Convert train_pytorch.py state dict to PyKEEN ConvE format.

    Args:
        pytorch_state_dict: State dict from custom PyTorch ConvE model

    Returns:
        State dict in PyKEEN ConvE format
    """
    pykeen_state_dict = {}

    # Layer name mapping from custom PyTorch to PyKEEN format
    name_map = {
        # Embeddings
        'entity_embeddings.weight': 'entity_representations.0._embeddings.weight',
        'relation_embeddings.weight': 'relation_representations.0._embeddings.weight',

        # Convolutional layer
        'conv1.weight': 'interaction.hr2d.2.weight',
        'conv1.bias': 'interaction.hr2d.2.bias',

        # Fully connected layer
        'fc.weight': 'interaction.hr1d.0.weight',
        'fc.bias': 'interaction.hr1d.0.bias',

        # Batch normalization - input (bn0)
        'bn0.weight': 'interaction.hr2d.0.weight',
        'bn0.bias': 'interaction.hr2d.0.bias',
        'bn0.running_mean': 'interaction.hr2d.0.running_mean',
        'bn0.running_var': 'interaction.hr2d.0.running_var',
        'bn0.num_batches_tracked': 'interaction.hr2d.0.num_batches_tracked',

        # Batch normalization - after conv (bn1)
        'bn1.weight': 'interaction.hr2d.3.weight',
        'bn1.bias': 'interaction.hr2d.3.bias',
        'bn1.running_mean': 'interaction.hr2d.3.running_mean',
        'bn1.running_var': 'interaction.hr2d.3.running_var',
        'bn1.num_batches_tracked': 'interaction.hr2d.3.num_batches_tracked',

        # Batch normalization - after fc (bn2)
        'bn2.weight': 'interaction.hr1d.1.weight',
        'bn2.bias': 'interaction.hr1d.1.bias',
        'bn2.running_mean': 'interaction.hr1d.1.running_mean',
        'bn2.running_var': 'interaction.hr1d.1.running_var',
        'bn2.num_batches_tracked': 'interaction.hr1d.1.num_batches_tracked',
    }

    # Convert each layer
    converted_count = 0
    missing_keys = []

    for old_name, new_name in name_map.items():
        if old_name in pytorch_state_dict:
            pykeen_state_dict[new_name] = pytorch_state_dict[old_name].clone()
            converted_count += 1
        else:
            missing_keys.append(old_name)

    logger.info(f"Converted {converted_count}/{len(name_map)} layers")

    if missing_keys:
        logger.warning(f"Missing keys in source checkpoint: {missing_keys}")

    # Report unconverted keys (should be none if format is correct)
    unconverted = set(pytorch_state_dict.keys()) - set(name_map.keys())
    if unconverted:
        logger.warning(f"Unconverted keys (not in name_map): {unconverted}")

    return pykeen_state_dict


def convert_checkpoint(
    input_path: str,
    output_path: str,
    verify: bool = False
) -> bool:
    """Convert a single checkpoint file.

    Args:
        input_path: Path to train_pytorch.py checkpoint
        output_path: Path to save PyKEEN-compatible checkpoint
        verify: If True, verify the conversion by comparing tensor shapes

    Returns:
        True if conversion successful, False otherwise
    """
    try:
        logger.info(f"Loading checkpoint from: {input_path}")
        checkpoint = torch.load(input_path, map_location='cpu')

        # Validate format
        if not isinstance(checkpoint, dict):
            logger.error("Checkpoint is not a dictionary")
            return False

        if 'model_state_dict' not in checkpoint:
            logger.error("Checkpoint missing 'model_state_dict' key")
            return False

        # Extract components
        pytorch_state_dict = checkpoint['model_state_dict']
        model_config = checkpoint.get('model_config', {})
        epoch = checkpoint.get('epoch', None)

        logger.info(f"Source checkpoint info:")
        logger.info(f"  Epoch: {epoch}")
        logger.info(f"  State dict keys: {len(pytorch_state_dict)}")
        logger.info(f"  Model config: {model_config}")

        # Convert state dict
        logger.info("Converting state dict...")
        pykeen_state_dict = convert_pytorch_to_pykeen_state_dict(pytorch_state_dict)

        logger.info(f"Converted state dict has {len(pykeen_state_dict)} keys")

        # Create output checkpoint with PyKEEN format
        output_checkpoint = {
            'model_state_dict': pykeen_state_dict,
            'model_config': model_config,  # Preserve config
            'epoch': epoch,  # Preserve epoch
            'conversion_info': {
                'original_format': 'train_pytorch.py',
                'converted_format': 'pykeen',
                'source_file': str(input_path)
            }
        }

        # Preserve optimizer state if present (though PyKEEN may not use it)
        if 'optimizer_state_dict' in checkpoint:
            output_checkpoint['optimizer_state_dict'] = checkpoint['optimizer_state_dict']
            logger.info("  Preserved optimizer state")

        # Preserve metrics if present
        if 'metrics' in checkpoint:
            output_checkpoint['metrics'] = checkpoint['metrics']
            logger.info("  Preserved metrics")

        # Save converted checkpoint
        logger.info(f"Saving converted checkpoint to: {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(output_checkpoint, output_path)

        logger.info("✓ Conversion successful!")

        # Verification
        if verify:
            logger.info("\nVerifying conversion...")
            verify_conversion(pytorch_state_dict, pykeen_state_dict, model_config)

        return True

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_conversion(
    pytorch_state: Dict[str, torch.Tensor],
    pykeen_state: Dict[str, torch.Tensor],
    model_config: Dict
):
    """Verify that the conversion preserved all information.

    Args:
        pytorch_state: Original state dict
        pykeen_state: Converted state dict
        model_config: Model configuration
    """
    logger.info("Verification checks:")

    # Check tensor counts
    logger.info(f"  Original tensors: {len(pytorch_state)}")
    logger.info(f"  Converted tensors: {len(pykeen_state)}")

    # Check total parameters
    pytorch_params = sum(t.numel() for t in pytorch_state.values())
    pykeen_params = sum(t.numel() for t in pykeen_state.values())

    logger.info(f"  Original parameters: {pytorch_params:,}")
    logger.info(f"  Converted parameters: {pykeen_params:,}")

    if pytorch_params == pykeen_params:
        logger.info("  ✓ Parameter count matches!")
    else:
        logger.warning(f"  ⚠ Parameter count mismatch: {pytorch_params:,} vs {pykeen_params:,}")

    # Check key shapes (for layers we know about)
    known_mappings = {
        'entity_embeddings.weight': 'entity_representations.0._embeddings.weight',
        'relation_embeddings.weight': 'relation_representations.0._embeddings.weight',
        'conv1.weight': 'interaction.hr2d.2.weight',
        'fc.weight': 'interaction.hr1d.0.weight',
    }

    logger.info("\n  Layer shape verification:")
    for old_key, new_key in known_mappings.items():
        if old_key in pytorch_state and new_key in pykeen_state:
            old_shape = pytorch_state[old_key].shape
            new_shape = pykeen_state[new_key].shape
            if old_shape == new_shape:
                logger.info(f"    ✓ {old_key}: {old_shape}")
            else:
                logger.warning(f"    ✗ {old_key}: {old_shape} → {new_shape} (MISMATCH!)")

    # Check embedding dimensions match config
    if 'entity_representations.0._embeddings.weight' in pykeen_state:
        num_entities, embedding_dim = pykeen_state['entity_representations.0._embeddings.weight'].shape
        logger.info(f"\n  Model dimensions:")
        logger.info(f"    Entities: {num_entities}")
        logger.info(f"    Embedding dim: {embedding_dim}")

        if model_config:
            config_entities = model_config.get('num_entities')
            config_dim = model_config.get('embedding_dim')
            if config_entities == num_entities and config_dim == embedding_dim:
                logger.info(f"    ✓ Matches config")
            else:
                logger.warning(f"    ⚠ Config mismatch: {config_entities}, {config_dim}")

    logger.info("\n✓ Verification complete")


def convert_directory(
    input_dir: str,
    output_dir: str,
    pattern: str = "checkpoint_*.pt",
    verify: bool = False
):
    """Convert all checkpoints in a directory.

    Args:
        input_dir: Directory containing train_pytorch.py checkpoints
        output_dir: Directory to save converted checkpoints
        pattern: Glob pattern for checkpoint files
        verify: If True, verify each conversion
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return

    checkpoint_files = sorted(input_path.glob(pattern))

    if not checkpoint_files:
        logger.warning(f"No checkpoint files found matching pattern: {pattern}")
        return

    logger.info(f"Found {len(checkpoint_files)} checkpoint files")

    output_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    for ckpt_file in checkpoint_files:
        logger.info(f"\n{'='*60}")
        logger.info(f"Converting: {ckpt_file.name}")
        logger.info(f"{'='*60}")

        output_file = output_path / f"{ckpt_file.stem}_pykeen.pt"

        if convert_checkpoint(str(ckpt_file), str(output_file), verify=verify):
            success_count += 1

    logger.info(f"\n{'='*60}")
    logger.info(f"Conversion complete: {success_count}/{len(checkpoint_files)} successful")
    logger.info(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert train_pytorch.py checkpoints to PyKEEN format'
    )

    # Input/output
    parser.add_argument('--input', type=str, help='Input checkpoint file')
    parser.add_argument('--output', type=str, help='Output checkpoint file')
    parser.add_argument('--input-dir', type=str, help='Input directory (for batch conversion)')
    parser.add_argument('--output-dir', type=str, help='Output directory (for batch conversion)')

    # Options
    parser.add_argument('--pattern', type=str, default='checkpoint_*.pt',
                       help='Glob pattern for checkpoint files (default: checkpoint_*.pt)')
    parser.add_argument('--verify', action='store_true',
                       help='Verify conversion by comparing tensor shapes')

    args = parser.parse_args()

    # Validate arguments
    if args.input and args.output:
        # Single file conversion
        convert_checkpoint(args.input, args.output, verify=args.verify)
    elif args.input_dir and args.output_dir:
        # Directory conversion
        convert_directory(args.input_dir, args.output_dir,
                         pattern=args.pattern, verify=args.verify)
    else:
        parser.error("Must specify either (--input and --output) or (--input-dir and --output-dir)")


if __name__ == '__main__':
    main()
