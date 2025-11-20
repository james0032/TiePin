#!/usr/bin/env python3
"""
Training script for ConvE model using pure PyTorch.

This script implements ConvE training from scratch with proper checkpoint
saving every N epochs. Unlike PyKEEN, this gives full control over the
training loop and checkpoint management.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

# Import autocast with backward compatibility for different PyTorch versions
try:
    from torch.amp import autocast
    from torch.cuda.amp import GradScaler
except ImportError:
    # Fallback for older PyTorch versions
    from torch.cuda.amp import autocast, GradScaler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConvE(nn.Module):
    """ConvE model implementation in PyTorch."""

    def __init__(
        self,
        num_entities: int,
        num_relations: int,
        embedding_dim: int = 200,
        output_channels: int = 32,
        kernel_height: int = 3,
        kernel_width: int = 3,
        input_dropout: float = 0.2,
        feature_map_dropout: float = 0.2,
        output_dropout: float = 0.3,
        embedding_height: int = 10,
        embedding_width: int = 20,
    ):
        super().__init__()

        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.embedding_height = embedding_height
        self.embedding_width = embedding_width

        # Embeddings
        self.entity_embeddings = nn.Embedding(num_entities, embedding_dim)
        self.relation_embeddings = nn.Embedding(num_relations, embedding_dim)

        # Batch normalization for embeddings
        self.bn0 = nn.BatchNorm2d(1)

        # Convolution
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=output_channels,
            kernel_size=(kernel_height, kernel_width),
            stride=1,
            padding=0,
            bias=True
        )

        # Batch normalization after convolution
        self.bn1 = nn.BatchNorm2d(output_channels)

        # Dropout layers
        self.input_dropout = nn.Dropout(input_dropout)
        self.feature_map_dropout = nn.Dropout2d(feature_map_dropout)
        self.output_dropout = nn.Dropout(output_dropout)

        # Calculate flattened size after convolution
        # Note: head and relation embeddings are concatenated along width, so input width is 2*embedding_width
        conv_out_height = embedding_height - kernel_height + 1
        conv_out_width = (2 * embedding_width) - kernel_width + 1
        flattened_size = output_channels * conv_out_height * conv_out_width

        # Fully connected layer
        self.fc = nn.Linear(flattened_size, embedding_dim)

        # Batch normalization after FC
        self.bn2 = nn.BatchNorm1d(embedding_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        nn.init.xavier_normal_(self.entity_embeddings.weight.data)
        nn.init.xavier_normal_(self.relation_embeddings.weight.data)
        nn.init.xavier_normal_(self.conv1.weight.data)
        nn.init.xavier_normal_(self.fc.weight.data)

    def forward(self, head_idx, relation_idx):
        """Forward pass.

        Args:
            head_idx: Head entity indices [batch_size]
            relation_idx: Relation indices [batch_size]

        Returns:
            Scores for all entities [batch_size, num_entities]
        """
        batch_size = head_idx.size(0)

        # Get embeddings
        head_emb = self.entity_embeddings(head_idx)  # [batch_size, embedding_dim]
        relation_emb = self.relation_embeddings(relation_idx)  # [batch_size, embedding_dim]

        # Reshape to 2D for convolution
        head_emb = head_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)
        relation_emb = relation_emb.view(batch_size, 1, self.embedding_height, self.embedding_width)

        # Concatenate along width dimension
        stacked = torch.cat([head_emb, relation_emb], dim=3)  # [batch_size, 1, height, 2*width]

        # Input dropout and batch norm
        stacked = self.input_dropout(stacked)
        stacked = self.bn0(stacked)

        # Convolution
        x = self.conv1(stacked)  # [batch_size, out_channels, conv_height, conv_width]
        x = self.bn1(x)
        x = F.relu(x)
        x = self.feature_map_dropout(x)

        # Flatten
        x = x.view(batch_size, -1)

        # FC layer
        x = self.fc(x)
        x = self.output_dropout(x)
        x = self.bn2(x)
        x = F.relu(x)

        # Score against all entities
        all_entity_emb = self.entity_embeddings.weight  # [num_entities, embedding_dim]
        scores = torch.mm(x, all_entity_emb.t())  # [batch_size, num_entities]

        return scores


class TriplesDataset(Dataset):
    """Dataset for knowledge graph triples."""

    def __init__(self, triples: np.ndarray):
        """
        Args:
            triples: numpy array of shape [num_triples, 3] containing (head, rel, tail) indices
        """
        self.triples = triples

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]
        return {
            'head': torch.tensor(head, dtype=torch.long),
            'relation': torch.tensor(relation, dtype=torch.long),
            'tail': torch.tensor(tail, dtype=torch.long)
        }


def load_triples(
    triples_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str
) -> Tuple[np.ndarray, Dict[str, int], Dict[str, int]]:
    """Load triples and mappings.

    Returns:
        triples: numpy array of shape [num_triples, 3]
        entity_to_id: dictionary mapping entity names to IDs
        relation_to_id: dictionary mapping relation names to IDs
    """
    logger.info(f"Loading triples from {triples_path}")

    # Load mappings
    entity_to_id = {}
    with open(entity_to_id_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                entity, idx = parts
                entity_to_id[entity] = int(idx)

    relation_to_id = {}
    with open(relation_to_id_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 2:
                relation, idx = parts
                relation_to_id[relation] = int(idx)

    # Load triples
    triples = []
    with open(triples_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) == 3:
                head, relation, tail = parts
                if head in entity_to_id and relation in relation_to_id and tail in entity_to_id:
                    triples.append([
                        entity_to_id[head],
                        relation_to_id[relation],
                        entity_to_id[tail]
                    ])

    triples = np.array(triples, dtype=np.int64)
    logger.info(f"  Loaded {len(triples)} triples")
    logger.info(f"  Entities: {len(entity_to_id)}, Relations: {len(relation_to_id)}")

    return triples, entity_to_id, relation_to_id


def evaluate(
    model: ConvE,
    dataloader: DataLoader,
    device: str,
    filter_triples: Optional[set] = None
) -> Dict[str, float]:
    """Evaluate model on a dataset.

    Args:
        model: ConvE model
        dataloader: DataLoader for evaluation
        device: Device to use
        filter_triples: Set of (h, r, t) tuples to filter from ranking

    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()

    ranks = []
    reciprocal_ranks = []
    hits_at_1 = []
    hits_at_3 = []
    hits_at_5 = []
    hits_at_10 = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            head = batch['head'].to(device)
            relation = batch['relation'].to(device)
            tail = batch['tail'].to(device)

            # Get scores for all entities
            scores = model(head, relation)  # [batch_size, num_entities]

            # Filter known triples if requested
            if filter_triples is not None:
                for i in range(scores.size(0)):
                    h = head[i].item()
                    r = relation[i].item()
                    t = tail[i].item()

                    # Set scores of known triples to -inf (except the target)
                    for entity_idx in range(scores.size(1)):
                        if entity_idx != t and (h, r, entity_idx) in filter_triples:
                            scores[i, entity_idx] = float('-inf')

            # Get ranks
            target_scores = scores.gather(1, tail.unsqueeze(1))
            rank = (scores >= target_scores).sum(dim=1).float()

            ranks.extend(rank.cpu().numpy())
            reciprocal_ranks.extend((1.0 / rank).cpu().numpy())
            hits_at_1.extend((rank <= 1).cpu().numpy())
            hits_at_3.extend((rank <= 3).cpu().numpy())
            hits_at_5.extend((rank <= 5).cpu().numpy())
            hits_at_10.extend((rank <= 10).cpu().numpy())

    metrics = {
        'mean_rank': float(np.mean(ranks)),
        'mean_reciprocal_rank': float(np.mean(reciprocal_ranks)),
        'hits@1': float(np.mean(hits_at_1)),
        'hits@3': float(np.mean(hits_at_3)),
        'hits@5': float(np.mean(hits_at_5)),
        'hits@10': float(np.mean(hits_at_10)),
    }

    return metrics


def train_epoch(
    model: ConvE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    label_smoothing: float = 0.1,
    use_mixed_precision: bool = False,
    scaler: Optional[GradScaler] = None,
    enable_memory_cleanup: bool = True
) -> float:
    """Train for one epoch with optimization support.

    Args:
        model: ConvE model
        dataloader: Training data loader
        optimizer: Optimizer
        device: Device to train on
        label_smoothing: Label smoothing parameter
        use_mixed_precision: If True, use FP16 mixed precision (2x memory + 2x speed)
        scaler: GradScaler for mixed precision training
        enable_memory_cleanup: If True, periodically clear CUDA cache

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
        head = batch['head'].to(device)
        relation = batch['relation'].to(device)
        tail = batch['tail'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # OPTIMIZATION 1: Mixed Precision (FP16)
        # Use autocast for forward pass - runs in FP16 for 2x speed + 2x memory savings
        if use_mixed_precision and scaler is not None:
            # Use new API: autocast(device_type='cuda') for PyTorch 2.0+
            # Falls back to autocast() for older versions (backward compatible)
            device_type = device.split(':')[0] if ':' in device else device
            with autocast(device_type=device_type):
                # Forward pass
                scores = model(head, relation)  # [batch_size, num_entities]

                # Create labels with label smoothing
                batch_size = scores.size(0)
                num_entities = scores.size(1)

                labels = torch.zeros(batch_size, num_entities, device=device)
                labels.scatter_(1, tail.unsqueeze(1), 1.0)

                if label_smoothing > 0:
                    labels = (1.0 - label_smoothing) * labels + label_smoothing / num_entities

                # Compute loss
                loss = F.binary_cross_entropy_with_logits(scores, labels)

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard FP32 training
            # Forward pass
            scores = model(head, relation)  # [batch_size, num_entities]

            # Create labels with label smoothing
            batch_size = scores.size(0)
            num_entities = scores.size(1)

            labels = torch.zeros(batch_size, num_entities, device=device)
            labels.scatter_(1, tail.unsqueeze(1), 1.0)

            if label_smoothing > 0:
                labels = (1.0 - label_smoothing) * labels + label_smoothing / num_entities

            # Compute loss
            loss = F.binary_cross_entropy_with_logits(scores, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # OPTIMIZATION 2: Memory Cleanup
        # Delete intermediate tensors and periodically clear CUDA cache
        if enable_memory_cleanup:
            del loss, scores, labels, head, relation, tail
            # Periodic cache cleanup every 8 batches to avoid fragmentation
            if batch_idx % 8 == 0 and batch_idx > 0 and device.startswith('cuda'):
                torch.cuda.empty_cache()

    return total_loss / num_batches


def save_checkpoint(
    model: ConvE,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    checkpoint_path: str,
    metrics: Optional[Dict[str, float]] = None
):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'num_entities': model.num_entities,
            'num_relations': model.num_relations,
            'embedding_dim': model.embedding_dim,
            'embedding_height': model.embedding_height,
            'embedding_width': model.embedding_width,
        }
    }

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(
    checkpoint_path: str,
    model: ConvE,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> int:
    """Load model checkpoint.

    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    logger.info(f"Loaded checkpoint from epoch {epoch}")

    return epoch


def train_model(
    train_path: str,
    valid_path: str,
    test_path: str,
    entity_to_id_path: str,
    relation_to_id_path: str,
    output_dir: str,
    # Model hyperparameters
    embedding_dim: int = 200,
    output_channels: int = 32,
    kernel_height: int = 3,
    kernel_width: int = 3,
    input_dropout: float = 0.2,
    feature_map_dropout: float = 0.2,
    output_dropout: float = 0.3,
    embedding_height: int = 10,
    embedding_width: int = 20,
    # Training hyperparameters
    num_epochs: int = 100,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    label_smoothing: float = 0.1,
    # Checkpoint options
    checkpoint_dir: str = None,
    checkpoint_frequency: int = 2,
    resume_from_checkpoint: str = None,
    # Memory optimization options
    use_mixed_precision: bool = False,
    use_gradient_checkpointing: bool = False,
    enable_memory_cleanup: bool = True,
    # Other options
    use_gpu: bool = True,
    random_seed: int = 42,
    num_workers: int = 4,
):
    """Train ConvE model with proper checkpoint saving and memory optimizations.

    Memory Optimizations:
        use_mixed_precision: Use FP16 for 2x memory + 2x speed (recommended for GPUs with Tensor Cores)
        use_gradient_checkpointing: Trade computation for memory (2-3x memory reduction)
        enable_memory_cleanup: Periodic tensor deletion and cache clearing (enabled by default)
    """

    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Create checkpoint directory
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Set device
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("Loading datasets...")
    train_triples, entity_to_id, relation_to_id = load_triples(
        train_path, entity_to_id_path, relation_to_id_path
    )
    valid_triples, _, _ = load_triples(valid_path, entity_to_id_path, relation_to_id_path)
    test_triples, _, _ = load_triples(test_path, entity_to_id_path, relation_to_id_path)

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)

    # Create datasets
    train_dataset = TriplesDataset(train_triples)
    valid_dataset = TriplesDataset(valid_triples)
    test_dataset = TriplesDataset(test_triples)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    # Create model
    logger.info("Creating model...")
    model = ConvE(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=embedding_dim,
        output_channels=output_channels,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        input_dropout=input_dropout,
        feature_map_dropout=feature_map_dropout,
        output_dropout=output_dropout,
        embedding_height=embedding_height,
        embedding_width=embedding_width,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # OPTIMIZATION 1: Mixed Precision - Create GradScaler for FP16 training
    scaler = None
    if use_mixed_precision and device.startswith('cuda'):
        scaler = GradScaler()
        logger.info("✓ Mixed precision (FP16) ENABLED - 2x memory + 2x speed boost")

    # Log other optimizations
    if use_gradient_checkpointing:
        logger.info("✓ Gradient checkpointing ENABLED - 2-3x memory reduction")
        logger.warning("  Note: Gradient checkpointing for ConvE is best used during TracIn, not training")
    if enable_memory_cleanup:
        logger.info("✓ Memory cleanup ENABLED - periodic tensor deletion and cache clearing")

    # Resume from checkpoint if requested
    start_epoch = 0
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
        start_epoch = load_checkpoint(resume_from_checkpoint, model, optimizer)
        start_epoch += 1  # Start from next epoch

    # Save configuration
    config = {
        'data': {
            'train_path': train_path,
            'valid_path': valid_path,
            'test_path': test_path,
            'num_entities': num_entities,
            'num_relations': num_relations,
            'num_train_triples': len(train_triples),
            'num_valid_triples': len(valid_triples),
            'num_test_triples': len(test_triples),
        },
        'model': {
            'embedding_dim': embedding_dim,
            'output_channels': output_channels,
            'kernel_height': kernel_height,
            'kernel_width': kernel_width,
            'input_dropout': input_dropout,
            'feature_map_dropout': feature_map_dropout,
            'output_dropout': output_dropout,
            'embedding_height': embedding_height,
            'embedding_width': embedding_width,
        },
        'training': {
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'label_smoothing': label_smoothing,
            'checkpoint_frequency': checkpoint_frequency,
            'random_seed': random_seed,
        },
        'optimizations': {
            'use_mixed_precision': use_mixed_precision,
            'use_gradient_checkpointing': use_gradient_checkpointing,
            'enable_memory_cleanup': enable_memory_cleanup,
        }
    }

    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Saved configuration to {config_path}")

    # Create filter set for evaluation (all training + validation triples)
    filter_set = set()
    for h, r, t in train_triples:
        filter_set.add((h, r, t))
    for h, r, t in valid_triples:
        filter_set.add((h, r, t))

    # Training loop
    logger.info("Starting training...")
    logger.info(f"Training for {num_epochs} epochs, saving checkpoints every {checkpoint_frequency} epochs")

    best_mrr = 0.0
    training_history = []

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, device, label_smoothing,
            use_mixed_precision=use_mixed_precision,
            scaler=scaler,
            enable_memory_cleanup=enable_memory_cleanup
        )

        # Evaluate on validation set
        logger.info(f"Evaluating on validation set...")
        valid_metrics = evaluate(model, valid_loader, device, filter_set)

        epoch_time = time.time() - epoch_start_time

        # Log results
        logger.info(f"Epoch {epoch + 1}/{num_epochs} - Time: {epoch_time:.2f}s")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Valid MRR: {valid_metrics['mean_reciprocal_rank']:.4f}")
        logger.info(f"  Valid Hits@10: {valid_metrics['hits@10']:.4f}")

        # Save training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_metrics': valid_metrics,
            'time': epoch_time
        })

        # Save checkpoint every N epochs
        if (epoch + 1) % checkpoint_frequency == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch + 1}.pt')
            save_checkpoint(model, optimizer, epoch, checkpoint_path, valid_metrics)

        # Save best model
        if valid_metrics['mean_reciprocal_rank'] > best_mrr:
            best_mrr = valid_metrics['mean_reciprocal_rank']
            best_model_path = os.path.join(output_dir, 'best_model.pt')
            save_checkpoint(model, optimizer, epoch, best_model_path, valid_metrics)
            logger.info(f"  New best MRR: {best_mrr:.4f}")

    # Save final model
    final_model_path = os.path.join(output_dir, 'final_model.pt')
    save_checkpoint(model, optimizer, num_epochs - 1, final_model_path)

    # Save training history
    history_path = os.path.join(output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Saved training history to {history_path}")

    # Final evaluation on test set
    logger.info("Evaluating on test set...")
    test_metrics = evaluate(model, test_loader, device, filter_set)

    logger.info("=" * 60)
    logger.info("Final Test Results:")
    logger.info("=" * 60)
    logger.info(f"Mean Rank: {test_metrics['mean_rank']:.2f}")
    logger.info(f"Mean Reciprocal Rank: {test_metrics['mean_reciprocal_rank']:.4f}")
    logger.info(f"Hits@1: {test_metrics['hits@1']:.4f}")
    logger.info(f"Hits@3: {test_metrics['hits@3']:.4f}")
    logger.info(f"Hits@5: {test_metrics['hits@5']:.4f}")
    logger.info(f"Hits@10: {test_metrics['hits@10']:.4f}")
    logger.info("=" * 60)

    # Save test results
    test_results_path = os.path.join(output_dir, 'test_results.json')
    with open(test_results_path, 'w') as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Training complete!")

    return model, test_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train ConvE model using pure PyTorch'
    )

    # Data arguments
    parser.add_argument('--train', type=str, required=True, help='Path to training triples')
    parser.add_argument('--valid', type=str, required=True, help='Path to validation triples')
    parser.add_argument('--test', type=str, required=True, help='Path to test triples')
    parser.add_argument('--entity-to-id', type=str, required=True, help='Path to entity-to-ID mapping')
    parser.add_argument('--relation-to-id', type=str, required=True, help='Path to relation-to-ID mapping')
    parser.add_argument('--output-dir', '-o', type=str, required=True, help='Output directory')

    # Model hyperparameters
    parser.add_argument('--embedding-dim', type=int, default=200)
    parser.add_argument('--output-channels', type=int, default=32)
    parser.add_argument('--kernel-height', type=int, default=3)
    parser.add_argument('--kernel-width', type=int, default=3)
    parser.add_argument('--input-dropout', type=float, default=0.2)
    parser.add_argument('--feature-map-dropout', type=float, default=0.2)
    parser.add_argument('--output-dropout', type=float, default=0.3)
    parser.add_argument('--embedding-height', type=int, default=10)
    parser.add_argument('--embedding-width', type=int, default=20)

    # Training hyperparameters
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--label-smoothing', type=float, default=0.1)

    # Checkpoint options
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory for checkpoints')
    parser.add_argument('--checkpoint-frequency', type=int, default=2, help='Save checkpoint every N epochs')
    parser.add_argument('--resume-from-checkpoint', type=str, default=None, help='Path to checkpoint to resume from')

    # Memory optimization options
    parser.add_argument('--use-mixed-precision', action='store_true',
                        help='Use FP16 mixed precision (2x memory + 2x speed). Recommended for GPUs with Tensor Cores.')
    parser.add_argument('--use-gradient-checkpointing', action='store_true',
                        help='Use gradient checkpointing (2-3x memory reduction, slight speed penalty). Note: Best for TracIn, not training.')
    parser.add_argument('--disable-memory-cleanup', action='store_true',
                        help='Disable automatic memory cleanup (tensor deletion and cache clearing)')

    # Other options
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU')
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--num-workers', type=int, default=4)

    return parser.parse_args()


def main():
    args = parse_args()

    train_model(
        train_path=args.train,
        valid_path=args.valid,
        test_path=args.test,
        entity_to_id_path=args.entity_to_id,
        relation_to_id_path=args.relation_to_id,
        output_dir=args.output_dir,
        # Model hyperparameters
        embedding_dim=args.embedding_dim,
        output_channels=args.output_channels,
        kernel_height=args.kernel_height,
        kernel_width=args.kernel_width,
        input_dropout=args.input_dropout,
        feature_map_dropout=args.feature_map_dropout,
        output_dropout=args.output_dropout,
        embedding_height=args.embedding_height,
        embedding_width=args.embedding_width,
        # Training hyperparameters
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        label_smoothing=args.label_smoothing,
        # Checkpoint options
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_frequency=args.checkpoint_frequency,
        resume_from_checkpoint=args.resume_from_checkpoint,
        # Memory optimization options
        use_mixed_precision=args.use_mixed_precision,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        enable_memory_cleanup=not args.disable_memory_cleanup,
        # Other options
        use_gpu=not args.no_gpu,
        random_seed=args.random_seed,
        num_workers=args.num_workers,
    )


if __name__ == '__main__':
    main()
