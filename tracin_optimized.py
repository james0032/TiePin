"""
Optimized TracIn implementation with advanced strategies.

Implements two major optimization strategies:
1. Random Projection - Project gradients to lower dimensions for faster dot products
2. Influence Sketching - Sample training data and cache test gradients

These can provide 100-1000x speedup over baseline TracIn while maintaining
good approximation quality for influence rankings.
"""

import logging
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from pykeen.models import Model
from pykeen.triples import CoreTriplesFactory

from tracin import TracInAnalyzer

logger = logging.getLogger(__name__)


class TracInAnalyzerOptimized(TracInAnalyzer):
    """Optimized TracIn analyzer with random projection and sampling.

    This class extends TracInAnalyzer with two key optimizations:

    1. Random Projection: Projects high-dimensional gradients to lower dimensions
       using Johnson-Lindenstrauss random projection. This preserves approximate
       dot products while being much faster to compute.

    2. Stratified Sampling: Samples a subset of training data for influence
       computation, with optional stratification by relation or entity.

    Combined with last-layers-only mode, these can provide 100-1000x speedup.
    """

    def __init__(
        self,
        model: Model,
        loss_fn: str = 'bce',
        device: Optional[str] = None,
        use_last_layers_only: bool = False,
        last_layer_names: Optional[List[str]] = None,
        num_last_layers: int = 2,
        # Optimization parameters
        use_projection: bool = False,
        projection_dim: int = 256,
        projection_type: str = 'gaussian'
    ):
        """Initialize optimized TracIn analyzer.

        Args:
            model: Trained PyKEEN model
            loss_fn: Loss function to use
            device: Device to run computation on
            use_last_layers_only: If True, only track last N layers
            last_layer_names: Specific layer names to track
            num_last_layers: Number of last layers to track
            use_projection: If True, use random projection for gradients
            projection_dim: Target dimension for random projection (default: 256)
            projection_type: Type of projection ('gaussian' or 'sparse')
        """
        super().__init__(
            model=model,
            loss_fn=loss_fn,
            device=device,
            use_last_layers_only=use_last_layers_only,
            last_layer_names=last_layer_names,
            num_last_layers=num_last_layers
        )

        # Projection settings
        self.use_projection = use_projection
        self.projection_dim = projection_dim
        self.projection_type = projection_type
        self.projection_matrix = None
        self.original_grad_dim = None

        # Caching
        self.test_gradient_cache = {}

        if use_projection:
            logger.info(f"Using random projection with dim={projection_dim}, type={projection_type}")

    def _create_projection_matrix(self, gradient_dim: int) -> torch.Tensor:
        """Create random projection matrix.

        Args:
            gradient_dim: Dimension of original gradient vector

        Returns:
            Projection matrix of shape (gradient_dim, projection_dim)
        """
        logger.info(f"Creating projection matrix: {gradient_dim} -> {self.projection_dim}")

        if self.projection_type == 'gaussian':
            # Gaussian random projection (most common)
            # Normalized by sqrt(projection_dim) to preserve norms
            P = torch.randn(gradient_dim, self.projection_dim, device=self.device)
            P = P / np.sqrt(self.projection_dim)

        elif self.projection_type == 'sparse':
            # Sparse random projection (even faster)
            # Uses mostly zeros with occasional +1/-1
            density = 1.0 / np.sqrt(gradient_dim)
            P = torch.zeros(gradient_dim, self.projection_dim, device=self.device)

            nnz = int(gradient_dim * self.projection_dim * density)
            indices = np.random.choice(gradient_dim * self.projection_dim, nnz, replace=False)
            rows = indices // self.projection_dim
            cols = indices % self.projection_dim
            values = np.random.choice([-1, 1], nnz)

            for r, c, v in zip(rows, cols, values):
                P[r, c] = v / np.sqrt(self.projection_dim)

        else:
            raise ValueError(f"Unknown projection type: {self.projection_type}")

        return P

    def _gradient_to_vector(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert gradient dictionary to flat vector.

        Args:
            grad_dict: Dictionary of gradients

        Returns:
            Flattened gradient vector
        """
        grad_parts = []
        for name in sorted(grad_dict.keys()):  # Sort for consistency
            grad_parts.append(grad_dict[name].flatten())
        return torch.cat(grad_parts)

    def project_gradient(self, grad_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Project gradient dictionary to lower-dimensional space.

        Args:
            grad_dict: Dictionary of gradients

        Returns:
            Projected gradient vector of shape (projection_dim,)
        """
        # Flatten gradient
        grad_flat = self._gradient_to_vector(grad_dict)

        # Initialize projection matrix if needed
        if self.projection_matrix is None:
            self.original_grad_dim = len(grad_flat)
            self.projection_matrix = self._create_projection_matrix(self.original_grad_dim)

            compression_ratio = self.original_grad_dim / self.projection_dim
            logger.info(f"Gradient compression: {compression_ratio:.1f}x ({self.original_grad_dim:,} -> {self.projection_dim})")

        # Project: grad_flat @ projection_matrix
        projected = grad_flat @ self.projection_matrix

        return projected

    def compute_influence_optimized(
        self,
        train_triple: Tuple[int, int, int],
        test_triple: Tuple[int, int, int],
        learning_rate: float = 1e-3,
        use_cached_test_grad: bool = True
    ) -> float:
        """Compute influence with optimizations.

        Args:
            train_triple: Training triple
            test_triple: Test triple
            learning_rate: Learning rate
            use_cached_test_grad: If True, cache and reuse test gradient

        Returns:
            Influence score
        """
        # Get or compute test gradient
        test_key = tuple(test_triple)

        if use_cached_test_grad and test_key in self.test_gradient_cache:
            if self.use_projection:
                proj_test = self.test_gradient_cache[test_key]
            else:
                grad_test_flat = self.test_gradient_cache[test_key]
        else:
            # Compute test gradient
            grad_test = self.compute_gradient(*test_triple, label=1.0)

            if self.use_projection:
                # Project and cache
                proj_test = self.project_gradient(grad_test)
                self.test_gradient_cache[test_key] = proj_test
            else:
                # Flatten and cache
                grad_test_flat = {name: g.flatten() for name, g in grad_test.items()}
                self.test_gradient_cache[test_key] = grad_test_flat

        # Compute training gradient
        grad_train = self.compute_gradient(*train_triple, label=1.0)

        # Compute influence
        if self.use_projection:
            # Project training gradient
            proj_train = self.project_gradient(grad_train)

            # Dot product in low-dimensional space (much faster!)
            influence = torch.dot(proj_train, proj_test).item()
        else:
            # Standard dot product
            influence = 0.0
            for name in grad_train:
                if name in grad_test_flat:
                    grad_train_flat = grad_train[name].flatten()
                    influence += torch.dot(grad_train_flat, grad_test_flat[name]).item()

        influence *= learning_rate
        return influence

    def compute_influences_sampled(
        self,
        test_triple: Tuple[int, int, int],
        training_triples: CoreTriplesFactory,
        sample_rate: float = 0.1,
        stratify_by: Optional[str] = None,
        learning_rate: float = 1e-3,
        top_k: Optional[int] = None,
        seed: Optional[int] = None
    ) -> List[Dict]:
        """Compute influences using sampled training data.

        Args:
            test_triple: Test triple to analyze
            training_triples: Training triples factory
            sample_rate: Fraction of training data to sample (0.0-1.0)
            stratify_by: Stratification strategy ('relation', 'head', 'tail', or None)
            learning_rate: Learning rate used during training
            top_k: Number of top influences to return
            seed: Random seed for reproducibility

        Returns:
            List of influence dictionaries
        """
        if seed is not None:
            np.random.seed(seed)

        # Sample training indices
        n_train = len(training_triples.mapped_triples)
        n_samples = max(1, int(n_train * sample_rate))

        logger.info(f"Sampling {n_samples} / {n_train} training examples ({sample_rate*100:.1f}%)")

        if stratify_by == 'relation':
            sampled_indices = self._stratified_sample_by_relation(
                training_triples, n_samples
            )
        elif stratify_by == 'head':
            sampled_indices = self._stratified_sample_by_entity(
                training_triples, n_samples, entity_idx=0
            )
        elif stratify_by == 'tail':
            sampled_indices = self._stratified_sample_by_entity(
                training_triples, n_samples, entity_idx=2
            )
        else:
            # Random sampling
            sampled_indices = np.random.choice(n_train, n_samples, replace=False)

        # Compute influences for sampled examples
        influences = []
        for idx in sampled_indices:
            h, r, t = training_triples.mapped_triples[idx]
            train_triple = (int(h), int(r), int(t))

            influence = self.compute_influence_optimized(
                train_triple=train_triple,
                test_triple=test_triple,
                learning_rate=learning_rate
            )

            influences.append({
                'train_head': train_triple[0],
                'train_relation': train_triple[1],
                'train_tail': train_triple[2],
                'train_index': int(idx),
                'influence': influence
            })

        # Sort by absolute influence
        influences.sort(key=lambda x: abs(x['influence']), reverse=True)

        if top_k is not None:
            influences = influences[:top_k]

        return influences

    def _stratified_sample_by_relation(
        self,
        training_triples: CoreTriplesFactory,
        n_samples: int
    ) -> np.ndarray:
        """Sample training examples stratified by relation type.

        Args:
            training_triples: Training triples factory
            n_samples: Number of samples to draw

        Returns:
            Array of sampled indices
        """
        # Group triples by relation
        relation_to_indices = {}
        for idx, (_, r, _) in enumerate(training_triples.mapped_triples):
            r = int(r)
            if r not in relation_to_indices:
                relation_to_indices[r] = []
            relation_to_indices[r].append(idx)

        # Sample proportionally from each relation
        sampled_indices = []
        n_relations = len(relation_to_indices)

        for r, indices in relation_to_indices.items():
            n_from_relation = max(1, int(n_samples / n_relations))
            n_from_relation = min(n_from_relation, len(indices))

            sampled = np.random.choice(indices, n_from_relation, replace=False)
            sampled_indices.extend(sampled)

        # If we don't have enough, sample more randomly
        if len(sampled_indices) < n_samples:
            all_indices = set(range(len(training_triples.mapped_triples)))
            remaining = list(all_indices - set(sampled_indices))
            extra = np.random.choice(remaining, n_samples - len(sampled_indices), replace=False)
            sampled_indices.extend(extra)

        # If we have too many, randomly drop some
        if len(sampled_indices) > n_samples:
            sampled_indices = np.random.choice(sampled_indices, n_samples, replace=False)

        return np.array(sampled_indices)

    def _stratified_sample_by_entity(
        self,
        training_triples: CoreTriplesFactory,
        n_samples: int,
        entity_idx: int = 0
    ) -> np.ndarray:
        """Sample training examples stratified by entity (head or tail).

        Args:
            training_triples: Training triples factory
            n_samples: Number of samples to draw
            entity_idx: 0 for head, 2 for tail

        Returns:
            Array of sampled indices
        """
        # Similar logic to relation stratification
        entity_to_indices = {}
        for idx, triple in enumerate(training_triples.mapped_triples):
            e = int(triple[entity_idx])
            if e not in entity_to_indices:
                entity_to_indices[e] = []
            entity_to_indices[e].append(idx)

        # Sample proportionally
        sampled_indices = []
        n_entities = len(entity_to_indices)

        for e, indices in entity_to_indices.items():
            n_from_entity = max(1, int(n_samples / n_entities))
            n_from_entity = min(n_from_entity, len(indices))

            sampled = np.random.choice(indices, n_from_entity, replace=False)
            sampled_indices.extend(sampled)

        # Adjust to target size
        if len(sampled_indices) < n_samples:
            all_indices = set(range(len(training_triples.mapped_triples)))
            remaining = list(all_indices - set(sampled_indices))
            extra = np.random.choice(remaining, n_samples - len(sampled_indices), replace=False)
            sampled_indices.extend(extra)

        if len(sampled_indices) > n_samples:
            sampled_indices = np.random.choice(sampled_indices, n_samples, replace=False)

        return np.array(sampled_indices)

    def clear_cache(self):
        """Clear gradient cache to free memory."""
        self.test_gradient_cache.clear()
        logger.info("Cleared gradient cache")
