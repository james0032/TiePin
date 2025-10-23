"""
TracIn (Tracing Influence) implementation for ConvE model.

TracIn computes the influence of training examples on test predictions by
approximating the influence through gradients. This helps understand which
training triples are most important for specific predictions.

Reference: Pruthi et al. "Estimating Training Data Influence by Tracing Gradient Descent"
"""

import json
import logging
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pykeen.models import Model
from pykeen.triples import CoreTriplesFactory
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TracInAnalyzer:
    """TracIn analyzer for computing training data influence.

    This class computes the influence of training examples on test predictions
    using gradient-based approximations.
    """

    def __init__(
        self,
        model: Model,
        loss_fn: str = 'bce',
        device: Optional[str] = None,
        use_last_layers_only: bool = False,
        last_layer_names: Optional[List[str]] = None,
        num_last_layers: int = 2
    ):
        """Initialize TracIn analyzer.

        Args:
            model: Trained PyKEEN model
            loss_fn: Loss function to use ('bce' for binary cross-entropy)
            device: Device to run computation on
            use_last_layers_only: If True, only compute gradients for last layers (faster)
            last_layer_names: List of parameter names to track. If None and use_last_layers_only=True,
                             will auto-detect based on num_last_layers. Common patterns:
                             - For ConvE: ['interaction.linear.weight', 'interaction.linear.bias']
            num_last_layers: Number of last layers to track when auto-detecting (default: 2)
                            Options: 1 (fastest, least accurate), 2-3 (recommended), 5+ (slower)
        """
        self.model = model
        self.loss_fn = loss_fn
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.use_last_layers_only = use_last_layers_only
        self.num_last_layers = num_last_layers

        # Auto-detect or set last layers to track
        if use_last_layers_only:
            if last_layer_names is None:
                # Auto-detect last N layers
                self.tracked_params = self._auto_detect_last_layers(num_layers=num_last_layers)
                logger.info(f"Auto-detected last {num_last_layers} layer(s): {self.tracked_params}")
            else:
                self.tracked_params = set(last_layer_names)
                logger.info(f"Using specified {len(last_layer_names)} layer(s): {self.tracked_params}")
        else:
            # Track all parameters
            self.tracked_params = None
            logger.info("Tracking gradients for ALL model parameters")

    def _auto_detect_last_layers(self, num_layers: int = 2) -> set:
        """Auto-detect last N layers of the model.

        For ConvE, this typically includes (in order from last to first):
        1. interaction.linear.bias (final scoring layer bias)
        2. interaction.linear.weight (final scoring layer weight)
        3. interaction.bn2.* (final batch norm, if num_layers >= 3)
        4. Earlier conv/batch norm layers (if num_layers > 3)

        Args:
            num_layers: Number of last layers to track (default: 2)

        Returns:
            Set of parameter names to track
        """
        all_params = list(self.model.named_parameters())

        if len(all_params) == 0:
            logger.warning("Model has no parameters!")
            return None

        # Strategy 1: Look for semantic final layer patterns (preferred)
        last_layers = set()

        # Look for final linear/fc layer (most common final layer)
        linear_params = []
        for name, _ in all_params:
            if any(pattern in name for pattern in ['linear.weight', 'linear.bias', 'fc.weight', 'fc.bias']):
                # Check if it's NOT in an embedding layer (we want final scoring layer)
                if 'embedding' not in name:
                    linear_params.append(name)

        # If we found final linear layers, use them as a base
        if linear_params:
            last_layers.update(linear_params)

            # If user wants more layers, add preceding layers
            if num_layers > len(linear_params):
                # Find the index of the first linear layer in all_params
                first_linear_idx = None
                for i, (name, _) in enumerate(all_params):
                    if name in linear_params:
                        first_linear_idx = i
                        break

                if first_linear_idx is not None:
                    # Add layers before the linear layer
                    extra_layers_needed = num_layers - len(linear_params)
                    start_idx = max(0, first_linear_idx - extra_layers_needed)

                    for i in range(start_idx, first_linear_idx):
                        name, _ = all_params[i]
                        # Skip embedding layers
                        if 'embedding' not in name:
                            last_layers.add(name)

            return last_layers

        # Strategy 2: Just take the last N parameters (fallback)
        logger.info(f"Could not find semantic final layers, using last {num_layers} parameter(s)")

        num_to_take = min(num_layers, len(all_params))
        for name, _ in all_params[-num_to_take:]:
            last_layers.add(name)

        if last_layers:
            return last_layers

        # Final fallback: track all (if model is very small)
        logger.warning("Could not auto-detect last layers, will track all parameters")
        return None

    def compute_gradient(
        self,
        head: int,
        relation: int,
        tail: int,
        label: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient of loss with respect to model parameters for a single triple.

        Args:
            head: Head entity index
            relation: Relation index
            tail: Tail entity index
            label: Label for the triple (1.0 for positive, 0.0 for negative)

        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        # Keep model in eval mode to avoid BatchNorm issues with batch_size=1
        # But enable gradient computation for parameters
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True
        self.model.zero_grad()

        # Forward pass
        hr_batch = torch.LongTensor([[head, relation]]).to(self.device)
        scores = self.model.score_t(hr_batch)  # Shape: (1, num_entities)

        # Get score for the specific tail
        score = scores[0, tail]

        # Compute loss
        if self.loss_fn == 'bce':
            # Binary cross-entropy loss
            target = torch.tensor([label], dtype=torch.float32, device=self.device)
            loss = F.binary_cross_entropy_with_logits(score.unsqueeze(0), target)
        else:
            raise ValueError(f"Unknown loss function: {self.loss_fn}")

        # Backward pass
        loss.backward()

        # Collect gradients (only for tracked parameters if specified)
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Only include if we're tracking all params OR this param is in tracked set
                if self.tracked_params is None or name in self.tracked_params:
                    gradients[name] = param.grad.clone()

        return gradients

    def compute_influence(
        self,
        train_triple: Tuple[int, int, int],
        test_triple: Tuple[int, int, int],
        learning_rate: float = 1e-3
    ) -> float:
        """Compute influence of a training triple on a test triple.

        The influence is computed as:
            influence = learning_rate * dot(grad_train, grad_test)

        A positive influence means the training example pushes the model
        towards the correct prediction for the test example.

        Args:
            train_triple: (head, relation, tail) training triple
            test_triple: (head, relation, tail) test triple
            learning_rate: Learning rate used during training

        Returns:
            Influence score (scalar)
        """
        # Compute gradients for training triple (positive example)
        train_h, train_r, train_t = train_triple
        grad_train = self.compute_gradient(train_h, train_r, train_t, label=1.0)

        # Compute gradients for test triple (positive example)
        test_h, test_r, test_t = test_triple
        grad_test = self.compute_gradient(test_h, test_r, test_t, label=1.0)

        # Compute dot product of gradients
        influence = 0.0
        for name in grad_train:
            if name in grad_test:
                # Flatten and compute dot product
                grad_train_flat = grad_train[name].flatten()
                grad_test_flat = grad_test[name].flatten()
                influence += torch.dot(grad_train_flat, grad_test_flat).item()

        # Scale by learning rate
        influence *= learning_rate

        return influence

    def compute_batch_individual_gradients(
        self,
        triples_batch: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:
        """Compute individual gradients for each triple in a batch.

        Uses per-sample gradient computation by iterating through samples
        but keeping them on GPU for faster processing.

        Args:
            triples_batch: Tensor of shape (batch_size, 3) with [head, relation, tail]

        Returns:
            List of gradient dictionaries, one per triple in the batch
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = True

        batch_size = triples_batch.shape[0]
        batch_gradients = []

        # Move entire batch to device once
        triples_batch = triples_batch.to(self.device)

        # Process each sample to get individual gradients
        for i in range(batch_size):
            self.model.zero_grad()

            h, r, t = triples_batch[i]
            hr_batch = torch.LongTensor([[int(h), int(r)]]).to(self.device)
            scores = self.model.score_t(hr_batch)
            score = scores[0, int(t)]

            # Compute loss for this sample
            if self.loss_fn == 'bce':
                target = torch.tensor([1.0], dtype=torch.float32, device=self.device)
                loss = F.binary_cross_entropy_with_logits(score.unsqueeze(0), target)
            else:
                raise ValueError(f"Unknown loss function: {self.loss_fn}")

            # Backward pass
            loss.backward()

            # Collect gradients for this sample (only tracked parameters)
            sample_grads = {}
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    # Only include if we're tracking all params OR this param is in tracked set
                    if self.tracked_params is None or name in self.tracked_params:
                        sample_grads[name] = param.grad.clone().detach()

            batch_gradients.append(sample_grads)

        return batch_gradients

    def compute_influences_for_test_triple(
        self,
        test_triple: Tuple[int, int, int],
        training_triples: CoreTriplesFactory,
        learning_rate: float = 1e-3,
        top_k: Optional[int] = None,
        batch_size: int = 256
    ) -> List[Dict[str, any]]:
        """Compute influences of all training triples on a single test triple.

        Args:
            test_triple: (head, relation, tail) test triple
            training_triples: Training triples factory
            learning_rate: Learning rate used during training
            top_k: If specified, return only top-k most influential triples
            batch_size: Batch size for GPU processing (larger = faster, but more memory)

        Returns:
            List of dictionaries with training triple info and influence score
        """
        logger.info(f"Computing influences for test triple: {test_triple}")
        logger.info(f"Using batch size: {batch_size} on device: {self.device}")

        # Compute gradient for test triple once
        test_h, test_r, test_t = test_triple
        grad_test = self.compute_gradient(test_h, test_r, test_t, label=1.0)

        # Flatten test gradients once for efficient dot products
        grad_test_flat = {}
        for name in grad_test:
            grad_test_flat[name] = grad_test[name].flatten()

        influences = []
        train_triples_array = training_triples.mapped_triples
        if not isinstance(train_triples_array, torch.Tensor):
            train_triples_array = torch.from_numpy(train_triples_array.numpy())

        num_train = len(train_triples_array)

        # Process training triples in batches
        for batch_start in tqdm(range(0, num_train, batch_size), desc="Computing influences"):
            batch_end = min(batch_start + batch_size, num_train)
            batch_triples = train_triples_array[batch_start:batch_end]

            # Compute gradients for all triples in the batch
            batch_gradients = self.compute_batch_individual_gradients(batch_triples)

            # Compute influence for each training triple in the batch
            for i, grad_train in enumerate(batch_gradients):
                triple_idx = batch_start + i
                h, r, t = train_triples_array[triple_idx]

                # Compute dot product with test gradient
                influence = 0.0
                for name in grad_train:
                    if name in grad_test_flat:
                        grad_train_flat = grad_train[name].flatten()
                        influence += torch.dot(grad_train_flat, grad_test_flat[name]).item()

                influence *= learning_rate

                influences.append({
                    'train_head': int(h),
                    'train_relation': int(r),
                    'train_tail': int(t),
                    'influence': influence
                })

        # Sort by influence (descending by absolute value)
        influences.sort(key=lambda x: abs(x['influence']), reverse=True)

        # Return top-k if specified
        if top_k is not None:
            influences = influences[:top_k]

        return influences

    def analyze_test_set(
        self,
        test_triples: CoreTriplesFactory,
        training_triples: CoreTriplesFactory,
        learning_rate: float = 1e-3,
        top_k: int = 10,
        max_test_triples: Optional[int] = None,
        output_path: Optional[str] = None,
        batch_size: int = 256
    ) -> Dict[str, any]:
        """Analyze influence of training data on test predictions.

        Args:
            test_triples: Test triples factory
            training_triples: Training triples factory
            learning_rate: Learning rate used during training
            top_k: Number of top influential training triples to return per test triple
            max_test_triples: Maximum number of test triples to analyze (None for all)
            output_path: Optional path to save results
            batch_size: Batch size for processing training triples

        Returns:
            Dictionary with influence analysis results
        """
        logger.info(f"Analyzing influences for test set...")

        results = []
        test_triple_list = [(int(h), int(r), int(t)) for h, r, t in test_triples.mapped_triples]

        if max_test_triples is not None:
            test_triple_list = test_triple_list[:max_test_triples]

        for test_triple in tqdm(test_triple_list, desc="Analyzing test triples"):
            influences = self.compute_influences_for_test_triple(
                test_triple=test_triple,
                training_triples=training_triples,
                learning_rate=learning_rate,
                top_k=top_k,
                batch_size=batch_size
            )

            results.append({
                'test_head': test_triple[0],
                'test_relation': test_triple[1],
                'test_tail': test_triple[2],
                'top_influences': influences
            })

        analysis = {
            'num_test_triples': len(results),
            'num_training_triples': training_triples.num_triples,
            'top_k': top_k,
            'learning_rate': learning_rate,
            'results': results
        }

        # Save results if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            logger.info(f"Saved TracIn analysis to {output_path}")

        return analysis

    def compute_self_influence(
        self,
        training_triples: CoreTriplesFactory,
        learning_rate: float = 1e-3,
        output_path: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """Compute self-influence for each training triple.

        Self-influence measures how much a training example influences itself,
        which can indicate the importance or difficulty of that example.

        Args:
            training_triples: Training triples factory
            learning_rate: Learning rate used during training
            output_path: Optional path to save results

        Returns:
            List of dictionaries with triple info and self-influence score
        """
        logger.info("Computing self-influences for training set...")

        influences = []

        for h, r, t in tqdm(training_triples.mapped_triples, desc="Computing self-influences"):
            triple = (int(h), int(r), int(t))

            # Compute influence on itself
            grad = self.compute_gradient(*triple, label=1.0)

            # Compute squared L2 norm of gradient
            self_influence = 0.0
            for name in grad:
                grad_flat = grad[name].flatten()
                self_influence += torch.dot(grad_flat, grad_flat).item()

            self_influence *= learning_rate

            influences.append({
                'head': triple[0],
                'relation': triple[1],
                'tail': triple[2],
                'self_influence': self_influence
            })

        # Sort by self-influence (descending)
        influences.sort(key=lambda x: x['self_influence'], reverse=True)

        # Save results if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(influences, f, indent=2)
            logger.info(f"Saved self-influence analysis to {output_path}")

        return influences

    def _extract_predicate_from_json(self, relation_str: str) -> str:
        """Extract predicate value from JSON-formatted relation string.

        Args:
            relation_str: Either a JSON string like '{"predicate": "biolink:affects", ...}'
                         or a simple string like 'predicate:27'

        Returns:
            Extracted predicate value (e.g., 'biolink:affects') or original string if not JSON
        """
        try:
            # Try to parse as JSON
            relation_obj = json.loads(relation_str)
            # Extract the predicate field
            if isinstance(relation_obj, dict) and 'predicate' in relation_obj:
                return relation_obj['predicate']
        except (json.JSONDecodeError, TypeError):
            # Not JSON, return as-is
            pass
        return relation_str

    def save_influences_to_csv(
        self,
        test_triple: Tuple[int, int, int],
        influences: List[Dict],
        output_path: str,
        id_to_entity: Dict[int, str],
        id_to_relation: Dict[int, str],
        entity_labels: Optional[Dict[int, str]] = None,
        relation_labels: Optional[Dict[int, str]] = None,
        self_influence: Optional[float] = None
    ):
        """Save TracIn influences to CSV with IDs and labels.

        Args:
            test_triple: Test triple (head, relation, tail) as indices
            influences: List of influence dictionaries from compute_influences_for_test_triple
            output_path: Path to save CSV file
            id_to_entity: Mapping from entity index to entity CURIE
            id_to_relation: Mapping from relation index to relation CURIE (may contain JSON strings)
            entity_labels: Optional mapping from entity index to human-readable name
            relation_labels: Optional mapping from relation index to human-readable name
            self_influence: Optional self-influence score for the test triple
        """
        test_h, test_r, test_t = test_triple

        # Create output directory if needed
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header - exactly as requested, with SelfInfluence added
            writer.writerow([
                'TestHead', 'TestHead_label',
                'TestRel', 'TestRel_label',
                'TestTail', 'TestTail_label',
                'TrainHead', 'TrainHead_label',
                'TrainRel', 'TrainRel_label',
                'TrainTail', 'TrainTail_label',
                'TracInScore',
                'SelfInfluence'
            ])

            # Get test triple IDs
            test_h_id = id_to_entity.get(test_h, f'UNKNOWN_{test_h}')
            test_r_id = id_to_relation.get(test_r, f'UNKNOWN_{test_r}')
            test_t_id = id_to_entity.get(test_t, f'UNKNOWN_{test_t}')

            # Get test triple labels (extract predicate from JSON for relations)
            test_h_label = entity_labels.get(test_h, test_h_id) if entity_labels else test_h_id
            if relation_labels and test_r in relation_labels:
                test_r_label = self._extract_predicate_from_json(relation_labels[test_r])
            else:
                test_r_label = self._extract_predicate_from_json(test_r_id)
            test_t_label = entity_labels.get(test_t, test_t_id) if entity_labels else test_t_id

            # Write each influence
            for inf in influences:
                train_h = inf['train_head']
                train_r = inf['train_relation']
                train_t = inf['train_tail']
                score = inf['influence']

                # Get training triple IDs
                train_h_id = id_to_entity.get(train_h, f'UNKNOWN_{train_h}')
                train_r_id = id_to_relation.get(train_r, f'UNKNOWN_{train_r}')
                train_t_id = id_to_entity.get(train_t, f'UNKNOWN_{train_t}')

                # Get training triple labels (extract predicate from JSON for relations)
                train_h_label = entity_labels.get(train_h, train_h_id) if entity_labels else train_h_id
                if relation_labels and train_r in relation_labels:
                    train_r_label = self._extract_predicate_from_json(relation_labels[train_r])
                else:
                    train_r_label = self._extract_predicate_from_json(train_r_id)
                train_t_label = entity_labels.get(train_t, train_t_id) if entity_labels else train_t_id

                writer.writerow([
                    test_h_id, test_h_label,
                    test_r_id, test_r_label,
                    test_t_id, test_t_label,
                    train_h_id, train_h_label,
                    train_r_id, train_r_label,
                    train_t_id, train_t_label,
                    score,
                    self_influence if self_influence is not None else ''
                ])

        logger.info(f"Saved {len(influences)} influences to CSV: {output_path}")


def compute_tracin_influence(
    model: Model,
    test_triple: Tuple[int, int, int],
    training_triples: CoreTriplesFactory,
    learning_rate: float = 1e-3,
    top_k: int = 10,
    device: Optional[str] = None,
    use_last_layers_only: bool = False,
    last_layer_names: Optional[List[str]] = None,
    num_last_layers: int = 2
) -> List[Dict[str, any]]:
    """Convenience function to compute TracIn influences.

    Args:
        model: Trained PyKEEN model
        test_triple: (head, relation, tail) test triple
        training_triples: Training triples factory
        learning_rate: Learning rate used during training
        top_k: Number of top influential training triples to return
        device: Device to run computation on
        use_last_layers_only: If True, only use last layer gradients (much faster)
        last_layer_names: Specific layer names to track (optional)
        num_last_layers: Number of last layers to track (default: 2)

    Returns:
        List of dictionaries with training triple info and influence score
    """
    analyzer = TracInAnalyzer(
        model=model,
        device=device,
        use_last_layers_only=use_last_layers_only,
        last_layer_names=last_layer_names,
        num_last_layers=num_last_layers
    )
    return analyzer.compute_influences_for_test_triple(
        test_triple=test_triple,
        training_triples=training_triples,
        learning_rate=learning_rate,
        top_k=top_k
    )
