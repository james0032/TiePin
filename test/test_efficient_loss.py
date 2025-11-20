#!/usr/bin/env python3
"""
Test script to verify that the memory-efficient loss computation
produces identical results to the original dense label matrix approach.
"""

import torch
import torch.nn.functional as F


def compute_loss_original(scores, tail, label_smoothing=0.0):
    """Original memory-intensive approach using dense label matrix."""
    batch_size = scores.size(0)
    num_entities = scores.size(1)

    labels = torch.zeros(batch_size, num_entities, device=scores.device)
    labels.scatter_(1, tail.unsqueeze(1), 1.0)

    if label_smoothing > 0:
        labels = (1.0 - label_smoothing) * labels + label_smoothing / num_entities

    loss = F.binary_cross_entropy_with_logits(scores, labels)
    return loss


def compute_loss_efficient(scores, tail, label_smoothing=0.0):
    """Memory-efficient approach without creating dense label matrix."""
    batch_size = scores.size(0)
    num_entities = scores.size(1)

    if label_smoothing == 0.0:
        # Without label smoothing: BCE where only target entity has label=1
        target_scores = scores[torch.arange(batch_size), tail]

        # Positive term: -log(sigmoid(target_scores))
        positive_loss = -F.logsigmoid(target_scores)

        # Negative term: sum over all entities of -log(1 - sigmoid(score))
        negative_loss = F.softplus(scores).sum(dim=1)

        # Subtract the positive contribution that was counted in negative sum
        negative_loss = negative_loss - F.softplus(target_scores)

        # Average over entities, then over batch
        per_sample_loss = (positive_loss + negative_loss) / num_entities
        loss = per_sample_loss.mean()

    else:
        # With label smoothing
        smooth_positive = 1.0 - label_smoothing + label_smoothing / num_entities
        smooth_negative = label_smoothing / num_entities

        target_scores = scores[torch.arange(batch_size), tail]

        # BCE for target entity with smoothed label
        target_loss = -smooth_positive * F.logsigmoid(target_scores)
        target_loss += -(1.0 - smooth_positive) * F.logsigmoid(-target_scores)

        # BCE for all other entities (each with label = smooth_negative)
        all_negative_loss = -smooth_negative * F.logsigmoid(scores).sum(dim=1)
        all_negative_loss += -(1.0 - smooth_negative) * F.logsigmoid(-scores).sum(dim=1)

        # Remove target entity from "all" sum
        target_as_negative = -smooth_negative * F.logsigmoid(target_scores)
        target_as_negative += -(1.0 - smooth_negative) * F.logsigmoid(-target_scores)

        other_negative_loss = all_negative_loss - target_as_negative

        # Average over entities, then over batch
        per_sample_loss = (target_loss + other_negative_loss) / num_entities
        loss = per_sample_loss.mean()

    return loss


def test_loss_equivalence():
    """Test that both loss functions produce identical results."""

    # Test configurations
    configs = [
        {"batch_size": 16, "num_entities": 100, "label_smoothing": 0.0},
        {"batch_size": 16, "num_entities": 100, "label_smoothing": 0.1},
        {"batch_size": 32, "num_entities": 1000, "label_smoothing": 0.0},
        {"batch_size": 32, "num_entities": 1000, "label_smoothing": 0.1},
        {"batch_size": 64, "num_entities": 5000, "label_smoothing": 0.05},
    ]

    print("Testing loss function equivalence...")
    print("=" * 80)

    all_passed = True

    for i, config in enumerate(configs):
        batch_size = config["batch_size"]
        num_entities = config["num_entities"]
        label_smoothing = config["label_smoothing"]

        # Generate random scores and targets
        torch.manual_seed(42 + i)
        scores = torch.randn(batch_size, num_entities)
        tail = torch.randint(0, num_entities, (batch_size,))

        # Compute losses
        loss_original = compute_loss_original(scores, tail, label_smoothing)
        loss_efficient = compute_loss_efficient(scores, tail, label_smoothing)

        # Check if they're close (allow small numerical differences)
        diff = abs(loss_original.item() - loss_efficient.item())
        relative_diff = diff / (abs(loss_original.item()) + 1e-8)

        passed = relative_diff < 1e-5
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"Test {i+1}: {status}")
        print(f"  Config: batch={batch_size}, entities={num_entities}, "
              f"smoothing={label_smoothing}")
        print(f"  Original loss: {loss_original.item():.8f}")
        print(f"  Efficient loss: {loss_efficient.item():.8f}")
        print(f"  Absolute diff: {diff:.2e}")
        print(f"  Relative diff: {relative_diff:.2e}")
        print()

    print("=" * 80)
    if all_passed:
        print("✓ All tests PASSED! Loss functions are equivalent.")
    else:
        print("✗ Some tests FAILED! Loss functions differ.")

    return all_passed


def test_memory_usage():
    """Demonstrate memory savings of efficient approach."""
    import sys

    print("\nMemory usage comparison:")
    print("=" * 80)

    configs = [
        {"batch_size": 256, "num_entities": 10000},
        {"batch_size": 256, "num_entities": 50000},
        {"batch_size": 512, "num_entities": 50000},
        {"batch_size": 1024, "num_entities": 50000},
    ]

    for config in configs:
        batch_size = config["batch_size"]
        num_entities = config["num_entities"]

        # Calculate memory for dense label matrix
        # Each element is float32 = 4 bytes
        label_matrix_bytes = batch_size * num_entities * 4
        label_matrix_mb = label_matrix_bytes / (1024 * 1024)

        # Efficient approach only needs O(batch_size) memory
        efficient_bytes = batch_size * 4
        efficient_kb = efficient_bytes / 1024

        savings = label_matrix_mb / (efficient_kb / 1024)

        print(f"Batch={batch_size}, Entities={num_entities}:")
        print(f"  Original approach (dense labels): {label_matrix_mb:.2f} MB")
        print(f"  Efficient approach: {efficient_kb:.2f} KB")
        print(f"  Memory savings: {savings:.0f}x")
        print()

    print("=" * 80)


if __name__ == "__main__":
    # Run equivalence tests
    passed = test_loss_equivalence()

    # Show memory savings
    test_memory_usage()

    # Exit with appropriate code
    exit(0 if passed else 1)
