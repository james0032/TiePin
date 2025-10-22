"""
Example script demonstrating the use of last-layers-only mode for TracIn.

This follows the original TracIn paper's approach of only computing gradients
for the last 2-3 layers, which provides:
- MUCH faster computation (10-100x speedup)
- Lower memory usage
- Often comparable influence scores

Usage:
    python example_last_layers.py
"""

from tracin import TracInAnalyzer
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory
import torch

# Example: Create a dummy model to show parameter structure
print("=" * 70)
print("ConvE Model Parameter Analysis")
print("=" * 70)

# Create dummy triples factory
dummy_triples = torch.LongTensor([[0, 0, 1], [1, 1, 2], [2, 2, 3]])
triples_factory = TriplesFactory.from_labeled_triples(
    triples=dummy_triples,
    create_inverse_triples=False
)

# Create ConvE model
model = ConvE(
    triples_factory=triples_factory,
    embedding_dim=200,
    output_channels=32
)

print(f"\nTotal model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"\nAll parameter layers:")
print("-" * 70)

for i, (name, param) in enumerate(model.named_parameters(), 1):
    print(f"{i:2d}. {name:45s} {str(list(param.shape)):25s} ({param.numel():,} params)")

print("\n" + "=" * 70)
print("TracIn Analyzer - Last Layers Detection")
print("=" * 70)

# Create analyzer with last-layers-only mode
analyzer = TracInAnalyzer(
    model=model,
    use_last_layers_only=True,
    device='cpu'
)

if analyzer.tracked_params:
    print(f"\n✓ Auto-detected last layers ({len(analyzer.tracked_params)} parameters):")
    for param_name in sorted(analyzer.tracked_params):
        for name, param in model.named_parameters():
            if name == param_name:
                print(f"  - {name:45s} {str(list(param.shape)):25s} ({param.numel():,} params)")
                break

    # Calculate reduction
    tracked_size = sum(
        param.numel() for name, param in model.named_parameters()
        if name in analyzer.tracked_params
    )
    total_size = sum(p.numel() for p in model.parameters())
    reduction = 100 * (1 - tracked_size / total_size)

    print(f"\n  Tracking {tracked_size:,} / {total_size:,} parameters ({100 - reduction:.1f}%)")
    print(f"  Reduction in gradient computation: {reduction:.1f}%")
else:
    print("\n✗ Could not auto-detect last layers (will use all parameters)")

print("\n" + "=" * 70)
print("Usage Examples")
print("=" * 70)

print("""
1. Using command line with --use-last-layers-only flag:

   python run_tracin.py \\
       --model-path trained_model.pt \\
       --train train.txt \\
       --test test.txt \\
       --entity-to-id entity_to_id.tsv \\
       --relation-to-id relation_to_id.tsv \\
       --output results.json \\
       --use-last-layers-only \\     # <-- Enable last-layers mode
       --mode single

2. Using Python API with auto-detection:

   from tracin import TracInAnalyzer

   analyzer = TracInAnalyzer(
       model=model,
       use_last_layers_only=True,  # Auto-detect last layers
       device='cuda'
   )

3. Using Python API with specific layers:

   analyzer = TracInAnalyzer(
       model=model,
       use_last_layers_only=True,
       last_layer_names=[
           'interaction.linear.weight',
           'interaction.linear.bias'
       ],
       device='cuda'
   )

4. Compare speed (all layers vs. last layers):

   # Slow: Track all parameters
   analyzer_full = TracInAnalyzer(model=model, use_last_layers_only=False)

   # Fast: Track only last layers (10-100x faster!)
   analyzer_fast = TracInAnalyzer(model=model, use_last_layers_only=True)

Benefits of last-layers-only mode:
- 10-100x faster gradient computation
- Significantly lower memory usage
- Follows original TracIn paper methodology
- Often provides comparable influence rankings
""")

print("=" * 70)
