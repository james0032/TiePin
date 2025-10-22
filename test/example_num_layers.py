"""
Example script demonstrating variable number of last layers for TracIn.

This shows how to control the speed/accuracy tradeoff by adjusting
how many layers are tracked.

Usage:
    python example_num_layers.py
"""

from tracin import TracInAnalyzer
from pykeen.models import ConvE
from pykeen.triples import TriplesFactory
import torch

# Create dummy model
dummy_triples = torch.LongTensor([[0, 0, 1], [1, 1, 2], [2, 2, 3]])
triples_factory = TriplesFactory.from_labeled_triples(
    triples=dummy_triples,
    create_inverse_triples=False
)

model = ConvE(
    triples_factory=triples_factory,
    embedding_dim=200,
    output_channels=32
)

print("=" * 80)
print("TracIn: Variable Number of Last Layers Demo")
print("=" * 80)

# Show all model parameters
print("\nAll model parameters:")
print("-" * 80)
all_params = list(model.named_parameters())
for i, (name, param) in enumerate(all_params, 1):
    print(f"{i:2d}. {name:50s} {str(list(param.shape)):20s} ({param.numel():,} params)")

total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

print("\n" + "=" * 80)
print("Testing Different Layer Configurations")
print("=" * 80)

# Test different configurations
configs = [
    (1, "Fastest - Only last layer (bias or weight)"),
    (2, "Recommended - Last 2 layers (weight + bias)"),
    (3, "Balanced - Last 3 layers (includes batch norm)"),
    (5, "More complete - Last 5 layers"),
]

for num_layers, description in configs:
    print(f"\n{'─' * 80}")
    print(f"Configuration: {num_layers} layer(s) - {description}")
    print('─' * 80)

    analyzer = TracInAnalyzer(
        model=model,
        use_last_layers_only=True,
        num_last_layers=num_layers,
        device='cpu'
    )

    if analyzer.tracked_params:
        # Calculate statistics
        tracked_size = sum(
            param.numel() for name, param in model.named_parameters()
            if name in analyzer.tracked_params
        )
        reduction = 100 * (1 - tracked_size / total_params)
        speedup = 1 / (1 - reduction/100)  # Approximate speedup

        print(f"\n✓ Tracking {len(analyzer.tracked_params)} parameter group(s):")
        for param_name in sorted(analyzer.tracked_params):
            for name, param in model.named_parameters():
                if name == param_name:
                    print(f"  • {name:50s} ({param.numel():,} params)")
                    break

        print(f"\n  Summary:")
        print(f"    Parameters tracked: {tracked_size:,} / {total_params:,} ({100 - reduction:.1f}%)")
        print(f"    Parameter reduction: {reduction:.1f}%")
        print(f"    Estimated speedup: ~{speedup:.1f}x")
    else:
        print("\n✗ Could not auto-detect layers")

print("\n" + "=" * 80)
print("Usage Examples")
print("=" * 80)

print("""
1. Command line - Use last 1 layer (fastest):

   python run_tracin.py \\
       --model-path model.pt \\
       --train train.txt \\
       --test test.txt \\
       --entity-to-id entity_to_id.tsv \\
       --relation-to-id relation_to_id.tsv \\
       --output results.json \\
       --use-last-layers-only \\
       --num-last-layers 1 \\
       --mode single

2. Command line - Use last 2 layers (recommended):

   python run_tracin.py \\
       --model-path model.pt \\
       ... \\
       --use-last-layers-only \\
       --num-last-layers 2 \\    # This is the default
       --mode single

3. Command line - Use last 5 layers (more complete):

   python run_tracin.py \\
       --model-path model.pt \\
       ... \\
       --use-last-layers-only \\
       --num-last-layers 5 \\
       --mode single

4. Python API - Different configurations:

   # Fastest: Only 1 layer
   analyzer_fast = TracInAnalyzer(
       model=model,
       use_last_layers_only=True,
       num_last_layers=1,
       device='cuda'
   )

   # Recommended: 2-3 layers
   analyzer_balanced = TracInAnalyzer(
       model=model,
       use_last_layers_only=True,
       num_last_layers=2,  # or 3
       device='cuda'
   )

   # More complete: 5+ layers
   analyzer_complete = TracInAnalyzer(
       model=model,
       use_last_layers_only=True,
       num_last_layers=5,
       device='cuda'
   )

   # Full: All layers (slowest)
   analyzer_full = TracInAnalyzer(
       model=model,
       use_last_layers_only=False,  # Tracks everything
       device='cuda'
   )

5. Manually specify exact layers:

   analyzer = TracInAnalyzer(
       model=model,
       use_last_layers_only=True,
       last_layer_names=[
           'interaction.bn2.weight',
           'interaction.bn2.bias',
           'interaction.linear.weight',
           'interaction.linear.bias'
       ],
       device='cuda'
   )

Recommendation:
- Start with num_last_layers=2 (default) for good speed/accuracy balance
- Use num_last_layers=1 if you need maximum speed
- Use num_last_layers=3-5 if you want more complete gradients
- Use use_last_layers_only=False only if you need full gradient tracking

Speed/Accuracy Tradeoff:
  Layers  |  Speed    |  Accuracy  |  Use Case
  --------|-----------|------------|------------------------------------
     1    |  Fastest  |  Lower     |  Quick exploration, large datasets
    2-3   |  Fast     |  Good      |  Recommended for most use cases
    5+    |  Medium   |  Better    |  When you need more complete info
    All   |  Slowest  |  Best      |  Final analysis, small datasets
""")

print("=" * 80)
