# Variable Number of Last Layers - Quick Guide

## Yes! You can now track the last N layers (not just 1)

The implementation supports flexible control over **how many** last layers to track.

## Quick Examples

### 1. Track Last 1 Layer (Fastest)
```bash
python run_tracin.py \
    --model-path model.pt \
    --use-last-layers-only \
    --num-last-layers 1 \
    ...
```

### 2. Track Last 2 Layers (Recommended - Default)
```bash
python run_tracin.py \
    --model-path model.pt \
    --use-last-layers-only \
    --num-last-layers 2 \    # This is the default
    ...
```

### 3. Track Last 3 Layers (Good Balance)
```bash
python run_tracin.py \
    --model-path model.pt \
    --use-last-layers-only \
    --num-last-layers 3 \
    ...
```

### 4. Track Last 5+ Layers (More Complete)
```bash
python run_tracin.py \
    --model-path model.pt \
    --use-last-layers-only \
    --num-last-layers 5 \
    ...
```

### 5. Track All Layers (Slowest, Most Accurate)
```bash
python run_tracin.py \
    --model-path model.pt \
    # Don't use --use-last-layers-only flag
    ...
```

## Python API

```python
from tracin import TracInAnalyzer

# Fast: Last 1 layer
analyzer = TracInAnalyzer(
    model=model,
    use_last_layers_only=True,
    num_last_layers=1,
    device='cuda'
)

# Recommended: Last 2 layers
analyzer = TracInAnalyzer(
    model=model,
    use_last_layers_only=True,
    num_last_layers=2,  # Default
    device='cuda'
)

# More complete: Last 5 layers
analyzer = TracInAnalyzer(
    model=model,
    use_last_layers_only=True,
    num_last_layers=5,
    device='cuda'
)

# Full: All layers
analyzer = TracInAnalyzer(
    model=model,
    use_last_layers_only=False,
    device='cuda'
)
```

## How It Works

For a typical ConvE model, layers are tracked from last to first:

| num_last_layers | Layers Tracked | Speed | Accuracy |
|-----------------|----------------|-------|----------|
| 1 | Final bias OR weight | Fastest | Lower |
| 2 | Final weight + bias | Fast | Good |
| 3 | Final layer + batch norm | Medium | Better |
| 5 | Multiple conv/norm layers | Slower | High |
| All | Everything (20M+ params) | Slowest | Best |

## Auto-Detection Strategy

When you specify `num_last_layers=N`, the system:

1. **Finds final linear layer** (e.g., `interaction.linear.weight`, `interaction.linear.bias`)
2. **Adds preceding layers** if N > 2 (e.g., batch norm, conv layers)
3. **Skips embedding layers** (these are usually huge and less informative)

### Example for ConvE:

```
num_last_layers=1: ['interaction.linear.bias']
num_last_layers=2: ['interaction.linear.weight', 'interaction.linear.bias']
num_last_layers=3: ['interaction.bn2.weight', 'interaction.bn2.bias',
                    'interaction.linear.weight', 'interaction.linear.bias']
```

## Speed/Accuracy Tradeoff

| Configuration | Use Case |
|--------------|----------|
| **1 layer** | Quick exploration, very large datasets, when speed is critical |
| **2-3 layers** | Most use cases, good balance of speed and accuracy |
| **5+ layers** | When you need more information, medium datasets |
| **All layers** | Final publication-quality analysis, small datasets, benchmarking |

## Speedup Estimates

For a ConvE model with 100K entities:

- **1 layer**: ~100x faster than all layers
- **2 layers**: ~50-100x faster than all layers
- **3 layers**: ~30-50x faster than all layers
- **5 layers**: ~20-30x faster than all layers
- **All layers**: 1x (baseline)

*Actual speedup depends on model size and which layers are tracked.*

## Recommendation

**Start with `num_last_layers=2`** (the default):
- Good speed/accuracy balance
- Follows original TracIn paper methodology
- Tracks both weight and bias of final layer
- ~50-100x faster than all layers

If you need more speed: Try `num_last_layers=1`
If you need more accuracy: Try `num_last_layers=3` or `5`

## To Answer Your Original Question

> "Is it possible to get last n layers instead of just last one?"

**Absolutely YES!**

The implementation:
- ✅ Already supported multiple layers by default (2 layers)
- ✅ Now has explicit `num_last_layers` parameter for full control
- ✅ Can track anywhere from 1 to N layers
- ✅ Auto-detects the appropriate layers based on your specification
- ✅ Works via both Python API and command line

The original TracIn paper used the **last 2 layers** of ResNet, and our default (`num_last_layers=2`) matches this! 🎯
