# Self-Influence Integration Summary

## Overview

Successfully integrated self-influence into TracIn CSV and JSON outputs as requested.

## Changes Made

### 1. **tracin.py** - Updated `save_influences_to_csv()` method

**Location**: Lines 519-606

**Changes**:
- Added `self_influence` parameter to method signature (line 528)
- Added "SelfInfluence" column to CSV header (line 559)
- Added self-influence value to each row in CSV output (line 603)
- Uses empty string if self_influence is None for backward compatibility

**Code**:
```python
def save_influences_to_csv(
    self,
    test_triple: Tuple[int, int, int],
    influences: List[Dict],
    output_path: str,
    id_to_entity: Dict[int, str],
    id_to_relation: Dict[int, str],
    entity_labels: Optional[Dict[int, str]] = None,
    relation_labels: Optional[Dict[int, str]] = None,
    self_influence: Optional[float] = None  # ← ADDED
):
    ...
    # Header includes SelfInfluence
    writer.writerow([
        'TestHead', 'TestHead_label',
        'TestRel', 'TestRel_label',
        'TestTail', 'TestTail_label',
        'TrainHead', 'TrainHead_label',
        'TrainRel', 'TrainRel_label',
        'TrainTail', 'TrainTail_label',
        'TracInScore',
        'SelfInfluence'  # ← ADDED
    ])

    # Each row includes self-influence value
    writer.writerow([
        test_h_id, test_h_label,
        test_r_id, test_r_label,
        test_t_id, test_t_label,
        train_h_id, train_h_label,
        train_r_id, train_r_label,
        train_t_id, train_t_label,
        score,
        self_influence if self_influence is not None else ''  # ← ADDED
    ])
```

### 2. **run_tracin.py** - Updated two CSV export calls

**Changes**:

#### Call 1: Line 328-337 (batch mode with separate files)
```python
analyzer.save_influences_to_csv(
    test_triple=test_triple,
    influences=influences,
    output_path=str(csv_file),
    id_to_entity=id_to_entity,
    id_to_relation=id_to_relation,
    entity_labels=idx_to_entity_name,
    relation_labels=relation_labels if relation_labels else None,
    self_influence=self_influence  # ← ADDED
)
```

#### Call 2: Line 446-455 (single mode)
```python
analyzer.save_influences_to_csv(
    test_triple=test_triple,
    influences=influences,
    output_path=str(csv_file),
    id_to_entity=id_to_entity,
    id_to_relation=id_to_relation,
    entity_labels=idx_to_entity_name,
    relation_labels=relation_labels if relation_labels else None,
    self_influence=self_influence  # ← ADDED
)
```

**Note**: Self-influence was already computed in run_tracin.py (lines 384-396) and already included in JSON output (line 412). Now it's also in CSV output.

### 3. **tracin_to_csv.py** - Added self-influence computation and export

**Location**: Lines 227-251

**Changes**:
- Added self-influence computation before CSV export (lines 227-239)
- Passed self_influence to save_influences_to_csv (line 251)
- Added logging of self-influence value (line 239)

**Code**:
```python
# Compute self-influence for the test triple
logger.info(f"\nComputing self-influence...")
test_h, test_r, test_t = test_triple
grad = analyzer.compute_gradient(test_h, test_r, test_t, label=1.0)

# Compute squared L2 norm of gradient (self-influence)
self_influence = 0.0
for name in grad:
    grad_flat = grad[name].flatten()
    self_influence += torch.dot(grad_flat, grad_flat).item()
self_influence *= learning_rate

logger.info(f"  Self-influence: {self_influence:.6f}")

# Save to CSV
analyzer.save_influences_to_csv(
    test_triple=test_triple,
    influences=influences,
    output_path=output_csv,
    id_to_entity=id_to_entity,
    id_to_relation=id_to_relation,
    entity_labels=entity_labels,
    relation_labels=relation_labels,
    self_influence=self_influence  # ← ADDED
)
```

## Output Format

### CSV Format (Updated)

The CSV now includes a "SelfInfluence" column at the end:

```csv
TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore,SelfInfluence
CHEBI:34911,Permethrin,biolink:affects,biolink:affects,MONDO:0004525,nephrolithiasis,CHEBI:34911,Permethrin,biolink:affects,biolink:affects,MONDO:0004525,nephrolithiasis,0.0234567,0.0123456
...
```

**Key Points**:
- Self-influence value is the same for all rows (since it's per test triple, not per training triple)
- Self-influence represents the squared L2 norm of the test triple's gradient multiplied by learning rate
- Value measures how much the test triple would influence itself if it were in the training set

### JSON Format (Already Existed)

JSON output already included self-influence:

```json
{
  "test_triple": [123, 45, 678],
  "test_head": 123,
  "test_relation": 45,
  "test_tail": 678,
  "self_influence": 0.0123456,
  "influences": [...]
}
```

## What Self-Influence Means

**Self-Influence**: Measures how much a training example would influence its own prediction if it were in the training set.

**Computation**:
```
self_influence = learning_rate × ||∇L(test_triple)||²
```

Where:
- `∇L(test_triple)` is the gradient of the loss with respect to model parameters for the test triple
- `||·||²` is the squared L2 norm
- Result is scaled by the learning rate used during training

**Interpretation**:
- Higher self-influence → test triple has larger gradient → model is more uncertain about this prediction
- Lower self-influence → test triple has smaller gradient → model is more confident
- Can be used to identify difficult or unusual test examples

## Logging

Self-influence is now logged in all scripts:

### run_tracin.py
```
Self-influence: 0.012345
```

### tracin_to_csv.py
```
Computing self-influence...
  Self-influence: 0.012345
```

## Backward Compatibility

The implementation maintains backward compatibility:
- If `self_influence=None` is passed, CSV column will be empty (blank string)
- Existing code that doesn't pass self_influence will continue to work
- Optional parameter with default value of `None`

## Files Modified

1. **git/conve_pykeen/tracin.py** - Lines 519-606
2. **git/conve_pykeen/run_tracin.py** - Lines 328-337, 446-455
3. **git/conve_pykeen/tracin_to_csv.py** - Lines 227-251

## Testing

To verify the changes work correctly:

```bash
# Single test triple with CSV output
python git/conve_pykeen/run_tracin.py \
    --model-path examples/trained_model.pt \
    --train examples/train.txt \
    --test examples/test.txt \
    --entity-to-id examples/entity_to_id.tsv \
    --relation-to-id examples/relation_to_id.tsv \
    --output results/tracin_output.json \
    --csv-output results/tracin_output.csv \
    --mode single \
    --test-indices 0

# Check that CSV has SelfInfluence column
head -1 results/tracin_output.csv
# Expected: TestHead,TestHead_label,TestRel,TestRel_label,TestTail,TestTail_label,TrainHead,TrainHead_label,TrainRel,TrainRel_label,TrainTail,TrainTail_label,TracInScore,SelfInfluence

# Check that all rows have self-influence value
head -3 results/tracin_output.csv
```

## Summary

✅ Self-influence is now included in:
- Logger output (already existed, verified still working)
- JSON output (already existed, verified still working)
- CSV output (newly added)

✅ All changes maintain backward compatibility
✅ Self-influence is computed and logged in all TracIn scripts
✅ CSV header updated to include "SelfInfluence" column
✅ All CSV rows include the self-influence value for the test triple
