# Computing All TracIn Influences - Update

## Change Summary

Updated `batch_tracin_with_filtering.py` to **return ALL TracIn scores by default** instead of limiting to top-k.

---

## What Changed

### 1. Default Behavior

**Before:**
- `--top-k` default: 100
- Only returned top 100 most influential training triples

**After:**
- `--top-k` default: None
- Returns **ALL** training triple influences (no limit)

### 2. Command-Line Argument

```python
# Old
parser.add_argument('--top-k', type=int, default=100,
                    help='Number of top influences (default: 100)')

# New
parser.add_argument('--top-k', type=int, default=None,
                    help='Number of top influences (default: None = all influences)')
```

### 3. Function Signature

```python
# Old
def run_tracin_analysis(..., top_k: int = 100, ...):

# New
def run_tracin_analysis(..., top_k: int = None, ...):
```

---

## Why This Change

### Benefits of Computing All Influences

1. **Complete Data**: No arbitrary cutoff that might exclude important influences
2. **Post-hoc Analysis**: Can apply any threshold or filtering after computation
3. **No Data Loss**: Ensures all influence patterns are captured
4. **Comprehensive**: Better for exploratory analysis and pattern discovery

### Use Cases

- **Exploratory analysis**: Don't know which threshold to use yet
- **Comparative studies**: Need full distribution of influences
- **Tail analysis**: Want to examine low-influence triples too
- **Statistical analysis**: Need complete data for proper statistics

---

## Impact on Output

### CSV File Sizes

**Before (top-k=100):**
- Each CSV file: ~100 rows
- Example: `triple_001_CHEBI_34911_MONDO_0004525_tracin.csv` = ~10 KB

**After (all influences):**
- Each CSV file: ~10,000-30,000 rows (depends on filtering)
- Example: `triple_001_CHEBI_34911_MONDO_0004525_tracin.csv` = ~1-3 MB

### Total Output Size

**For 25 test triples:**

| Configuration | Rows per CSV | Total Rows | Total Size |
|---------------|--------------|------------|------------|
| top-k=100 | ~100 | ~2,500 | ~250 KB |
| **top-k=None (all)** | **~20,000** | **~500,000** | **~50 MB** |

**Note:** Actual size depends on filtering parameters (n-hops, min-degree)

---

## Performance Impact

### Computation Time

**No significant impact** on computation time:
- TracIn still computes influence for ALL training triples
- top-k only affected which results were saved
- Now we save all results instead of filtering

**Expected time remains the same:** ~20-30 seconds per triple

### Storage Impact

**Moderate increase** in storage:
- CSV files are ~100x larger
- Total storage for 25 triples: ~50 MB (instead of ~250 KB)
- Still manageable for modern systems

---

## How to Use

### Default: Get All Influences

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    --model-path model.pt \
    --train train.txt \
    --entity-to-id entity_to_id.tsv \
    --relation-to-id relation_to_id.tsv \
    --output-dir results/batch_tracin \
    --device cuda
# No --top-k argument = returns ALL influences
```

### Optional: Limit to Top-K

If you want only top influences:

```bash
python batch_tracin_with_filtering.py \
    --test-triples examples/20251017_top_test_triples.txt \
    ... (other args) \
    --top-k 100  # Only return top 100 influences
```

---

## Post-Processing Examples

### Filter Top-K After Computation

```python
import pandas as pd

# Load CSV with all influences
df = pd.read_csv('triple_001_CHEBI_34911_MONDO_0004525_tracin.csv')

# Get top 100
top_100 = df.nlargest(100, 'TracInScore')

# Get top 10%
top_10_percent = df.nlargest(int(len(df) * 0.1), 'TracInScore')

# Filter by threshold
high_influence = df[df['TracInScore'] > 0.5]
```

### Analyze Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('triple_001_CHEBI_34911_MONDO_0004525_tracin.csv')

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(df['TracInScore'], bins=50, edgecolor='black')
plt.xlabel('TracIn Score')
plt.ylabel('Frequency')
plt.title('Distribution of TracIn Influences')
plt.savefig('influence_distribution.png')

# Summary statistics
print(df['TracInScore'].describe())
```

### Identify Tail Influences

```python
import pandas as pd

df = pd.read_csv('triple_001_CHEBI_34911_MONDO_0004525_tracin.csv')

# Get bottom 100 (least influential)
bottom_100 = df.nsmallest(100, 'TracInScore')

# Get negative influences
negative = df[df['TracInScore'] < 0]

print(f"Negative influences: {len(negative)}")
print(negative[['TrainHead_label', 'TrainRel_label', 'TrainTail_label', 'TracInScore']].head(10))
```

---

## Recommendations

### When to Use top-k=None (Default)

✅ **Use for:**
- Initial exploratory analysis
- When you need complete data
- When you want to analyze influence distributions
- When storage is not a concern (~50 MB for 25 triples)

### When to Use top-k Limit

✅ **Use when:**
- You only care about highest influences
- Storage is limited
- You want smaller, more manageable files
- You already know the threshold

**Example:**
```bash
--top-k 50   # Minimal: only top 50
--top-k 100  # Recommended: good balance
--top-k 500  # Comprehensive: captures more patterns
```

---

## Example Output Comparison

### With top-k=100

```csv
TestHead,TestHead_label,...,TracInScore
CHEBI:34911,Permethrin,...,0.9872
CHEBI:34911,Permethrin,...,0.9654
...
(100 rows total)
```

### With top-k=None (all)

```csv
TestHead,TestHead_label,...,TracInScore
CHEBI:34911,Permethrin,...,0.9872
CHEBI:34911,Permethrin,...,0.9654
...
CHEBI:34911,Permethrin,...,0.0023
CHEBI:34911,Permethrin,...,0.0001
CHEBI:34911,Permethrin,...,-0.0045
(20,000+ rows total - includes ALL filtered training triples)
```

---

## Migration Guide

### If You Were Using Default Before

**No changes needed!** Just be aware:
- CSV files will be larger (~1-3 MB each)
- You'll get all influences instead of top 100
- Can filter in post-processing if needed

### If You Want Old Behavior

Add `--top-k 100`:
```bash
python batch_tracin_with_filtering.py \
    ... (other args) \
    --top-k 100  # Restore old behavior
```

---

## Summary

✅ **Default changed**: Now returns ALL influences (not top-k=100)
✅ **Better for exploratory analysis**: Complete data for post-hoc filtering
✅ **Larger output files**: ~50 MB for 25 triples (vs ~250 KB)
✅ **Same computation time**: No performance penalty
✅ **Flexible**: Can still limit with `--top-k N` if desired
✅ **Post-processable**: Apply any threshold or ranking after computation

**This change provides more complete data by default, giving you full flexibility to analyze influences however you need!**
