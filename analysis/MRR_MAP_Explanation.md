# Mean Reciprocal Rank (MRR) and Mean Average Precision (MAP) Explained

## Overview

This document explains how **Mean Reciprocal Rank (MRR)** and **Mean Average Precision (MAP)** are calculated for evaluating TracIn scores in identifying ground truth edges from DrugMechDB.

---

## Context: TracIn Evaluation Task

We are evaluating how well TracIn scores can identify **ground truth training edges** (edges that appear in mechanistic paths from DrugMechDB) for each test triple.

**Setup:**
- For each test triple (e.g., Drug A treats Disease B), we have multiple training triples
- Each training triple has a **TracInScore** indicating its influence on the test prediction
- Some training triples are **ground truth** (In_path = 1), meaning they appear in known mechanistic pathways
- We rank all training triples by their TracInScore (descending: higher score = more influential)
- **Goal:** Ground truth edges should be ranked higher if TracIn is effective

---

## Mean Reciprocal Rank (MRR)

### What Does It Measure?

MRR measures how quickly you can find **at least one relevant item** in a ranked list. It focuses only on the **position of the first ground truth edge**.

### Formula

For a single test triple:

```
RR = 1 / rank_of_first_ground_truth_edge
```

For multiple test triples:

```
MRR = (1/N) × Σ(1 / rank_i)
```

where:
- `N` = number of test triples
- `rank_i` = position of the first ground truth edge for test triple i (1-indexed)

### Example Calculation

**Test Triple 1:**
```
Ranking (by TracInScore):
1. Training edge A (In_path = 0)
2. Training edge B (In_path = 0)
3. Training edge C (In_path = 1) ← First ground truth at rank 3
4. Training edge D (In_path = 1)
5. Training edge E (In_path = 0)
```
**RR₁ = 1/3 = 0.3333**

**Test Triple 2:**
```
Ranking:
1. Training edge F (In_path = 1) ← First ground truth at rank 1
2. Training edge G (In_path = 0)
3. Training edge H (In_path = 1)
```
**RR₂ = 1/1 = 1.0000**

**Test Triple 3:**
```
Ranking:
1. Training edge I (In_path = 0)
2. Training edge J (In_path = 0)
3. Training edge K (In_path = 0)
4. Training edge L (In_path = 0)
5. Training edge M (In_path = 1) ← First ground truth at rank 5
```
**RR₃ = 1/5 = 0.2000**

**MRR = (0.3333 + 1.0000 + 0.2000) / 3 = 0.5111**

### Interpretation

- **MRR = 1.0**: Perfect! First ground truth is always at rank 1
- **MRR = 0.5**: On average, the first ground truth appears at rank 2
- **MRR = 0.1**: On average, the first ground truth appears at rank 10
- **Higher is better**: MRR closer to 1.0 means TracIn is better at ranking ground truth edges highly

### When to Use MRR

- When you care about finding **at least one relevant item** quickly
- When the user typically looks at only the top-ranked results
- When all ground truth items are equally valuable
- Commonly used in information retrieval and search engines

---

## Mean Average Precision (MAP)

### What Does It Measure?

MAP measures the quality of the entire ranking by considering **all ground truth edges** and their positions. It rewards systems that place **multiple ground truth edges** higher in the ranking.

### Formula

For a single test triple:

```
AP = (1 / |relevant|) × Σ(Precision@k × relevance@k)
```

For multiple test triples:

```
MAP = (1/N) × Σ(AP_i)
```

where:
- `|relevant|` = total number of ground truth edges for this test triple
- `Precision@k` = (number of ground truth edges in top k) / k
- `relevance@k` = 1 if item at rank k is ground truth, 0 otherwise
- `N` = number of test triples

### Detailed Example Calculation

**Test Triple 1:**
```
Total ground truth edges: 3

Ranking (by TracInScore):
Rank 1: Training edge A (In_path = 0) → not relevant
Rank 2: Training edge B (In_path = 1) → relevant! ✓
Rank 3: Training edge C (In_path = 0) → not relevant
Rank 4: Training edge D (In_path = 1) → relevant! ✓
Rank 5: Training edge E (In_path = 0) → not relevant
Rank 6: Training edge F (In_path = 1) → relevant! ✓
Rank 7: Training edge G (In_path = 0) → not relevant
```

**Step-by-step calculation:**

1. **At rank 2** (first ground truth):
   - Precision@2 = 1/2 = 0.5000
   - This is a ground truth edge, so count it

2. **At rank 4** (second ground truth):
   - Precision@4 = 2/4 = 0.5000
   - This is a ground truth edge, so count it

3. **At rank 6** (third ground truth):
   - Precision@6 = 3/6 = 0.5000
   - This is a ground truth edge, so count it

**AP₁ = (1/3) × (0.5000 + 0.5000 + 0.5000) = 0.5000**

**Test Triple 2:**
```
Total ground truth edges: 2

Ranking:
Rank 1: Training edge H (In_path = 1) → relevant! ✓
Rank 2: Training edge I (In_path = 0) → not relevant
Rank 3: Training edge J (In_path = 1) → relevant! ✓
Rank 4: Training edge K (In_path = 0) → not relevant
```

**Step-by-step calculation:**

1. **At rank 1** (first ground truth):
   - Precision@1 = 1/1 = 1.0000

2. **At rank 3** (second ground truth):
   - Precision@3 = 2/3 = 0.6667

**AP₂ = (1/2) × (1.0000 + 0.6667) = 0.8334**

**Test Triple 3:**
```
Total ground truth edges: 1

Ranking:
Rank 1: Training edge L (In_path = 0) → not relevant
Rank 2: Training edge M (In_path = 0) → not relevant
Rank 3: Training edge N (In_path = 1) → relevant! ✓
```

**Step-by-step calculation:**

1. **At rank 3** (only ground truth):
   - Precision@3 = 1/3 = 0.3333

**AP₃ = (1/1) × 0.3333 = 0.3333**

**MAP = (0.5000 + 0.8334 + 0.3333) / 3 = 0.5556**

### Interpretation

- **MAP = 1.0**: Perfect! All ground truth edges are ranked at the very top
- **MAP = 0.8**: Excellent - most ground truth edges are ranked highly
- **MAP = 0.5**: Moderate - ground truth edges are scattered in the ranking
- **MAP = 0.1**: Poor - ground truth edges are ranked very low
- **Higher is better**: MAP closer to 1.0 means better overall ranking quality

### When to Use MAP

- When you have **multiple relevant items** per query
- When you want to evaluate the **entire ranking**, not just the first result
- When finding more relevant items is important
- When the order of all relevant items matters
- Commonly used in information retrieval, recommendation systems

---

## MRR vs MAP: Key Differences

| Aspect | MRR | MAP |
|--------|-----|-----|
| **Focus** | First relevant item only | All relevant items |
| **Sensitivity** | Only cares about position of first GT edge | Considers positions of all GT edges |
| **Use Case** | "Find me one good answer" | "Show me all relevant results, ranked well" |
| **Range** | 0 to 1 | 0 to 1 |
| **Best Score** | 1.0 (first item is GT) | 1.0 (all GT items at top) |
| **Multiple GTs** | Ignores all but the first | Rewards ranking all of them highly |

### Example Showing the Difference

**Scenario A:**
```
Ranking: [GT, -, -, GT, GT, -, -, -, -, -]
         (GT at ranks: 1, 4, 5)

MRR = 1.0 (first GT at rank 1)
MAP = (1/3) × (1/1 + 2/4 + 3/5) = 0.7333
```

**Scenario B:**
```
Ranking: [GT, GT, GT, -, -, -, -, -, -, -]
         (GT at ranks: 1, 2, 3)

MRR = 1.0 (first GT at rank 1)
MAP = (1/3) × (1/1 + 2/2 + 3/3) = 1.0000
```

**Analysis:**
- Both have the same MRR (1.0) because the first GT is at rank 1
- Scenario B has higher MAP (1.0 vs 0.7333) because all GTs are grouped at the top
- **MAP distinguishes** between these two scenarios; **MRR does not**

---

## Implementation in Our Code

### MRR Calculation

```python
def calculate_mrr(rankings: List[int]) -> float:
    """
    Calculate Mean Reciprocal Rank.

    Parameters:
        rankings: List of ranks (1-indexed) of first relevant item for each query

    Returns:
        Mean Reciprocal Rank
    """
    if not rankings:
        return 0.0

    reciprocal_ranks = [1.0 / rank for rank in rankings if rank > 0]
    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
```

### MAP Calculation

```python
def calculate_average_precision(relevant_positions: List[int],
                                 total_items: int) -> float:
    """
    Calculate Average Precision for a single query.

    Parameters:
        relevant_positions: Sorted list of positions (1-indexed) where GTs appear
        total_items: Total number of items in ranking

    Returns:
        Average Precision
    """
    if not relevant_positions:
        return 0.0

    # Calculate precision at each relevant position
    precisions = []
    for i, pos in enumerate(relevant_positions):
        # Precision@k = (number of relevant items in top k) / k
        num_relevant_at_k = i + 1
        precision_at_k = num_relevant_at_k / pos
        precisions.append(precision_at_k)

    # Average precision is mean of precisions at relevant positions
    return np.mean(precisions)


def calculate_map(all_relevant_positions: List[List[int]],
                   all_total_items: List[int]) -> float:
    """
    Calculate Mean Average Precision.

    Parameters:
        all_relevant_positions: List of relevant positions for each query
        all_total_items: List of total items for each query

    Returns:
        Mean Average Precision
    """
    if not all_relevant_positions:
        return 0.0

    aps = []
    for relevant_pos, total in zip(all_relevant_positions, all_total_items):
        ap = calculate_average_precision(relevant_pos, total)
        aps.append(ap)

    return np.mean(aps)
```

---

## Real-World Application to TracIn Analysis

### Our Evaluation Setup

1. **For each test triple** (e.g., "Promazine treats acute intermittent porphyria"):
   - We have ~1000-100,000 training edges
   - Each has a TracInScore (influence on prediction)
   - Some are ground truth (In_path = 1) from DrugMechDB mechanistic pathways

2. **We rank** all training edges by TracInScore (descending)

3. **We calculate**:
   - **MRR**: How quickly do we find the first mechanistic pathway edge?
   - **MAP**: How well are all mechanistic pathway edges ranked overall?

### Example from Real Data

**CCGGDD_alltreat Results:**
- MRR: 0.0123 → First GT edge found at average rank ~81
- MAP: 0.0048 → GT edges are spread across the ranking
- Avg GT edges per triple: 17.37

**CGGD_alltreat Results:**
- MRR: 0.0021 → First GT edge found at average rank ~476
- MAP: 0.0020 → GT edges are more scattered
- Avg GT edges per triple: 1.69

**Interpretation:**
- CCGGDD performs **6x better** on MRR (finds first GT much faster)
- CCGGDD performs **2.4x better** on MAP (ranks all GTs better overall)
- CCGGDD has **10x more** GT edges available, making it easier to find them

---

## Statistical Significance Testing

We use the **Mann-Whitney U test** to determine if differences in MRR and MAP between datasets are statistically significant:

```python
from scipy import stats

# Compare MRR between datasets
stat, pval = stats.mannwhitneyu(ccggdd_mrr, cggd_mrr, alternative='two-sided')

if pval < 0.05:
    print("Difference is statistically significant (p < 0.05)")
else:
    print("Difference is NOT statistically significant (p >= 0.05)")
```

**Our results:**
- MRR difference: p = 0.000017 (highly significant)
- MAP difference: p = 0.000024 (highly significant)

This means the performance difference between datasets is **not due to chance**.

---

## Limitations and Considerations

### MRR Limitations

1. **Ignores multiple relevant items**: If you have 10 GT edges but only rank the first one highly, MRR will still be high
2. **Binary view**: Only cares about "found first GT" vs "didn't find first GT"
3. **Not sensitive to tail performance**: Ranking GT #2-10 poorly doesn't affect MRR

### MAP Limitations

1. **Assumes all GTs are equally important**: Treats all mechanistic pathway edges the same
2. **Sensitive to number of GTs**: Easier to get high MAP when you have fewer GT edges
3. **Doesn't consider ranking of non-relevant items**: Only GT positions matter

### When to Use Both

Use **both metrics together** to get a complete picture:
- **MRR** tells you: "Can I quickly find at least one mechanistic pathway?"
- **MAP** tells you: "Are most/all mechanistic pathways ranked highly?"

---

## References

1. **MRR**: Voorhees, E. M. (1999). "The TREC-8 Question Answering Track Report." TREC.
2. **MAP**: Manning, C. D., Raghavan, P., & Schütze, H. (2008). "Introduction to Information Retrieval." Cambridge University Press.
3. **TracIn**: Pruthi, G., Liu, F., Kale, S., & Sundararajan, M. (2020). "Estimating Training Data Influence by Tracing Gradient Descent." NeurIPS.

---

## Summary

| Metric | What it measures | Formula | Best for |
|--------|-----------------|---------|----------|
| **MRR** | Position of first GT | `1/N × Σ(1/rank_i)` | Finding one relevant item fast |
| **MAP** | Quality of entire ranking | `1/N × Σ(AP_i)` | Finding all relevant items |

Both metrics range from 0 to 1, where **higher is better**. They complement each other to provide a comprehensive evaluation of TracIn's ability to identify mechanistic pathway edges.
