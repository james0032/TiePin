# Fix: Proper CSV Formatting for Mechanistic Paths Files

## Problem

The output files from `find_mechanistic_paths.py` were not properly formatted as CSV files:

```csv
Drug,Disease,Intermediate_Nodes,drugmechdb_path_id
CHEBI:10023,HP:0020103,[GO:0006696, GO:0030445, HGNC.FAMILY:862],DB00582_MESH_D055744_1
```

**Issues:**
- ❌ List values (Intermediate_Nodes column) were not quoted
- ❌ Commas inside lists broke CSV parsing
- ❌ `add_ground_truth_column.py` couldn't parse the files correctly
- ❌ Missing closing brackets caused additional parsing errors

## Root Cause

The `find_mechanistic_paths.py` script was writing CSV files **manually using string formatting** instead of using Python's `csv` module:

```python
# OLD (WRONG):
f.write("Drug,Disease,Intermediate_Nodes,drugmechdb_path_id\n")
f.write(f"{drug},{disease},{intermediates_str},{path_id}\n")
```

This doesn't properly quote fields that contain commas or special characters.

## Solution

### Changes Made

Updated `src/find_mechanistic_paths.py` to use Python's `csv.writer` with proper quoting:

#### 1. Added CSV import (Line 75)
```python
import csv
```

#### 2. Fixed `save_path_id_results()` (Lines 566-597)
**Before:**
```python
with open(output_path, 'w') as f:
    f.write("Drug,Disease,Intermediate_Nodes,drugmechdb_path_id\n")
    for r in results:
        intermediates_str = "[" + ", ".join(intermediates) + "]" if intermediates else "[]"
        f.write(f"{drug},{disease},{intermediates_str},{path_id}\n")
```

**After:**
```python
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)

    # Write header
    writer.writerow(['Drug', 'Disease', 'Intermediate_Nodes', 'drugmechdb_path_id'])

    for r in results:
        intermediates_str = "[" + ", ".join(intermediates) + "]" if intermediates else "[]"
        writer.writerow([drug, disease, intermediates_str, path_id])
```

#### 3. Fixed `save_results_txt()` (Lines 491-519)
Same approach - replaced manual string formatting with `csv.writer`.

#### 4. Fixed `save_results_txt_deduplicated()` (Lines 522-573)
Same approach - replaced manual string formatting with `csv.writer`.

### How CSV Quoting Works

With `csv.QUOTE_MINIMAL`, the CSV writer automatically quotes fields that contain:
- Commas (`,`)
- Quotes (`"`)
- Newlines (`\n`)

**Before (unquoted):**
```csv
CHEBI:10023,HP:0020103,[GO:0006696, GO:0030445, HGNC.FAMILY:862],DB00582_MESH_D055744_1
```
❌ The commas inside the list break parsing

**After (properly quoted):**
```csv
CHEBI:10023,HP:0020103,"[GO:0006696, GO:0030445, HGNC.FAMILY:862]",DB00582_MESH_D055744_1
```
✅ The list is quoted, so commas inside are preserved

## Impact

### Files Fixed:
1. ✅ `drugmechdb_path_id_results.txt` - Main output file
2. ✅ `treats_mechanistic_paths.txt` - Old method output
3. ✅ `dedup_treats_mechanistic_paths.txt` - Deduplicated output

### Downstream Benefits:
- ✅ `add_ground_truth_column.py` can now correctly parse Intermediate_Nodes
- ✅ No more warnings about "Failed to parse intermediate nodes"
- ✅ No more issues with missing closing brackets (data quality issues are now clear)
- ✅ `utils/add_pair_exists_column.py` already uses `csv.reader`, so it's compatible

## Testing

### Before Fix:
```python
import csv
with open('drugmechdb_path_id_results.txt', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['Intermediate_Nodes'])
# Output: [GO:0006696    ← Truncated at first comma!
```

### After Fix:
```python
import csv
with open('drugmechdb_path_id_results.txt', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(row['Intermediate_Nodes'])
# Output: [GO:0006696, GO:0030445, HGNC.FAMILY:862]    ← Complete!
```

## Usage

### To Re-generate Files with Proper CSV Format:

```bash
# Run the mechanistic paths extraction
python src/find_mechanistic_paths.py \
    --edges_file /path/to/edges.jsonl.gz \
    --output_dir results/mechanistic_paths/

# Output files will now be properly formatted CSV
```

### Files Generated:
1. **drugmechdb_path_id_results.txt** - ✅ Properly formatted CSV
2. **treats.txt** - TSV format (unchanged, already correct)
3. **treats_mechanistic_paths.json** - JSON format (unchanged)
4. *(Optional with `--run_old_method`)*:
   - treats_mechanistic_paths.txt - ✅ Properly formatted CSV
   - dedup_treats_mechanistic_paths.txt - ✅ Properly formatted CSV

### Then Process with add_ground_truth_column.py:

```bash
# Now this will work correctly!
python add_ground_truth_column.py \
    --tracin-csv results/triple_000_tracin.csv \
    --ground-truth ground_truth/drugmechdb_edges.jsonl \
    --mechanistic-paths results/mechanistic_paths/drugmechdb_path_id_results.txt \
    --output results/triple_000_tracin_with_gt.csv
```

No more parsing warnings! ✅

## Backward Compatibility

**Existing files:** If you have old files generated before this fix, they will still cause parsing issues. You should:

1. **Re-run `find_mechanistic_paths.py`** to regenerate the files with proper CSV formatting
2. **Or** manually add quotes to list values in existing files (not recommended)

**The `add_ground_truth_column.py` script:** Still handles both formats gracefully due to our earlier fixes, but proper CSV format is now the standard.

## Summary

| Aspect | Before | After |
|--------|--------|-------|
| **File format** | Pseudo-CSV (unquoted) | Proper CSV (quoted) |
| **List values** | `[A, B, C]` (unquoted) | `"[A, B, C]"` (quoted) |
| **Parsing** | Breaks on commas | Works correctly |
| **Compatibility** | Manual string parsing required | Standard `csv.reader` works |
| **Data quality issues** | Hidden by parsing errors | Clear and visible |

## Best Practices Going Forward

1. ✅ **Always use `csv.writer`** for CSV output
2. ✅ **Use `newline=''`** when opening CSV files in write mode
3. ✅ **Use `csv.QUOTE_MINIMAL`** for automatic quoting of special characters
4. ✅ **Test with `csv.reader`** to verify output is properly formatted
5. ❌ **Never use manual string formatting** for CSV files

## References

- Python CSV module docs: https://docs.python.org/3/library/csv.html
- CSV RFC 4180 spec: https://tools.ietf.org/html/rfc4180
