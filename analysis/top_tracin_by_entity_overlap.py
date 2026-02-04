"""
Get top 10 TracIn score edges categorized by entity overlap with test triple.
Outputs separate CSV files per category:
1. shares_head: TrainHead or TrainTail matches TestHead
2. shares_tail: TrainHead or TrainTail matches TestTail
3. shares_both: TrainHead or TrainTail matches both TestHead and TestTail
4. shares_neither: no entity overlap with test triple
"""

import pandas as pd
from pathlib import Path


def categorize_and_save(csv_path: Path, output_dir: Path = None, top_k: int = 10):
    df = pd.read_csv(csv_path)

    test_head = str(df.iloc[0]['TestHead'])
    test_tail = str(df.iloc[0]['TestTail'])

    df['TrainHead_str'] = df['TrainHead'].astype(str)
    df['TrainTail_str'] = df['TrainTail'].astype(str)

    head_match = (df['TrainHead_str'] == test_head) | (df['TrainTail_str'] == test_head)
    tail_match = (df['TrainHead_str'] == test_tail) | (df['TrainTail_str'] == test_tail)

    df_sorted = df.sort_values('TracInScore', ascending=False)

    if output_dir is None:
        output_dir = csv_path.parent

    keep_cols = [
        'TestHead', 'TestHead_label', 'TestTail', 'TestTail_label',
        'TrainHead', 'TrainHead_label', 'TrainRel_label', 'TrainTail', 'TrainTail_label',
        'TracInScore', 'IsGroundTruth', 'In_path', 'On_specific_path'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]

    stem = csv_path.stem
    categories = [
        ('shares_head', head_match & ~tail_match),
        ('shares_tail', tail_match & ~head_match),
        ('shares_both', head_match & tail_match),
        ('shares_neither', ~head_match & ~tail_match),
    ]

    print(f"Processing: {csv_path.name}")
    for category, mask in categories:
        subset = df_sorted[mask].head(top_k)[keep_cols]
        out_path = output_dir / f"{stem}_{category}.csv"
        subset.to_csv(out_path, index=False)
        print(f"  {category}: {len(subset)} rows -> {out_path.name}")
    print()


def main():
    base = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline/examples")

    files = [
        base / "triple_005_CHEBI_30411_MONDO_0008228_tracin_with_gt.csv",
        base / "triple_006_CHEBI_50381_MONDO_0018150_tracin_with_gt.csv",
    ]

    for csv_file in files:
        categorize_and_save(csv_file)


if __name__ == "__main__":
    main()
