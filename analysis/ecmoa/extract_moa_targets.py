"""Extract gene/protein/substance targets from 'Ideal/complete MOA' column.

Parses the MOA free-text descriptions to extract intermediate molecular
entities (genes, proteins, pathways, substances) between drug and disease.
Adds a 'target' column with semicolon-separated entity names.
"""

import pandas as pd
import re
from pathlib import Path
import sys


# All linking phrases between entities in MOA descriptions.
# Sorted longest-first at split time to avoid partial matches.
SPLIT_PHRASES = [
    # Action verbs
    'post-translationally modifies',
    'inhibits', 'targets', 'activates', 'binds', 'chelates',
    'stimulates', 'modifies', 'desensitizes',
    'supports', 'downregulates', 'upregulates',
    'treats', 'express', 'expresses',
    # Connector phrases (entity-to-entity or entity-to-disease)
    'is dysregulated in', 'dysregulated in',
    'mutated in', 'expressed in', 'overexpressed in', 'overactive in',
    'expressed by', 'activated by', 'regulated by', 'repressed by',
    'characterizes', 'contributes to', 'activated in', 'accumulates in',
    'involved in', 'associated with', 'observed in',
    'defective in', 'required in', 'lost in',
    'important in', 'decreased in',
    'includes', 'contains', 'is a', 'is an',
    'opposes', 'promotes', 'feature of',
    'member of', 'deposits', 'part of',
    'overactive in', 'active in',
    'causes', 'cause',
    'also includes',
    'affected by',
    'stimulate', 'production of',
    'contribute to',
]


def extract_targets(moa_text: str, drug: str, disease: str) -> list[str]:
    """Extract gene/protein/substance targets from MOA text.

    Strategy: Remove the drug name from the beginning and disease name from
    the end, then split on all linking phrases to isolate entity names.

    Args:
        moa_text: The full MOA description
        drug: Drug name to strip from beginning
        disease: Disease name to strip from end

    Returns:
        List of extracted target entity names
    """
    if not moa_text or pd.isna(moa_text):
        return []

    text = moa_text.strip()

    # Remove leading drug name (case-insensitive)
    if text.lower().startswith(drug.lower()):
        text = text[len(drug):].strip()

    # Remove trailing disease name (case-insensitive)
    lower_text = text.lower()
    lower_disease = disease.lower()
    disease_idx = lower_text.rfind(lower_disease)
    if disease_idx > 0:
        text = text[:disease_idx].strip()

    # Split by all linking phrases (longest first to avoid partial matches)
    phrase_pattern = '|'.join(
        re.escape(p) for p in sorted(SPLIT_PHRASES, key=len, reverse=True)
    )
    segments = re.split(f'(?i)\\b(?:{phrase_pattern})\\b', text)

    targets = []
    for seg in segments:
        seg = seg.strip().rstrip(',').strip()
        if not seg:
            continue

        # Skip if it's just the drug or disease name (or close variant)
        seg_lower = seg.lower()
        if seg_lower == drug.lower() or seg_lower == disease.lower():
            continue

        # Fuzzy match: skip if segment is very similar to drug name (typos)
        if len(seg) > 3 and len(drug) > 3:
            from difflib import SequenceMatcher
            ratio = SequenceMatcher(None, seg_lower, drug.lower()).ratio()
            if ratio > 0.85:
                continue

        # Skip very long segments (likely full sentences, not entity names)
        if len(seg) > 60:
            continue

        # Skip pure filler words
        if seg.lower() in ('is', 'the', 'a', 'an', 'of', 'and', 'or', 'also'):
            continue

        if seg:
            targets.append(seg)

    return targets


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        csv_path = Path(
            "/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/"
            "data/ec_moa_pairs/Favorite mechanistic pairs.csv"
        )

    df = pd.read_csv(csv_path)

    # Drop empty rows
    df = df.dropna(subset=['Drug', 'Disease', 'Ideal/complete MOA'], how='any')
    df = df[df['Drug'].str.strip() != '']

    # Extract targets for each row
    all_targets = []
    for _, row in df.iterrows():
        drug = str(row['Drug']).strip()
        disease = str(row['Disease']).strip()
        moa = str(row['Ideal/complete MOA']).strip()

        targets = extract_targets(moa, drug, disease)
        all_targets.append('; '.join(targets) if targets else '')

    df['target'] = all_targets

    # Report
    print(f"Processed {len(df)} rows")
    print(f"Rows with extracted targets: {sum(1 for t in all_targets if t)}")
    print()
    for _, row in df.iterrows():
        print(f"  Drug: {row['Drug']}")
        print(f"  Disease: {row['Disease']}")
        print(f"  MOA: {row['Ideal/complete MOA']}")
        print(f"  -> target: {row['target']}")
        print()

    # Save output
    # Keep only relevant columns
    out_cols = [
        'Category (readily able to surface vs. stretch)',
        'Plausible/established/red herring?',
        'Drug', 'Disease',
        'Ideal/complete MOA',
        'Abbreviated/bare minimum MOA',
        'Entity pattern (of abbreviated MOA)',
        'target',
        'Notes',
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df_out = df[out_cols]

    out_path = csv_path.parent / (csv_path.stem + '_with_targets.csv')
    df_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
