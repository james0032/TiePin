"""
Resolve lexical names to CURIE IDs using the NCATS Translator NameResolution service.

API Documentation: https://name-resolution-sri.renci.org/docs
GitHub: https://github.com/NCATSTranslator/NameResolution

Input: CSV with Drug and Disease columns (lexical names)
Output: CSV with Drug, Drug_CURIE, Disease, Disease_CURIE columns
"""

import pandas as pd
import requests
import time
from pathlib import Path
from typing import Optional


NAME_RESOLUTION_URL = "https://name-resolution-sri.renci.org"


def lookup_name(
    name: str,
    biolink_type: Optional[str] = None,
    limit: int = 1
) -> Optional[str]:
    """
    Look up a single name and return the top CURIE match.

    Args:
        name: The lexical name to resolve
        biolink_type: Optional biolink type filter (e.g., "Drug", "Disease")
        limit: Number of results to return

    Returns:
        The top CURIE match, or None if no match found
    """
    params = {
        "string": name,
        "autocomplete": "false",
        "limit": limit
    }

    if biolink_type:
        params["biolink_type"] = biolink_type

    try:
        response = requests.get(
            f"{NAME_RESOLUTION_URL}/lookup",
            params=params,
            timeout=30
        )
        response.raise_for_status()
        results = response.json()

        if results and len(results) > 0:
            return results[0].get("curie")
        return None

    except requests.exceptions.RequestException as e:
        print(f"  Error looking up '{name}': {e}")
        return None


def bulk_lookup(
    names: list[str],
    biolink_type: Optional[str] = None,
    limit: int = 1
) -> dict[str, Optional[str]]:
    """
    Bulk lookup multiple names at once.

    Args:
        names: List of names to resolve
        biolink_type: Optional biolink type filter
        limit: Number of results per name

    Returns:
        Dictionary mapping names to their top CURIE match (or None)
    """
    payload = {
        "strings": names,
        "autocomplete": False,
        "limit": limit
    }

    if biolink_type:
        payload["biolink_type"] = biolink_type

    try:
        response = requests.post(
            f"{NAME_RESOLUTION_URL}/bulk-lookup",
            json=payload,
            timeout=120
        )
        response.raise_for_status()
        results = response.json()

        # Extract top CURIE for each name
        curie_map = {}
        for name in names:
            matches = results.get(name, [])
            if matches and len(matches) > 0:
                curie_map[name] = matches[0].get("curie")
            else:
                curie_map[name] = None

        return curie_map

    except requests.exceptions.RequestException as e:
        print(f"  Error in bulk lookup: {e}")
        # Fall back to individual lookups
        return {name: lookup_name(name, biolink_type, limit) for name in names}


def resolve_names(input_path: str, output_path: str, batch_size: int = 50):
    """
    Resolve all drug and disease names in the input CSV to CURIEs.

    Args:
        input_path: Path to input CSV with Drug and Disease columns
        output_path: Path for output CSV with CURIE columns added
        batch_size: Number of names to process in each batch
    """
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path, encoding='utf-8-sig')  # Handle BOM
    print(f"Loaded {len(df):,} rows")

    # Get unique names
    unique_drugs = df["Drug"].dropna().unique().tolist()
    unique_diseases = df["Disease"].dropna().unique().tolist()

    print(f"\nUnique drugs: {len(unique_drugs):,}")
    print(f"Unique diseases: {len(unique_diseases):,}")

    # Resolve drugs (using ChemicalEntity biolink type)
    print("\n=== Resolving Drug Names ===")
    drug_curies = {}

    for i in range(0, len(unique_drugs), batch_size):
        batch = unique_drugs[i:i + batch_size]
        print(f"  Processing drugs {i+1}-{i+len(batch)} of {len(unique_drugs)}...")

        batch_results = bulk_lookup(batch, biolink_type="ChemicalEntity", limit=1)
        drug_curies.update(batch_results)

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Resolve diseases (using Disease biolink type)
    print("\n=== Resolving Disease Names ===")
    disease_curies = {}

    for i in range(0, len(unique_diseases), batch_size):
        batch = unique_diseases[i:i + batch_size]
        print(f"  Processing diseases {i+1}-{i+len(batch)} of {len(unique_diseases)}...")

        batch_results = bulk_lookup(batch, biolink_type="Disease", limit=1)
        disease_curies.update(batch_results)

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Add CURIE columns to dataframe
    df["Drug_CURIE"] = df["Drug"].map(drug_curies)
    df["Disease_CURIE"] = df["Disease"].map(disease_curies)

    # Reorder columns
    df = df[["Drug", "Drug_CURIE", "Disease", "Disease_CURIE"]]

    # Save output
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Print statistics
    drug_resolved = df["Drug_CURIE"].notna().sum()
    disease_resolved = df["Disease_CURIE"].notna().sum()

    print(f"\n=== Resolution Statistics ===")
    print(f"Drugs resolved:    {drug_resolved:,}/{len(df):,} ({100*drug_resolved/len(df):.1f}%)")
    print(f"Diseases resolved: {disease_resolved:,}/{len(df):,} ({100*disease_resolved/len(df):.1f}%)")

    # Show unresolved names
    unresolved_drugs = [name for name, curie in drug_curies.items() if curie is None]
    unresolved_diseases = [name for name, curie in disease_curies.items() if curie is None]

    if unresolved_drugs:
        print(f"\nUnresolved drugs ({len(unresolved_drugs)}):")
        for name in unresolved_drugs[:10]:
            print(f"  - {name}")
        if len(unresolved_drugs) > 10:
            print(f"  ... and {len(unresolved_drugs) - 10} more")

    if unresolved_diseases:
        print(f"\nUnresolved diseases ({len(unresolved_diseases)}):")
        for name in unresolved_diseases[:10]:
            print(f"  - {name}")
        if len(unresolved_diseases) > 10:
            print(f"  ... and {len(unresolved_diseases) - 10} more")

    print("\nDone!")


def main():
    base_path = Path("/Users/jchung/Documents/RENCI/everycure/git/conve_pykeen/data/clean_baseline/EC_MOA_pairs")

    input_path = base_path / "ec_moa_pairs_names.csv"
    output_path = base_path / "ec_moa_pairs_with_curies.csv"

    resolve_names(str(input_path), str(output_path))


if __name__ == "__main__":
    main()
