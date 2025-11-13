#!/usr/bin/env python3
"""
Verify that snakemake and pulp are correctly installed with compatible versions.
"""

import sys

def verify_pulp():
    """Check pulp installation and API compatibility."""
    try:
        import pulp
        print(f"✓ pulp installed: version {pulp.__version__}")

        # Check if the new API is available
        if hasattr(pulp, 'list_solvers'):
            print("✓ pulp.list_solvers() is available (new API)")
            solvers = pulp.list_solvers(onlyAvailable=True)
            print(f"  Available solvers: {solvers}")
            return True
        else:
            print("✗ pulp.list_solvers() not found (old API)")
            print("  Found: pulp.listSolvers() - need to upgrade pulp")
            return False
    except ImportError as e:
        print(f"✗ pulp not installed: {e}")
        return False

def verify_snakemake():
    """Check snakemake installation."""
    try:
        import snakemake
        version = snakemake.__version__
        print(f"✓ snakemake installed: version {version}")

        if version >= "7.32.0":
            print("✓ snakemake version is compatible (>= 7.32.0)")
            return True
        else:
            print(f"✗ snakemake version {version} may be too old")
            return False
    except ImportError as e:
        print(f"✗ snakemake not installed: {e}")
        return False

def main():
    print("Verifying dependency compatibility...\n")

    pulp_ok = verify_pulp()
    print()
    snakemake_ok = verify_snakemake()
    print()

    if pulp_ok and snakemake_ok:
        print("=" * 60)
        print("✓ All dependencies are correctly installed!")
        print("=" * 60)
        print("\nYou can now run: snakemake --cores all")
        return 0
    else:
        print("=" * 60)
        print("✗ Dependency issues found!")
        print("=" * 60)
        print("\nPlease run the fix commands from FIX_PULP_ERROR.md")
        return 1

if __name__ == "__main__":
    sys.exit(main())
