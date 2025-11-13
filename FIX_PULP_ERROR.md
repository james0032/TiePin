# Fix for Pulp Compatibility Error

## Problem
Snakemake 7.32.4 requires pulp >= 2.7.0 which uses the newer `list_solvers()` API, but an older version of pulp is installed that only has `listSolvers()`.

## Solution

Run these commands in your Docker container:

```bash
# Uninstall the conflicting packages
pip uninstall snakemake pulp -y

# Clear pip cache to ensure clean install
pip cache purge

# Reinstall with correct versions
pip install 'snakemake==7.32.4' 'pulp==2.7.0'

# Verify installation
snakemake --version
python -c "import pulp; print(f'pulp version: {pulp.__version__}')"
```

## Verification

After running the above commands, you should see:
- Snakemake version: 7.32.4
- pulp version: 2.7.0

Then you can run:
```bash
snakemake --cores all
```

## Alternative: Install all requirements from scratch

If the above doesn't work, try:

```bash
# Reinstall all requirements
pip install -r requirements.txt --force-reinstall --no-cache-dir
```

## Notes

- The error occurs because snakemake tries to call `pulp.list_solvers()` but older pulp versions use `pulp.listSolvers()` (camelCase)
- pulp 2.7.0+ uses the snake_case convention that snakemake 7.x expects
