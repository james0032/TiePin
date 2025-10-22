"""
Syntax validation and basic static analysis tests.
Tests that don't require PyKeen installation.
"""

import ast
import sys
from pathlib import Path


def test_python_syntax(file_path):
    """Test if a Python file has valid syntax."""
    print(f"\n{'='*60}")
    print(f"Testing: {file_path}")
    print('='*60)

    with open(file_path, 'r') as f:
        code = f.read()

    try:
        ast.parse(code)
        print("✓ Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"✗ Syntax error: {e}")
        return False


def test_imports(file_path):
    """Test that imports are properly structured."""
    print("\nChecking imports...")

    with open(file_path, 'r') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.append(node.module)

        print(f"✓ Found {len(imports)} import statements")
        return True
    except Exception as e:
        print(f"✗ Import check failed: {e}")
        return False


def test_function_definitions(file_path):
    """Test that functions are properly defined."""
    print("\nChecking function definitions...")

    with open(file_path, 'r') as f:
        code = f.read()

    try:
        tree = ast.parse(code)
        functions = []
        classes = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.append(node.name)

        print(f"✓ Found {len(classes)} classes")
        print(f"✓ Found {len(functions)} functions")

        if len(functions) == 0 and len(classes) == 0:
            print("⚠ Warning: No functions or classes found")
            return False

        return True
    except Exception as e:
        print(f"✗ Function definition check failed: {e}")
        return False


def test_docstrings(file_path):
    """Test that key functions have docstrings."""
    print("\nChecking docstrings...")

    with open(file_path, 'r') as f:
        code = f.read()

    try:
        tree = ast.parse(code)

        functions_with_docs = 0
        functions_without_docs = 0

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                docstring = ast.get_docstring(node)
                if docstring:
                    functions_with_docs += 1
                else:
                    functions_without_docs += 1

        total = functions_with_docs + functions_without_docs
        if total > 0:
            coverage = (functions_with_docs / total) * 100
            print(f"✓ Docstring coverage: {coverage:.1f}% ({functions_with_docs}/{total})")
            return True
        else:
            print("⚠ No functions/classes found")
            return True

    except Exception as e:
        print(f"✗ Docstring check failed: {e}")
        return False


def test_tracin_specific_logic():
    """Test TracIn-specific logic without running the code."""
    print(f"\n{'='*60}")
    print("Testing TracIn-specific logic")
    print('='*60)

    # Read tracin.py
    with open('tracin.py', 'r') as f:
        code = f.read()

    # Check for key methods
    required_methods = [
        'compute_gradient',
        'compute_batch_individual_gradients',
        'compute_influences_for_test_triple',
        'compute_self_influence',
        'analyze_test_set'
    ]

    print("\nChecking for required methods...")
    for method in required_methods:
        if f'def {method}' in code:
            print(f"✓ Found method: {method}")
        else:
            print(f"✗ Missing method: {method}")
            return False

    # Check for eval mode fix
    print("\nChecking for BatchNorm fix...")
    if 'self.model.eval()' in code and 'param.requires_grad = True' in code:
        print("✓ BatchNorm fix is present (model.eval() with requires_grad)")
    else:
        print("⚠ Warning: BatchNorm fix might be missing")

    # Check for batch_size parameter
    print("\nChecking for batch_size parameter...")
    if 'batch_size' in code:
        print("✓ batch_size parameter found")
    else:
        print("✗ batch_size parameter missing")
        return False

    return True


def test_run_tracin_specific_logic():
    """Test run_tracin.py-specific logic."""
    print(f"\n{'='*60}")
    print("Testing run_tracin.py-specific logic")
    print('='*60)

    # Read run_tracin.py
    with open('run_tracin.py', 'r') as f:
        code = f.read()

    # Check for modes
    print("\nChecking for analysis modes...")
    modes = ['test', 'self', 'single']
    for mode in modes:
        if f"mode == '{mode}'" in code or f'mode: {mode}' in code:
            print(f"✓ Found mode: {mode}")
        else:
            print(f"⚠ Warning: mode '{mode}' might be missing")

    # Check for CLI arguments
    print("\nChecking for CLI arguments...")
    cli_args = [
        '--model-path',
        '--train',
        '--test',
        '--entity-to-id',
        '--relation-to-id',
        '--output',
        '--batch-size',
        '--device'
    ]

    for arg in cli_args:
        if arg in code:
            print(f"✓ Found CLI arg: {arg}")
        else:
            print(f"✗ Missing CLI arg: {arg}")

    # Check for batch_size parameter
    print("\nChecking if batch_size is passed to analyzer...")
    if 'batch_size=batch_size' in code or 'batch_size=args.batch_size' in code:
        print("✓ batch_size is passed to analyzer")
    else:
        print("⚠ Warning: batch_size might not be passed correctly")

    return True


def main():
    """Run all tests."""
    print("="*60)
    print("STATIC CODE ANALYSIS")
    print("="*60)

    all_passed = True

    # Test tracin.py
    files_to_test = ['tracin.py', 'run_tracin.py']

    for file_path in files_to_test:
        if not Path(file_path).exists():
            print(f"\n✗ File not found: {file_path}")
            all_passed = False
            continue

        passed = test_python_syntax(file_path)
        all_passed = all_passed and passed

        if passed:
            passed = test_imports(file_path)
            all_passed = all_passed and passed

            passed = test_function_definitions(file_path)
            all_passed = all_passed and passed

            passed = test_docstrings(file_path)
            all_passed = all_passed and passed

    # TracIn-specific tests
    if Path('tracin.py').exists():
        passed = test_tracin_specific_logic()
        all_passed = all_passed and passed

    if Path('run_tracin.py').exists():
        passed = test_run_tracin_specific_logic()
        all_passed = all_passed and passed

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ ALL TESTS PASSED")
    else:
        print("✗ SOME TESTS FAILED")
    print('='*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
