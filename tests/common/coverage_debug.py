#!/usr/bin/env python3
"""
Coverage Debug Script for COCONUT Tests
Helps diagnose why coverage is 0% and provides fixes.
"""

import os
import sys
import importlib.util
import subprocess


def check_python_path():
    """Check Python path and module availability."""
    print("🔍 PYTHON PATH ANALYSIS")
    print("=" * 50)
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python executable: {sys.executable}")
    print(f"Python path entries:")
    for i, path in enumerate(sys.path):
        print(f"  {i}: {path}")
    print()


def check_module_files():
    """Check what module files exist."""
    print("📁 MODULE FILE ANALYSIS")
    print("=" * 50)

    # Check in order of preference: coconut.py first since user mentioned it
    candidates = [
        "coconut.py",  # User's actual file
        "coconut_improved.py",  # Artifact name
        "coconut_module.py",  # Alternative
        "coconut_bio.py",  # Alternative
    ]

    found_modules = []
    for filename in candidates:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"  ✅ Found: {filename} ({size} bytes)")
            found_modules.append(filename)

            # Special note for coconut.py since it's the user's file
            if filename == "coconut.py":
                print(f"      📌 This appears to be your main implementation file!")
        else:
            print(f"  ❌ Not found: {filename}")

    if "coconut.py" in found_modules:
        print(f"\n💡 Note: Found coconut.py - this is likely your implementation file.")
        print(
            f"   Make sure your tests import from 'coconut' module, not 'coconut_improved'"
        )

    print(f"\nFound {len(found_modules)} potential module file(s)")
    return found_modules


def test_module_import(module_files):
    """Test importing each module file."""
    print("\n🔧 MODULE IMPORT TESTING")
    print("=" * 50)

    successful_imports = []

    for filename in module_files:
        module_name = filename.replace(".py", "")
        print(f"\nTesting: {filename}")

        try:
            # Test file compilation
            print(f"  Compiling {filename}...")
            result = subprocess.run(
                [sys.executable, "-m", "py_compile", filename],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                print(f"  ❌ Compilation failed: {result.stderr}")
                continue
            else:
                print(f"  ✅ Compilation successful")

            # Test import
            print(f"  Importing {module_name}...")
            spec = importlib.util.spec_from_file_location(module_name, filename)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"  ✅ Import successful")

            # Check for key functions
            key_functions = ["coconut", "combine_coco_output", "_it_sol", "_int_eprior"]
            found_functions = []
            for func_name in key_functions:
                if hasattr(module, func_name):
                    print(f"    ✅ Has function: {func_name}")
                    found_functions.append(func_name)
                else:
                    print(f"    ❌ Missing function: {func_name}")

            successful_imports.append((filename, module_name, module, found_functions))

        except Exception as e:
            print(f"  ❌ Import failed: {e}")

    return successful_imports


def test_coverage_measurement(successful_imports):
    """Test coverage measurement on imported modules."""
    print("\n📊 COVERAGE MEASUREMENT TESTING")
    print("=" * 50)

    try:
        import coverage
    except ImportError:
        print("❌ Coverage package not installed. Install with: pip install coverage")
        return

    for filename, module_name, module, functions in successful_imports:
        print(f"\nTesting coverage on: {filename}")

        # Create a coverage instance
        cov = coverage.Coverage(source=[module_name])
        cov.start()

        # Try to execute some code from the module
        try:
            if "get_a_prior" in functions:
                print("  Testing get_a_prior...")
                try:
                    result = module.get_a_prior([1.0, 2.0, 3.0])
                    print("  ✅ get_a_prior executed (or raised expected error)")
                except NotImplementedError:
                    print("  ✅ get_a_prior is placeholder (expected)")
                except Exception as e:
                    print(f"  ⚠️ get_a_prior error: {e}")

            if "coconut" in functions:
                print("  Testing coconut function...")
                try:
                    # This should fail with NotImplementedError if it's a placeholder
                    result = module.coconut([], "test_col")
                    print("  ⚠️ coconut function returned result (unexpected)")
                except NotImplementedError:
                    print("  ✅ coconut is placeholder (expected)")
                except Exception as e:
                    print(f"  ⚠️ coconut error: {e}")

        except Exception as e:
            print(f"  ❌ Error testing functions: {e}")

        # Stop coverage and get report
        cov.stop()
        cov.save()

        print("  Coverage report:")
        try:
            # Capture coverage report
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                cov.report()
            report = f.getvalue()

            if report.strip():
                print("  " + "\n  ".join(report.split("\n")))
            else:
                print("  ❌ No coverage data collected")

        except Exception as e:
            print(f"  ❌ Error generating coverage report: {e}")


def test_pytest_coverage():
    """Test pytest coverage commands."""
    print("\n🧪 PYTEST COVERAGE TESTING")
    print("=" * 50)

    # Check if pytest is available
    try:
        import pytest

        print("✅ pytest is available")
    except ImportError:
        print("❌ pytest not installed. Install with: pip install pytest pytest-cov")
        return

    # Test different coverage commands
    test_commands = [
        "pytest --version",
        "pytest --cov-help",
    ]

    for cmd in test_commands:
        print(f"\nTesting command: {cmd}")
        try:
            result = subprocess.run(
                cmd.split(), capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                print("  ✅ Command successful")
                if result.stdout:
                    print(
                        "  Output:",
                        (
                            result.stdout[:200] + "..."
                            if len(result.stdout) > 200
                            else result.stdout
                        ),
                    )
            else:
                print(f"  ❌ Command failed: {result.stderr}")
        except Exception as e:
            print(f"  ❌ Command error: {e}")


def suggest_fixes(successful_imports):
    """Suggest fixes based on analysis."""
    print("\n💡 COVERAGE FIX SUGGESTIONS")
    print("=" * 50)

    if not successful_imports:
        print("❌ NO MODULES IMPORTED SUCCESSFULLY")
        print("\n🔧 IMMEDIATE ACTIONS:")
        if os.path.exists("coconut.py"):
            print("1. ✅ Found coconut.py - this appears to be your implementation")
            print(
                "2. Update test imports to use 'coconut' instead of 'coconut_improved'"
            )
            print("3. Run coverage with: pytest --cov=coconut test_coconut.py -v")
        else:
            print("1. Create coconut.py with your COCONUT implementation")
            print("2. Copy content from your implementation file")
            print("3. Verify syntax: python -m py_compile coconut.py")
            print("4. Test import: python -c 'import coconut; print(\"Success\")'")
        return

    print(f"✅ Found {len(successful_imports)} importable module(s)")

    # Special handling for coconut.py
    coconut_py_found = any(
        filename == "coconut.py" for filename, _, _, _ in successful_imports
    )

    if coconut_py_found:
        print("\n🎯 COCONUT.PY DETECTED - LIKELY SOLUTION:")
        print(
            "Your real implementation is in coconut.py but tests may be importing from elsewhere."
        )
        print("\nQuick fix:")
        print("1. Update test_coconut.py imports:")
        print("   from coconut import coconut, combine_coco_output, ...")
        print("2. Run coverage:")
        print("   pytest --cov=coconut --cov-report=term-missing test_coconut.py -v")
        print("3. Expected result: >0% coverage")

    # Check for mock functions vs real implementations
    for filename, module_name, module, functions in successful_imports:
        print(f"\n📋 Analysis for {filename}:")

        # Test if functions are mocks
        mock_functions = []
        real_functions = []

        for func_name in functions:
            if hasattr(module, func_name):
                try:
                    func = getattr(module, func_name)
                    if func_name == "get_a_prior":
                        func(
                            [1.0, 2.0]
                        )  # This should raise NotImplementedError if mock
                    mock_functions.append(func_name)
                except NotImplementedError:
                    mock_functions.append(func_name)
                except:
                    real_functions.append(func_name)

        if mock_functions and not real_functions:
            print(f"  ⚠️ ALL FUNCTIONS ARE MOCKS/PLACEHOLDERS")
            print(f"    Mock functions: {', '.join(mock_functions)}")
            print(f"    This explains 0% coverage - no real code is executed")
            print(f"    Fix: Replace {filename} with actual implementation")
        elif real_functions:
            print(f"  ✅ REAL FUNCTIONS DETECTED")
            print(f"    Real functions: {', '.join(real_functions)}")
            if mock_functions:
                print(f"    Mock functions: {', '.join(mock_functions)}")
            print(f"    Coverage should work with: pytest --cov={module_name}")

    print("\n🚀 RECOMMENDED NEXT STEPS:")

    # Prioritize coconut.py if available
    if coconut_py_found:
        print("Since coconut.py was found:")
        print(
            "1. Test import: python -c 'from coconut import coconut; print(\"✅ Success\")'"
        )
        print(
            "2. Run coverage: pytest --cov=coconut test_coconut.py::TestCoconutMainFunction::test_simple_coconut_functionality -v"
        )
        print(
            "3. Full coverage: pytest --cov=coconut --cov-report=html test_coconut.py"
        )
    else:
        best_module = successful_imports[0][1]  # Use first successful import
        print(f"Using {best_module} module:")
        print(f"1. Test basic coverage:")
        print(
            f"   coverage run --source={best_module} -m pytest test_coconut.py::TestCoconutMainFunction::test_simple_coconut_functionality -v"
        )
        print(f"   coverage report -m")

        print(f"\n2. Test pytest coverage:")
        print(
            f"   pytest --cov={best_module} --cov-report=term-missing test_coconut.py -v"
        )

        print(f"\n3. Generate HTML coverage report:")
        print(f"   pytest --cov={best_module} --cov-report=html test_coconut.py")
        print(f"   open htmlcov/index.html")

    print(f"\n4. If still 0%, verify real functions are imported in tests:")
    if coconut_py_found:
        print(f"   # In test_coconut.py, ensure you have:")
        print(f"   from coconut import coconut, combine_coco_output, ...")
    else:
        best_module = successful_imports[0][1]
        print(f"   # In test_coconut.py, ensure you have:")
        print(f"   from {best_module} import coconut, combine_coco_output, ...")


def main():
    """Main diagnostic function."""
    print("🔍 COVERAGE DIAGNOSTIC TOOL")
    print("=" * 60)

    check_python_path()
    module_files = check_module_files()

    if not module_files:
        print("\n❌ No module files found!")
        print(
            "Please create coconut_improved.py with the actual COCONUT implementation"
        )
        return False

    successful_imports = test_module_import(module_files)
    test_coverage_measurement(successful_imports)
    test_pytest_coverage()
    suggest_fixes(successful_imports)

    print("\n" + "=" * 60)
    print("Coverage diagnostic complete!")
    print("=" * 60)

    return len(successful_imports) > 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
