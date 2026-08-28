#!/usr/bin/env python3
"""
Test runner script for COCONUT module.

This script provides convenient ways to run different types of tests
for the COCONUT bioinformatics package.

Usage:
    python run_coconut_tests.py [test_type] [options]

Test types:
    all         - Run all tests (default)
    unit        - Run only unit tests
    integration - Run only integration tests
    fast        - Run fast tests only (exclude slow tests)
    performance - Run performance and stress tests
    coverage    - Run tests with coverage report

Examples:
    python run_coconut_tests.py                    # Run all tests
    python run_coconut_tests.py unit               # Run unit tests only
    python run_coconut_tests.py coverage           # Run with coverage
    python run_coconut_tests.py fast               # Skip slow tests
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(cmd, capture_output=False):
    """Run a command and handle the result."""
    print(f"Running: {' '.join(cmd)}")
    try:
        if capture_output:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
        else:
            subprocess.run(cmd, check=True)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if capture_output and e.stderr:
            print(f"Error output: {e.stderr}")
        sys.exit(e.returncode)


def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = ["pytest", "pandas", "numpy", "scipy"]
    missing = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Please install test dependencies:")
        print("pip install -r test-requirements.txt")
        sys.exit(1)


def run_all_tests():
    """Run all tests."""
    cmd = ["pytest", "test_coconut.py", "-v"]
    run_command(cmd)


def run_unit_tests():
    """Run only unit tests."""
    cmd = ["pytest", "test_coconut.py", "-v", "-m", "not integration"]
    run_command(cmd)


def run_integration_tests():
    """Run only integration tests."""
    cmd = ["pytest", "test_coconut.py::TestIntegrationTests", "-v"]
    run_command(cmd)


def run_fast_tests():
    """Run fast tests only (exclude slow tests)."""
    cmd = ["pytest", "test_coconut.py", "-v", "-m", "not slow"]
    run_command(cmd)


def run_performance_tests():
    """Run performance and stress tests."""
    cmd = ["pytest", "test_coconut.py::TestPerformanceAndStress", "-v", "-s"]
    run_command(cmd)


def basic_coverage_debug():
    """Basic coverage debugging when full debug script isn't available."""
    print("Running basic coverage diagnostics...")

    # Check for module files - prioritize coconut.py since that's what user has
    module_files = ["coconut.py", "coconut_improved.py", "coconut_module.py"]
    found_modules = [f for f in module_files if os.path.exists(f)]

    if not found_modules:
        print("❌ No COCONUT module files found!")
        print("Expected files: " + ", ".join(module_files))
        print("💡 Since you mentioned having coconut.py, make sure it's in the current directory")
        return

    print(f"✅ Found modules: {found_modules}")

    # Special note for coconut.py
    if "coconut.py" in found_modules:
        print("📌 Found coconut.py - this is likely your implementation file!")
        print("   Make sure tests import from 'coconut' module, not 'coconut_improved'")

    # Test import
    for module_file in found_modules:
        module_name = module_file.replace(".py", "")
        print(f"\nTesting import of {module_name}...")

        cmd = [
            sys.executable,
            "-c",
            f'import {module_name}; print("✅ Import successful")',
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(result.stdout.strip())
            else:
                print(f"❌ Import failed: {result.stderr.strip()}")
        except Exception as e:
            print(f"❌ Error testing import: {e}")

    # Test basic coverage command
    if found_modules:
        # Prioritize coconut.py if it exists
        best_module = (
            "coconut" if "coconut.py" in found_modules else found_modules[0].replace(".py", "")
        )
        print(f"\nTesting coverage with {best_module}...")

        coverage_cmd = [
            "coverage",
            "run",
            f"--source={best_module}",
            "-c",
            f'import {best_module}; print("Coverage test")',
        ]

        try:
            result = subprocess.run(coverage_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                print("✅ Basic coverage test successful")
                # Show coverage report
                subprocess.run(["coverage", "report"])
            else:
                print(f"❌ Coverage test failed: {result.stderr}")
        except FileNotFoundError:
            print("❌ Coverage tool not found. Install with: pip install coverage")
        except Exception as e:
            print(f"❌ Coverage test error: {e}")


def run_with_coverage():
    """Run tests with coverage reporting."""
    print("Running tests with coverage analysis...")

    # First, try to detect the correct module name
    module_candidates = ["coconut", "coconut_improved", "coconut_module", "coconut_bio"]
    target_module = None

    for module_name in module_candidates:
        if os.path.exists(f"{module_name}.py"):
            print(f"Found module file: {module_name}.py")
            target_module = module_name

            # Special note for coconut.py
            if module_name == "coconut":
                print("📌 Using coconut.py - this appears to be your main implementation!")
            break

    if not target_module:
        print("❌ No COCONUT module file found!")
        print("Please ensure coconut_improved.py (or similar) exists")
        return False

    print(f"Using module: {target_module}")

    # Test different coverage approaches
    approaches = [
        {
            "name": "Direct coverage with source specification",
            "cmd": [
                "coverage",
                "run",
                f"--source={target_module}",
                "-m",
                "pytest",
                "test_coconut.py::TestCoconutMainFunction::test_simple_coconut_functionality",
                "-v",
            ],
        },
        {
            "name": "Pytest with coverage plugin",
            "cmd": [
                "pytest",
                f"--cov={target_module}",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "test_coconut.py::TestCoconutMainFunction::test_simple_coconut_functionality",
                "-v",
            ],
        },
    ]

    for approach in approaches:
        print(f"\n--- {approach['name']} ---")
        try:
            print(f"Running: {' '.join(approach['cmd'])}")
            result = subprocess.run(approach["cmd"], check=False, timeout=60)

            if result.returncode == 0:
                print(f"✅ {approach['name']} succeeded")

                # If using coverage directly, show report
                if approach["cmd"][0] == "coverage":
                    print("\nGenerating coverage report...")
                    subprocess.run(["coverage", "report", "-m"], check=False)

                print(f"\nTo view HTML coverage report: open htmlcov/index.html")
                return True
            else:
                print(
                    f"⚠️ {approach['name']} completed with issues (exit code: {result.returncode})"
                )

        except subprocess.TimeoutExpired:
            print(f"❌ {approach['name']} timed out")
        except FileNotFoundError:
            print(f"❌ {approach['name']} - command not found")
        except Exception as e:
            print(f"❌ {approach['name']} failed: {e}")

    print("\n💡 Coverage debugging suggestions:")
    print("1. Run: python coverage_debug.py")
    print("2. Verify module imports: python -c 'import coconut_improved; print(\"Success\")'")
    print("3. Check for mock vs real functions")

    return False


def run_specific_test_class(class_name):
    """Run tests from a specific test class."""
    cmd = ["pytest", f"test_coconut.py::{class_name}", "-v"]
    run_command(cmd)


def run_specific_test(test_name):
    """Run a specific test method."""
    cmd = ["pytest", f"test_coconut.py::{test_name}", "-v", "-s"]
    run_command(cmd)


def lint_code():
    """Run code quality checks."""
    print("Running code quality checks...")

    # Check if linting tools are available
    tools = {
        "flake8": ["flake8", "coconut.py", "--max-line-length=100"],
        "black": ["black", "--check", "coconut.py"],
        "isort": ["isort", "--check-only", "coconut.py"],
        "mypy": ["mypy", "coconut.py"],
    }

    for tool_name, cmd in tools.items():
        try:
            __import__(tool_name.replace("-", "_"))
            print(f"\n{tool_name.upper()} check:")
            run_command(cmd)
        except ImportError:
            print(f"Skipping {tool_name} (not installed)")


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for COCONUT module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "test_type",
        nargs="?",
        default="all",
        choices=[
            "all",
            "unit",
            "integration",
            "fast",
            "performance",
            "coverage",
            "lint",
        ],
        help="Type of tests to run (default: all)",
    )

    parser.add_argument(
        "--class",
        dest="test_class",
        help="Run specific test class (e.g., TestCoconutMainFunction)",
    )

    parser.add_argument(
        "--test",
        dest="test_method",
        help="Run specific test method (e.g., TestCoconutMainFunction::test_coconut_basic_functionality)",
    )

    parser.add_argument("--verbose", action="store_true", help="Extra verbose output")

    parser.add_argument("--debug", action="store_true", help="Run with debug output and no capture")

    parser.add_argument(
        "--debug-coverage",
        action="store_true",
        help="Debug coverage issues and show detailed analysis",
    )

    args = parser.parse_args()

    # Check dependencies
    check_dependencies()

    # Handle coverage debugging
    if args.debug_coverage:
        print("🔍 Running coverage diagnostics...")
        if os.path.exists("coverage_debug.py"):
            subprocess.run([sys.executable, "coverage_debug.py"])
        else:
            print("⚠️ coverage_debug.py not found. Creating basic coverage test...")
            basic_coverage_debug()
        return

    # Handle specific test requests
    if args.test_class:
        run_specific_test_class(args.test_class)
        return

    if args.test_method:
        run_specific_test(args.test_method)
        return

    # Run tests based on type
    test_runners = {
        "all": run_all_tests,
        "unit": run_unit_tests,
        "integration": run_integration_tests,
        "fast": run_fast_tests,
        "performance": run_performance_tests,
        "coverage": run_with_coverage,
        "lint": lint_code,
    }

    runner = test_runners.get(args.test_type)
    if runner:
        runner()
    else:
        print(f"Unknown test type: {args.test_type}")
        sys.exit(1)


if __name__ == "__main__":
    main()
