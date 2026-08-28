#!/usr/bin/env python3
"""
Production-grade test runner with interactive HTML dashboard.

Features:
- Recursive test directory processing
- Accurate source-only coverage reporting
- Interactive HTML dashboard with pass/fail counts
- Clickable failure details with stack traces
- Grand total summary
- Per-directory breakdown
- Multiple output formats

Example:
  python test_runner_all.py \
    --input-dir tests \
    --recursive \
    --coverage \
    --generate-dashboard \
    --dashboard-dir test_results
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def _which(cmd: str) -> bool:
    """Return True if `cmd` is found on PATH."""
    return shutil.which(cmd) is not None


def _run(cmd: List[str], capture_output: bool = False) -> tuple:
    """Run command and optionally capture output."""
    print(f"[INFO] Executing: {' '.join(cmd)}")
    if capture_output:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    else:
        return subprocess.call(cmd), None, None


def find_test_subdirectories(input_dir: str) -> List[str]:
    """Find all subdirectories containing test files."""
    input_path = Path(input_dir).resolve()
    test_dirs = []

    for item in input_path.iterdir():
        if not item.is_dir():
            continue

        if item.name.startswith(".") or item.name in ["__pycache__", "coverage"]:
            continue

        has_tests = False
        try:
            for py_file in item.rglob("*.py"):
                if py_file.name.startswith("test_") or py_file.name.endswith("_test.py"):
                    has_tests = True
                    break
        except (PermissionError, OSError):
            continue

        if has_tests:
            test_dirs.append(str(item))

    return sorted(test_dirs)


def parse_junit_xml(xml_file: str) -> Dict[str, Any]:
    """Parse JUnit XML file to extract test results."""
    if not os.path.exists(xml_file):
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.0,
            "failures": [],
        }

    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Handle both <testsuite> and <testsuites> root elements
        if root.tag == "testsuites":
            testsuites = root.findall("testsuite")
        else:
            testsuites = [root]

        total = passed = failed = skipped = errors = 0
        duration = 0.0
        failures = []

        for testsuite in testsuites:
            total += int(testsuite.get("tests", 0))
            failed += int(testsuite.get("failures", 0))
            errors += int(testsuite.get("errors", 0))
            skipped += int(testsuite.get("skipped", 0))
            duration += float(testsuite.get("time", 0))

            # Extract failure details
            for testcase in testsuite.findall("testcase"):
                classname = testcase.get("classname", "")
                name = testcase.get("name", "")
                time = testcase.get("time", "0")

                failure = testcase.find("failure")
                error = testcase.find("error")

                if failure is not None:
                    failures.append(
                        {
                            "type": "failure",
                            "classname": classname,
                            "name": name,
                            "time": time,
                            "message": failure.get("message", ""),
                            "content": failure.text or "",
                        }
                    )
                elif error is not None:
                    failures.append(
                        {
                            "type": "error",
                            "classname": classname,
                            "name": name,
                            "time": time,
                            "message": error.get("message", ""),
                            "content": error.text or "",
                        }
                    )

        passed = total - failed - errors - skipped

        return {
            "total": total,
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "errors": errors,
            "duration": duration,
            "failures": failures,
        }
    except Exception as e:
        print(f"[WARN] Failed to parse {xml_file}: {e}")
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "duration": 0.0,
            "failures": [],
        }


def generate_html_dashboard(results: List[Dict[str, Any]], output_dir: str, args):
    """Generate interactive HTML dashboard with test results."""
    os.makedirs(output_dir, exist_ok=True)

    # Calculate grand totals
    grand_total = {
        "total": sum(r["results"]["total"] for r in results),
        "passed": sum(r["results"]["passed"] for r in results),
        "failed": sum(r["results"]["failed"] for r in results),
        "skipped": sum(r["results"]["skipped"] for r in results),
        "errors": sum(r["results"]["errors"] for r in results),
        "duration": sum(r["results"]["duration"] for r in results),
    }

    # Generate main dashboard HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Test Results Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }}

        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}

        .header h1 {{
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}

        .header .timestamp {{
            opacity: 0.9;
            font-size: 0.9rem;
        }}

        .summary-cards {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 40px;
            background: #f8f9fa;
        }}

        .card {{
            background: white;
            border-radius: 8px;
            padding: 24px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.15);
        }}

        .card-title {{
            font-size: 0.875rem;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .card-value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 4px;
        }}

        .card-subtitle {{
            font-size: 0.875rem;
            color: #6c757d;
        }}

        .card.total .card-value {{ color: #667eea; }}
        .card.passed .card-value {{ color: #10b981; }}
        .card.failed .card-value {{ color: #ef4444; }}
        .card.skipped .card-value {{ color: #f59e0b; }}
        .card.duration .card-value {{ font-size: 2rem; }}

        .results-table {{
            padding: 0 40px 40px 40px;
        }}

        .table-title {{
            font-size: 1.5rem;
            margin-bottom: 20px;
            color: #1f2937;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        th {{
            background: #f8f9fa;
            padding: 16px;
            text-align: left;
            font-weight: 600;
            color: #374151;
            border-bottom: 2px solid #e5e7eb;
        }}

        td {{
            padding: 16px;
            border-bottom: 1px solid #e5e7eb;
        }}

        tr:last-child td {{
            border-bottom: none;
        }}

        tr:hover {{
            background: #f9fafb;
        }}

        .directory-name {{
            font-weight: 500;
            color: #1f2937;
        }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.875rem;
            font-weight: 600;
        }}

        .badge-success {{
            background: #d1fae5;
            color: #065f46;
        }}

        .badge-danger {{
            background: #fee2e2;
            color: #991b1b;
        }}

        .badge-warning {{
            background: #fef3c7;
            color: #92400e;
        }}

        .clickable {{
            color: #ef4444;
            cursor: pointer;
            text-decoration: underline;
            font-weight: 600;
        }}

        .clickable:hover {{
            color: #dc2626;
        }}

        .duration {{
            color: #6b7280;
            font-size: 0.875rem;
        }}

        .footer {{
            padding: 20px 40px;
            background: #f8f9fa;
            text-align: center;
            color: #6b7280;
            font-size: 0.875rem;
            border-top: 1px solid #e5e7eb;
        }}

        .grand-total-row {{
            background: #f0f4ff !important;
            font-weight: 700;
        }}

        .grand-total-row td {{
            border-top: 2px solid #667eea;
            padding: 20px 16px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧪 Test Results Dashboard</h1>
            <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>

        <div class="summary-cards">
            <div class="card total">
                <div class="card-title">Total Tests</div>
                <div class="card-value">{grand_total['total']}</div>
                <div class="card-subtitle">Test cases executed</div>
            </div>
            <div class="card passed">
                <div class="card-title">Passed</div>
                <div class="card-value">{grand_total['passed']}</div>
                <div class="card-subtitle">{(grand_total['passed'] / grand_total['total'] * 100) if grand_total['total'] > 0 else 0:.1f}% success rate</div>
            </div>
            <div class="card failed">
                <div class="card-title">Failed</div>
                <div class="card-value">{grand_total['failed'] + grand_total['errors']}</div>
                <div class="card-subtitle">Failures + Errors</div>
            </div>
            <div class="card skipped">
                <div class="card-title">Skipped</div>
                <div class="card-value">{grand_total['skipped']}</div>
                <div class="card-subtitle">Tests not run</div>
            </div>
            <div class="card duration">
                <div class="card-title">Duration</div>
                <div class="card-value">{grand_total['duration']:.2f}s</div>
                <div class="card-subtitle">Total execution time</div>
            </div>
        </div>

        <div class="results-table">
            <h2 class="table-title">📊 Per-Directory Results</h2>
            <table>
                <thead>
                    <tr>
                        <th>Directory</th>
                        <th>Total</th>
                        <th>Passed</th>
                        <th>Failed</th>
                        <th>Skipped</th>
                        <th>Duration</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
"""

    # Add per-directory rows
    for result in results:
        dir_name = result["directory"]
        r = result["results"]
        failed_total = r["failed"] + r["errors"]

        status_class = "badge-success" if failed_total == 0 else "badge-danger"
        status_text = "✓ PASSED" if failed_total == 0 else "✗ FAILED"

        # Create link to failure details if there are failures
        if failed_total > 0 and r["failures"]:
            failure_file = f"failures_{dir_name.replace('/', '_')}.html"
            generate_failure_detail_page(result, os.path.join(output_dir, failure_file))
            failed_display = f'<a href="{failure_file}" class="clickable">{failed_total}</a>'
        else:
            failed_display = str(failed_total)

        html += f"""
                    <tr>
                        <td class="directory-name">{dir_name}</td>
                        <td>{r['total']}</td>
                        <td><span class="badge badge-success">{r['passed']}</span></td>
                        <td>{failed_display}</td>
                        <td>{r['skipped']}</td>
                        <td class="duration">{r['duration']:.2f}s</td>
                        <td><span class="badge {status_class}">{status_text}</span></td>
                    </tr>
"""

    # Add grand total row
    grand_failed = grand_total["failed"] + grand_total["errors"]
    grand_status = "✓ ALL PASSED" if grand_failed == 0 else f"✗ {grand_failed} FAILED"
    grand_status_class = "badge-success" if grand_failed == 0 else "badge-danger"

    html += f"""
                    <tr class="grand-total-row">
                        <td class="directory-name">GRAND TOTAL</td>
                        <td>{grand_total['total']}</td>
                        <td>{grand_total['passed']}</td>
                        <td>{grand_failed}</td>
                        <td>{grand_total['skipped']}</td>
                        <td class="duration">{grand_total['duration']:.2f}s</td>
                        <td><span class="badge {grand_status_class}">{grand_status}</span></td>
                    </tr>
                </tbody>
            </table>
        </div>

        <div class="footer">
            <p>Generated by test_runner_all.py • {args.run_type.upper()} Framework</p>
            <p>Command: {' '.join(sys.argv)}</p>
        </div>
    </div>
</body>
</html>
"""

    # Write dashboard HTML
    dashboard_file = os.path.join(output_dir, "index.html")
    with open(dashboard_file, "w") as f:
        f.write(html)

    print(f"\n{'=' * 70}")
    print(f"[SUCCESS] Interactive dashboard generated: {dashboard_file}")
    print(f"[INFO] Open in browser: file://{os.path.abspath(dashboard_file)}")
    print(f"{'=' * 70}\n")


def generate_failure_detail_page(result: Dict[str, Any], output_file: str):
    """Generate detailed HTML page for test failures."""
    dir_name = result["directory"]
    failures = result["results"]["failures"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Failure Details - {dir_name}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f8f9fa;
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            background: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}

        .header h1 {{
            color: #1f2937;
            margin-bottom: 10px;
        }}

        .header .breadcrumb {{
            color: #6b7280;
            font-size: 0.875rem;
        }}

        .header .breadcrumb a {{
            color: #667eea;
            text-decoration: none;
        }}

        .header .breadcrumb a:hover {{
            text-decoration: underline;
        }}

        .failure-card {{
            background: white;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            border-left: 4px solid #ef4444;
        }}

        .failure-header {{
            padding: 20px;
            background: #fef2f2;
            border-bottom: 1px solid #fee2e2;
        }}

        .failure-title {{
            font-size: 1.125rem;
            font-weight: 600;
            color: #991b1b;
            margin-bottom: 8px;
        }}

        .failure-meta {{
            font-size: 0.875rem;
            color: #6b7280;
        }}

        .failure-message {{
            padding: 20px;
            background: #fff5f5;
            border-bottom: 1px solid #fee2e2;
        }}

        .failure-message pre {{
            margin: 0;
            color: #dc2626;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.875rem;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .failure-trace {{
            padding: 20px;
            background: #fafafa;
            max-height: 400px;
            overflow-y: auto;
        }}

        .failure-trace pre {{
            margin: 0;
            color: #374151;
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.813rem;
            line-height: 1.5;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}

        .back-button {{
            display: inline-block;
            padding: 12px 24px;
            background: #667eea;
            color: white;
            text-decoration: none;
            border-radius: 6px;
            font-weight: 500;
            margin-bottom: 20px;
            transition: background 0.2s;
        }}

        .back-button:hover {{
            background: #5568d3;
        }}

        .summary {{
            background: #fee2e2;
            padding: 16px;
            border-radius: 6px;
            margin-bottom: 20px;
            color: #991b1b;
            font-weight: 500;
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="index.html" class="back-button">← Back to Dashboard</a>

        <div class="header">
            <h1>❌ Failure Details</h1>
            <p class="breadcrumb">
                <a href="index.html">Dashboard</a> / {dir_name}
            </p>
        </div>

        <div class="summary">
            Found {len(failures)} failed test(s) in {dir_name}
        </div>
"""

    for i, failure in enumerate(failures, 1):
        test_name = failure["name"]
        classname = failure["classname"]
        message = failure["message"]
        content = failure["content"]

        html += f"""
        <div class="failure-card">
            <div class="failure-header">
                <div class="failure-title">
                    Test #{i}: {test_name}
                </div>
                <div class="failure-meta">
                    Class: {classname} • Type: {failure['type'].upper()} • Time: {failure['time']}s
                </div>
            </div>

            <div class="failure-message">
                <pre>{message if message else 'No message provided'}</pre>
            </div>

            <div class="failure-trace">
                <pre>{content if content else 'No trace available'}</pre>
            </div>
        </div>
"""

    html += """
    </div>
</body>
</html>
"""

    with open(output_file, "w") as f:
        f.write(html)


def print_console_summary(results: List[Dict[str, Any]]):
    """Print formatted summary to console."""
    print("\n" + "=" * 70)
    print(" " * 25 + "TEST RESULTS SUMMARY")
    print("=" * 70)

    # Calculate grand totals
    grand_total = {
        "total": sum(r["results"]["total"] for r in results),
        "passed": sum(r["results"]["passed"] for r in results),
        "failed": sum(r["results"]["failed"] for r in results),
        "skipped": sum(r["results"]["skipped"] for r in results),
        "errors": sum(r["results"]["errors"] for r in results),
        "duration": sum(r["results"]["duration"] for r in results),
    }

    # Per-directory results
    print("\nPer-Directory Results:")
    print("-" * 70)
    print(f"{'Directory':<30} {'Total':>8} {'Passed':>8} {'Failed':>8} {'Status':>12}")
    print("-" * 70)

    for result in results:
        dir_name = Path(result["directory"]).name
        r = result["results"]
        failed_total = r["failed"] + r["errors"]
        status = "✓ PASSED" if failed_total == 0 else "✗ FAILED"

        print(f"{dir_name:<30} {r['total']:>8} {r['passed']:>8} {failed_total:>8} {status:>12}")

    # Grand total
    grand_failed = grand_total["failed"] + grand_total["errors"]
    grand_status = "✓ ALL PASSED" if grand_failed == 0 else f"✗ {grand_failed} FAILED"

    print("-" * 70)
    print(
        f"{'GRAND TOTAL':<30} {grand_total['total']:>8} {grand_total['passed']:>8} {grand_failed:>8} {grand_status:>12}"
    )
    print("=" * 70)

    # Summary stats
    success_rate = (
        (grand_total["passed"] / grand_total["total"] * 100) if grand_total["total"] > 0 else 0
    )
    print(f"\n📊 Overall Statistics:")
    print(f"   Total Tests: {grand_total['total']}")
    print(f"   Passed: {grand_total['passed']} ({success_rate:.1f}%)")
    print(f"   Failed: {grand_total['failed']}")
    print(f"   Errors: {grand_total['errors']}")
    print(f"   Skipped: {grand_total['skipped']}")
    print(f"   Duration: {grand_total['duration']:.2f}s")
    print("=" * 70 + "\n")


def run_unittest(args) -> int:
    """Run the suite via `unittest discover` and return its exit code."""
    cmd = [sys.executable, "-m", "unittest", "discover", args.input_dir]
    if args.verbosity:
        cmd.extend(["-v"])
    return _run(cmd)[0]


def run_pytest_with_dashboard(args) -> int:
    """Run pytest and generate dashboard."""
    if not _which("pytest"):
        raise RuntimeError("pytest is not installed or not on PATH")

    # Determine test directories
    if args.recursive:
        test_dirs = find_test_subdirectories(args.input_dir)
        if not test_dirs:
            print(f"[WARN] No test subdirectories found in {args.input_dir}")
            return 0
        print(f"[INFO] Found {len(test_dirs)} test subdirectories")
    else:
        test_dirs = [args.input_dir]

    results = []

    # Run tests for each directory
    for test_dir in test_dirs:
        dir_name = Path(test_dir).name if args.recursive else "tests"
        print(f"\n{'=' * 70}")
        print(f"[INFO] Running tests: {dir_name}")
        print(f"{'=' * 70}\n")

        cmd = ["pytest", test_dir]

        if args.verbosity:
            cmd.append(f"-{'v' * args.verbosity}")

        # Add JUnit XML output for dashboard
        if args.generate_dashboard:
            junit_file = os.path.join(args.dashboard_dir, f"junit_{dir_name}.xml")
            cmd.append(f"--junit-xml={junit_file}")

        # Add coverage
        if args.coverage:
            if not _which("coverage"):
                raise RuntimeError("coverage.py is required for coverage reporting")

            cmd.extend([f"--cov={args.source_dir}", "--cov-report=term"])

            for fmt in args.coverage_format:
                if fmt == "term":
                    continue
                elif fmt == "html":
                    cov_dir = (
                        f"{args.coverage_dir}_{dir_name}" if args.recursive else args.coverage_dir
                    )
                    cmd.append(f"--cov-report=html:{cov_dir}")
                elif fmt == "xml":
                    cov_dir = (
                        f"{args.coverage_dir}_{dir_name}" if args.recursive else args.coverage_dir
                    )
                    os.makedirs(cov_dir, exist_ok=True)
                    xml_path = os.path.join(cov_dir, "coverage.xml")
                    cmd.append(f"--cov-report=xml:{xml_path}")
                elif fmt == "json":
                    cov_dir = (
                        f"{args.coverage_dir}_{dir_name}" if args.recursive else args.coverage_dir
                    )
                    os.makedirs(cov_dir, exist_ok=True)
                    json_path = os.path.join(cov_dir, "coverage.json")
                    cmd.append(f"--cov-report=json:{json_path}")

        # Run tests
        returncode = _run(cmd)[0]

        # Parse results if generating dashboard
        if args.generate_dashboard:
            test_results = parse_junit_xml(junit_file)
            results.append(
                {
                    "directory": dir_name,
                    "results": test_results,
                    "returncode": returncode,
                }
            )

    # Generate dashboard
    if args.generate_dashboard and results:
        generate_html_dashboard(results, args.dashboard_dir, args)
        print_console_summary(results)

    # Return non-zero if any tests failed
    return 1 if any(r["returncode"] != 0 for r in results) else 0


def parse_args():
    """Parse this script's CLI arguments (see --help / the epilog for usage examples)."""
    parser = argparse.ArgumentParser(
        description="Production-grade test runner with interactive dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single directory with dashboard
  python test_runner_all.py --input-dir tests/metaintegrator --generate-dashboard

  # All subdirectories with dashboard and coverage
  python test_runner_all.py --input-dir tests --recursive --coverage --generate-dashboard

  # Custom dashboard location
  python test_runner_all.py --input-dir tests --recursive --generate-dashboard --dashboard-dir my_results
        """,
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing unit tests")
    parser.add_argument(
        "--run-type",
        choices=["unittest", "pytest"],
        default="pytest",
        help="Test framework to use (default: pytest)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find and run tests in all subdirectories",
    )
    parser.add_argument("--verbosity", type=int, default=0, help="Verbosity level (0-3)")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Enable coverage reporting (pytest only)",
    )
    parser.add_argument(
        "--source-dir",
        default="src",
        help="Source directory to measure coverage for (default: src)",
    )
    parser.add_argument("--coverage-dir", default="coverage", help="Directory for coverage output")
    parser.add_argument(
        "--coverage-format",
        default="html",
        help="Comma-separated coverage formats: html,xml,json,term (default: html)",
    )
    parser.add_argument(
        "--generate-dashboard",
        action="store_true",
        help="Generate interactive HTML dashboard with test results",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="test_dashboard",
        help="Directory for dashboard output (default: test_dashboard)",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point: validate args, then dispatch to run_unittest or
    run_pytest_with_dashboard."""
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    if args.coverage and not os.path.isdir(args.source_dir):
        raise FileNotFoundError(
            f"Source dir not found: {args.source_dir}. "
            f"Use --source-dir to specify the correct source directory."
        )

    args.coverage_format = [f.strip() for f in args.coverage_format.split(",")]

    if args.coverage:
        os.makedirs(args.coverage_dir, exist_ok=True)

    if args.generate_dashboard:
        os.makedirs(args.dashboard_dir, exist_ok=True)

    if args.run_type == "unittest":
        if args.coverage:
            print("[WARN] Coverage is ignored for unittest run type")
        if args.generate_dashboard:
            print("[WARN] Dashboard generation is only supported for pytest")
        return run_unittest(args)

    return run_pytest_with_dashboard(args)


if __name__ == "__main__":
    sys.exit(main())
