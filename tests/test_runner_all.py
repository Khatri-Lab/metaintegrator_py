#!/usr/bin/env python3
"""Run the metaintegrator_py test suite once and render an interactive HTML dashboard.

This project's suites (coconut, metaintegrator, preprocessing, visualization,
...) are already declared once, in ``pyproject.toml``'s
``[tool.pytest.ini_options] testpaths`` / ``addopts`` (which already
requests ``--cov=src``), so this script invokes pytest exactly once against
the whole ``tests/`` tree instead of spawning a separate pytest run per
subdirectory. Earlier versions of this script ran one pytest process per
test subdirectory, each with its own coverage report; that made it
impossible to show a single overall coverage percentage for the project
(each subdirectory's report only ever measured its own slice of ``src``).
A single run naturally produces one Cobertura XML report covering all of
``src``, so the previous "no overall coverage %" gap goes away for free.
The per-suite (per-folder) breakdown shown in the dashboard is derived
from that single JUnit XML report's ``file``/``classname`` attributes,
exactly as before.

Notable design points (mirrors the sibling ``metaintegrator-web`` project's
``infra/scripts/run_tests_dashboard.py`` in spirit and look, though the two
scripts are intentionally independent -- neither imports the other):

- **Exit code correctness.** The exit code is always exactly pytest's own
  return code, dashboard or not, so it's safe to use as a CI gate.
- **HTML escaping.** Every dynamic value (test names, failure messages,
  stack traces) goes through :func:`html.escape`, since a traceback
  containing ``<`` or ``&`` would otherwise mangle the page.
- **Coverage numbers in the dashboard.** Parses coverage.py's Cobertura
  XML at three granularities: the project-wide total, one row per
  top-level ``src/`` folder, and one row per source file, each linking to
  the full ``htmlcov/`` report for the deep dive.
- **Testability.** XML parsing and HTML rendering are pure functions with
  no subprocess/filesystem side effects, exercised in
  ``tests/test_test_runner_all.py``.

Example:
    python tests/test_runner_all.py
    python tests/test_runner_all.py --markers "unit and not slow"
    python tests/test_runner_all.py --input-dir tests/metaintegrator --dashboard-dir out
    python tests/test_runner_all.py --no-coverage -- -k test_scoring
"""

from __future__ import annotations

import argparse
import html
import subprocess
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

# ── Data model ──────────────────────────────────────────────────────────


@dataclass
class TestCase:
    """A single test case result, flattened from a JUnit XML report."""

    classname: str
    name: str
    file: str | None
    time: float
    status: str  # "passed" | "failed" | "error" | "skipped"
    message: str = ""
    content: str = ""

    @property
    def full_name(self) -> str:
        """Fully qualified test name, e.g. ``ClassName::test_x``."""
        return f"{self.classname}::{self.name}" if self.classname else self.name


@dataclass
class Totals:
    """Aggregate pass/fail/skip counts and total duration for a set of cases."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration: float = 0.0

    @classmethod
    def from_cases(cls, cases: list[TestCase]) -> "Totals":
        """Compute aggregate totals from a list of test cases."""
        t = cls()
        for c in cases:
            t.total += 1
            t.duration += c.time
            if c.status == "passed":
                t.passed += 1
            elif c.status == "failed":
                t.failed += 1
            elif c.status == "error":
                t.errors += 1
            elif c.status == "skipped":
                t.skipped += 1
        return t

    @property
    def failed_total(self) -> int:
        """Combined count of failed and errored tests."""
        return self.failed + self.errors

    @property
    def success_rate(self) -> float:
        """Percentage of tests that passed, 0-100."""
        return (self.passed / self.total * 100) if self.total else 0.0


@dataclass
class SuiteResult:
    """Test cases grouped under a single suite label, with derived totals."""

    label: str
    cases: list[TestCase] = field(default_factory=list)

    @property
    def totals(self) -> Totals:
        """Aggregate totals for this suite's cases."""
        return Totals.from_cases(self.cases)


@dataclass
class CoverageSummary:
    """Parsed project-wide line-coverage stats from a Cobertura XML report."""

    line_rate: float | None = None  # fraction 0..1
    lines_covered: int | None = None
    lines_valid: int | None = None

    @property
    def percent(self) -> float | None:
        """Coverage as a percentage 0-100, or None if unavailable."""
        return None if self.line_rate is None else round(self.line_rate * 100, 1)


@dataclass
class FolderCoverage:
    """Line coverage for one top-level ``src/`` folder (a Cobertura package)."""

    name: str
    lines_covered: int
    lines_valid: int

    @property
    def percent(self) -> float:
        """Coverage as a percentage 0-100. A folder with no measurable
        lines (e.g. only empty ``__init__.py`` files) reports 100."""
        return round(self.lines_covered / self.lines_valid * 100, 1) if self.lines_valid else 100.0


@dataclass
class FileCoverage:
    """Line coverage for a single source file (a Cobertura class entry)."""

    path: str
    lines_covered: int
    lines_valid: int

    @property
    def percent(self) -> float:
        """Coverage as a percentage 0-100. A file with no measurable lines
        (e.g. an empty ``__init__.py``) reports 100."""
        return round(self.lines_covered / self.lines_valid * 100, 1) if self.lines_valid else 100.0


# ── Parsing (pure, unit-tested) ────────────────────────────────────────


def parse_junit_xml(xml_path: Path) -> list[TestCase]:
    """Flatten a pytest JUnit-XML report into a list of :class:`TestCase`."""
    if not xml_path.exists():
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()
    testsuites = root.findall("testsuite") if root.tag == "testsuites" else [root]

    cases: list[TestCase] = []
    for suite in testsuites:
        for tc in suite.findall("testcase"):
            failure = tc.find("failure")
            error = tc.find("error")
            skipped = tc.find("skipped")

            if failure is not None:
                status, node = "failed", failure
            elif error is not None:
                status, node = "error", error
            elif skipped is not None:
                status, node = "skipped", skipped
            else:
                status, node = "passed", None

            cases.append(
                TestCase(
                    classname=tc.get("classname", ""),
                    name=tc.get("name", ""),
                    file=tc.get("file"),
                    time=float(tc.get("time", 0) or 0),
                    status=status,
                    message=(node.get("message", "") if node is not None else ""),
                    content=(node.text or "" if node is not None else ""),
                )
            )
    return cases


def parse_coverage_xml(xml_path: Path) -> CoverageSummary:
    """Parse coverage.py's Cobertura-format XML report's project-wide totals."""
    if not xml_path.exists():
        return CoverageSummary()
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return CoverageSummary()

    def _int(attr: str) -> int | None:
        val = root.get(attr)
        return int(val) if val is not None else None

    line_rate = root.get("line-rate")
    return CoverageSummary(
        line_rate=float(line_rate) if line_rate is not None else None,
        lines_covered=_int("lines-covered"),
        lines_valid=_int("lines-valid"),
    )


def _iter_cobertura_classes(root: ET.Element) -> Iterator[tuple[str, ET.Element]]:
    """Yield ``(package_name, <class> element)`` pairs from a Cobertura tree.

    Note: coverage.py's Cobertura output names each source *file* a
    "class" (a naming quirk inherited from the Cobertura format, which was
    originally designed for Java). This does not reflect real Python class
    boundaries -- it is one entry per file.
    """
    for package in root.findall("./packages/package"):
        pkg_name = package.get("name", ".")
        for cls in package.findall("./classes/class"):
            yield pkg_name, cls


def _line_hit_counts(cls_elem: ET.Element) -> tuple[int, int]:
    """Return ``(lines_covered, lines_valid)`` counted from a class's
    ``<line hits="...">`` children, independent of the (rounded) `line-rate`
    attribute coverage.py also writes."""
    lines = cls_elem.findall("./lines/line")
    valid = len(lines)
    covered = sum(1 for ln in lines if int(ln.get("hits", "0") or "0") > 0)
    return covered, valid


def parse_folder_coverage(xml_path: Path) -> list[FolderCoverage]:
    """Parse per-top-level-folder line coverage from a Cobertura XML report.

    Each Cobertura ``<package>`` already corresponds to one top-level
    folder under the measured source root (e.g. ``src/coconut``,
    ``src/metaintegrator``), so this aggregates each package's classes.
    Sorted worst-covered first, since that's the actionable order.
    """
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    totals: dict[str, list[int]] = {}
    for pkg_name, cls in _iter_cobertura_classes(root):
        covered, valid = _line_hit_counts(cls)
        acc = totals.setdefault(pkg_name, [0, 0])
        acc[0] += covered
        acc[1] += valid

    folders = [
        FolderCoverage(name=("(root)" if name == "." else name), lines_covered=c, lines_valid=v)
        for name, (c, v) in totals.items()
    ]
    return sorted(folders, key=lambda f: f.percent)


def parse_file_coverage(xml_path: Path) -> list[FileCoverage]:
    """Parse per-file line coverage from a Cobertura XML report.

    Sorted worst-covered first, since that's the most actionable order for
    a "which files most need tests" drill-down.
    """
    if not xml_path.exists():
        return []
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    files: list[FileCoverage] = []
    for _pkg_name, cls in _iter_cobertura_classes(root):
        covered, valid = _line_hit_counts(cls)
        path = cls.get("filename") or cls.get("name", "")
        files.append(FileCoverage(path=path, lines_covered=covered, lines_valid=valid))
    return sorted(files, key=lambda f: f.percent)


def suite_label_for(file_path: str) -> str:
    """Derive a stable suite label from a test file's path.

    ``tests/coconut/test_x.py``        -> ``tests/coconut``
    ``tests/metaintegrator/test_y.py`` -> ``tests/metaintegrator``
    ``tests/test_z.py``                -> ``tests``
    Anything without a "tests" segment falls back to its parent directory.
    """
    parts = Path(file_path).parts
    if "tests" in parts:
        idx = parts.index("tests")
        end = idx + 2 if idx + 2 < len(parts) else idx + 1
        return "/".join(parts[:end])
    return str(Path(file_path).parent) or "."


def group_by_suite(cases: list[TestCase]) -> list[SuiteResult]:
    """Bucket test cases into :class:`SuiteResult`s, sorted by label."""
    buckets: dict[str, list[TestCase]] = {}
    for case in cases:
        label = suite_label_for(case.file) if case.file else (case.classname or "other")
        buckets.setdefault(label, []).append(case)
    return [SuiteResult(label=label, cases=cases) for label, cases in sorted(buckets.items())]


# ── HTML rendering (pure, unit-tested) ─────────────────────────────────

_PAGE_STYLE = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; padding: 20px; }
.container { max-width: 1400px; margin: 0 auto; background: white; border-radius: 12px;
             box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden; }
.header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
          padding: 40px; text-align: center; }
.header h1 { font-size: 2.2rem; margin-bottom: 8px; }
.header .timestamp { opacity: 0.9; font-size: 0.85rem; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
         gap: 20px; padding: 32px; background: #f8f9fa; }
.card { background: white; border-radius: 8px; padding: 22px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
.card-title { font-size: 0.8rem; color: #6c757d; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px; }
.card-value { font-size: 2.2rem; font-weight: bold; }
.card-subtitle { font-size: 0.8rem; color: #6c757d; margin-top: 2px; }
.card.total .card-value { color: #667eea; }
.card.passed .card-value { color: #10b981; }
.card.failed .card-value { color: #ef4444; }
.card.skipped .card-value { color: #f59e0b; }
.card.duration .card-value { font-size: 1.9rem; color: #6c757d; }
.card.coverage .card-value { color: #0ea5e9; }
.section { padding: 0 32px 32px; }
.section h2 { font-size: 1.3rem; margin-bottom: 16px; color: #1f2937; }
table { width: 100%; border-collapse: collapse; background: white; border-radius: 8px;
        overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
th { background: #f8f9fa; padding: 14px; text-align: left; font-weight: 600; color: #374151;
     border-bottom: 2px solid #e5e7eb; }
td { padding: 14px; border-bottom: 1px solid #e5e7eb; }
tr:hover { background: #f9fafb; }
.badge { display: inline-block; padding: 3px 10px; border-radius: 10px; font-size: 0.8rem; font-weight: 600; }
.badge-success { background: #d1fae5; color: #065f46; }
.badge-danger { background: #fee2e2; color: #991b1b; }
.badge-warning { background: #fef3c7; color: #92400e; }
.clickable { color: #ef4444; cursor: pointer; text-decoration: underline; font-weight: 600; }
.grand-total-row { background: #f0f4ff !important; font-weight: 700; }
.grand-total-row td { border-top: 2px solid #667eea; padding: 16px 14px; }
.mono { font-family: Menlo, Monaco, monospace; font-size: 0.85rem; }
.footer { padding: 16px 32px; background: #f8f9fa; text-align: center; color: #6b7280;
          font-size: 0.8rem; border-top: 1px solid #e5e7eb; }
.footer a { color: #667eea; }
"""


def _card(label: str, value: str, subtitle: str, css_class: str) -> str:
    """Render a single summary stat card as an HTML snippet."""
    return (
        f'<div class="card {css_class}"><div class="card-title">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-subtitle">{html.escape(subtitle)}</div></div>'
    )


def _coverage_badge_class(percent: float) -> str:
    """Pick a badge color for a coverage percentage: red < 50, amber < 80, else green."""
    if percent < 50:
        return "badge-danger"
    if percent < 80:
        return "badge-warning"
    return "badge-success"


def render_dashboard_html(
    suites: list[SuiteResult],
    coverage: CoverageSummary | None,
    command: str,
    folder_coverage: list[FolderCoverage] | None = None,
    file_coverage: list[FileCoverage] | None = None,
    failure_page_of: dict[str, str] | None = None,
) -> str:
    """Render the top-level dashboard page.

    Args:
        suites: Per-suite (per-test-folder) results to summarize.
        coverage: Parsed project-wide coverage summary, or None if
            coverage wasn't collected.
        command: The pytest command line, shown in the footer.
        folder_coverage: Per-top-level-source-folder coverage rows.
        file_coverage: Per-file coverage rows.
        failure_page_of: Maps a suite label to the filename of its
            already-written failure detail page, for linking failed counts.
    """
    failure_page_of = failure_page_of or {}
    folder_coverage = folder_coverage or []
    file_coverage = file_coverage or []
    all_cases = [c for s in suites for c in s.cases]
    grand = Totals.from_cases(all_cases)

    cards = [
        _card("Total Tests", str(grand.total), "Test cases executed", "total"),
        _card("Passed", str(grand.passed), f"{grand.success_rate:.1f}% success rate", "passed"),
        _card("Failed", str(grand.failed_total), "Failures + errors", "failed"),
        _card("Skipped", str(grand.skipped), "Tests not run", "skipped"),
        _card("Duration", f"{grand.duration:.2f}s", "Total execution time", "duration"),
    ]
    if coverage is not None and coverage.percent is not None:
        subtitle = (
            f"{coverage.lines_covered}/{coverage.lines_valid} lines"
            if coverage.lines_covered is not None
            else "of source lines"
        )
        cards.append(_card("Coverage", f"{coverage.percent}%", subtitle, "coverage"))

    suite_rows = []
    for suite in suites:
        t = suite.totals
        status_class = "badge-success" if t.failed_total == 0 else "badge-danger"
        status_text = "PASSED" if t.failed_total == 0 else "FAILED"
        if t.failed_total and suite.label in failure_page_of:
            failed_cell = (
                f'<a href="{html.escape(failure_page_of[suite.label])}" class="clickable">'
                f"{t.failed_total}</a>"
            )
        else:
            failed_cell = str(t.failed_total)
        suite_rows.append(
            "<tr>"
            f"<td>{html.escape(suite.label)}</td>"
            f"<td>{t.total}</td>"
            f'<td><span class="badge badge-success">{t.passed}</span></td>'
            f"<td>{failed_cell}</td>"
            f"<td>{t.skipped}</td>"
            f"<td>{t.duration:.2f}s</td>"
            f'<td><span class="badge {status_class}">{status_text}</span></td>'
            "</tr>"
        )

    grand_status = "ALL PASSED" if grand.failed_total == 0 else f"{grand.failed_total} FAILED"
    grand_class = "badge-success" if grand.failed_total == 0 else "badge-danger"
    suite_rows.append(
        '<tr class="grand-total-row">'
        "<td>GRAND TOTAL</td>"
        f"<td>{grand.total}</td><td>{grand.passed}</td><td>{grand.failed_total}</td>"
        f"<td>{grand.skipped}</td><td>{grand.duration:.2f}s</td>"
        f'<td><span class="badge {grand_class}">{grand_status}</span></td>'
        "</tr>"
    )

    folder_rows = "".join(
        "<tr>"
        f"<td>{html.escape(f.name)}</td>"
        f"<td>{f.lines_covered}</td><td>{f.lines_valid}</td>"
        f'<td><span class="badge {_coverage_badge_class(f.percent)}">{f.percent}%</span></td>'
        "</tr>"
        for f in folder_coverage
    )
    folder_section = (
        f"""<div class="section">
    <h2>Coverage by Source Folder</h2>
    <table>
      <thead><tr><th>Folder</th><th>Lines Covered</th><th>Lines Valid</th><th>Coverage</th></tr></thead>
      <tbody>{folder_rows}</tbody>
    </table>
  </div>"""
        if folder_coverage
        else ""
    )

    file_rows = "".join(
        "<tr>"
        f'<td class="mono">{html.escape(f.path)}</td>'
        f"<td>{f.lines_covered}</td><td>{f.lines_valid}</td>"
        f'<td><span class="badge {_coverage_badge_class(f.percent)}">{f.percent}%</span></td>'
        "</tr>"
        for f in file_coverage
    )
    file_section = (
        f"""<div class="section">
    <h2>Coverage by File</h2>
    <table>
      <thead><tr><th>File</th><th>Lines Covered</th><th>Lines Valid</th><th>Coverage</th></tr></thead>
      <tbody>{file_rows}</tbody>
    </table>
  </div>"""
        if file_coverage
        else ""
    )

    coverage_link = (
        '<p><a href="htmlcov/index.html">Full coverage report &rarr;</a></p>'
        if coverage is not None and coverage.percent is not None
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Test Results Dashboard</title>
<style>{_PAGE_STYLE}</style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>Test Results Dashboard</h1>
    <p class="timestamp">Generated: {html.escape(datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC'))}</p>
  </div>
  <div class="cards">{''.join(cards)}</div>
  <div class="section">
    <h2>Test Results by Suite</h2>
    <table>
      <thead><tr><th>Suite</th><th>Total</th><th>Passed</th><th>Failed</th>
      <th>Skipped</th><th>Duration</th><th>Status</th></tr></thead>
      <tbody>{''.join(suite_rows)}</tbody>
    </table>
    {coverage_link}
  </div>
  {folder_section}
  {file_section}
  <div class="footer">
    <p>Generated by test_runner_all.py</p>
    <p>Command: {html.escape(command)}</p>
  </div>
</div>
</body>
</html>
"""


def render_failure_page_html(suite_label: str, cases: list[TestCase]) -> str:
    """Render a detail page listing every failed/errored case in a suite."""
    failing = [c for c in cases if c.status in ("failed", "error")]
    cards = []
    for i, c in enumerate(failing, 1):
        cards.append(
            '<div class="failure-card">'
            f"<h3>#{i}: {html.escape(c.full_name)}</h3>"
            f'<p class="meta">{html.escape(c.status.upper())} &middot; {c.time:.2f}s</p>'
            f'<pre class="message">{html.escape(c.message or "No message provided")}</pre>'
            f'<pre class="trace">{html.escape(c.content or "No trace available")}</pre>'
            "</div>"
        )
    style = """
    * { margin:0; padding:0; box-sizing:border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background:#f8f9fa; padding:20px; }
    .container { max-width:1100px; margin:0 auto; }
    a.back { display:inline-block; margin-bottom:16px; color:#667eea; }
    .failure-card { background:white; border-left:4px solid #ef4444; border-radius:8px;
                     margin-bottom:16px; padding:18px; box-shadow:0 2px 8px rgba(0,0,0,0.1); }
    .failure-card h3 { color:#991b1b; margin-bottom:6px; }
    .meta { color:#6b7280; font-size:0.85rem; margin-bottom:10px; }
    pre { white-space:pre-wrap; word-wrap:break-word; font-family:Menlo,Monaco,monospace; font-size:0.82rem; }
    pre.message { color:#dc2626; margin-bottom:10px; }
    pre.trace { color:#374151; max-height:360px; overflow-y:auto; }
    """
    return f"""<!DOCTYPE html>
<html lang="en">
<head><meta charset="UTF-8"><title>Failures: {html.escape(suite_label)}</title>
<style>{style}</style></head>
<body>
<div class="container">
  <a class="back" href="index.html">&larr; Back to dashboard</a>
  <h1>{html.escape(suite_label)}: {len(failing)} failing test(s)</h1>
  {''.join(cards)}
</div>
</body>
</html>
"""


# ── Orchestration (subprocess side effects; not unit-tested) ──────────


def build_pytest_command(
    args: argparse.Namespace, junit_path: Path, dashboard_dir: Path
) -> list[str]:
    """Build the pytest command line from parsed CLI args.

    Coverage collection (``--cov=src --cov-report=term-missing``) is
    already baked into ``pyproject.toml``'s ``addopts``, so this only adds
    the extra report formats needed for the dashboard, or explicitly
    disables coverage via pytest-cov's own ``--no-cov`` when
    ``--no-coverage`` was requested (addopts would otherwise still turn
    coverage on).
    """
    cmd = [sys.executable, "-m", "pytest", args.input_dir]
    if args.markers:
        cmd += ["-m", args.markers]
    cmd.append(f"--junit-xml={junit_path}")
    if args.coverage:
        cmd += [
            f"--cov-report=html:{dashboard_dir / 'htmlcov'}",
            f"--cov-report=xml:{dashboard_dir / 'coverage.xml'}",
        ]
    else:
        cmd.append("--no-cov")
    if args.extra:
        cmd += args.extra
    return cmd


def print_console_summary(
    suites: list[SuiteResult],
    coverage: CoverageSummary | None,
    folder_coverage: list[FolderCoverage],
) -> None:
    """Print a formatted summary of results and coverage to the console."""
    all_cases = [c for s in suites for c in s.cases]
    grand = Totals.from_cases(all_cases)

    print(f"\n{'=' * 70}\nTEST RESULTS SUMMARY\n{'=' * 70}")
    for suite in suites:
        t = suite.totals
        status = "PASSED" if t.failed_total == 0 else "FAILED"
        print(
            f"  {suite.label:<40} total={t.total:<4} passed={t.passed:<4} "
            f"failed={t.failed_total:<4} [{status}]"
        )
    print(
        f"{'-' * 70}\n  GRAND TOTAL  total={grand.total} passed={grand.passed} "
        f"failed={grand.failed_total} skipped={grand.skipped} "
        f"duration={grand.duration:.2f}s"
    )
    if coverage is not None and coverage.percent is not None:
        print(
            f"  COVERAGE     {coverage.percent}% ({coverage.lines_covered}/{coverage.lines_valid} lines)"
        )
        for folder in folder_coverage:
            print(
                f"    {folder.name:<38} {folder.percent}% ({folder.lines_covered}/{folder.lines_valid})"
            )
    print(f"{'=' * 70}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run pytest, parse results, and write the dashboard.

    Returns:
        pytest's own return code, so this script is safe to use as a CI gate.
    """
    parser = argparse.ArgumentParser(
        description="Run the metaintegrator_py test suite and generate an HTML "
        "pass/fail + coverage dashboard.",
    )
    parser.add_argument("-m", "--markers", help="pytest -m expression, e.g. 'unit and not slow'")
    parser.add_argument(
        "--coverage",
        dest="coverage",
        action="store_true",
        default=True,
        help="Collect coverage (default: on).",
    )
    parser.add_argument("--no-coverage", dest="coverage", action="store_false")
    parser.add_argument(
        "--input-dir",
        default="tests",
        help="Path passed to pytest as the test target (default: tests, the whole suite).",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="test_dashboard",
        help="Output directory for the dashboard and reports (default: test_dashboard).",
    )
    parser.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args passed through to pytest verbatim, after '--'.",
    )
    args = parser.parse_args(argv)
    if args.extra[:1] == ["--"]:
        args.extra = args.extra[1:]

    if not Path(args.input_dir).exists():
        raise FileNotFoundError(f"Input dir not found: {args.input_dir}")

    dashboard_dir = Path(args.dashboard_dir).resolve()
    dashboard_dir.mkdir(parents=True, exist_ok=True)
    junit_path = dashboard_dir / "junit.xml"

    cmd = build_pytest_command(args, junit_path, dashboard_dir)
    print(f"[INFO] Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    returncode = result.returncode

    cases = parse_junit_xml(junit_path)
    suites = group_by_suite(cases)
    coverage_xml_path = dashboard_dir / "coverage.xml"
    coverage = parse_coverage_xml(coverage_xml_path) if args.coverage else None
    folder_coverage = parse_folder_coverage(coverage_xml_path) if args.coverage else []
    file_coverage = parse_file_coverage(coverage_xml_path) if args.coverage else []

    failure_page_of: dict[str, str] = {}
    for suite in suites:
        if suite.totals.failed_total:
            fname = f"failures_{suite.label.replace('/', '_')}.html"
            (dashboard_dir / fname).write_text(render_failure_page_html(suite.label, suite.cases))
            failure_page_of[suite.label] = fname

    dashboard_html = render_dashboard_html(
        suites,
        coverage,
        command=" ".join(cmd),
        folder_coverage=folder_coverage,
        file_coverage=file_coverage,
        failure_page_of=failure_page_of,
    )
    (dashboard_dir / "index.html").write_text(dashboard_html)

    print_console_summary(suites, coverage, folder_coverage)
    print(f"[INFO] Dashboard: file://{dashboard_dir / 'index.html'}")

    if returncode != 0:
        print(
            f"[WARN] pytest exited with code {returncode}; dashboard reflects this run "
            "even though tests failed.",
            file=sys.stderr,
        )

    return returncode


if __name__ == "__main__":
    sys.exit(main())
