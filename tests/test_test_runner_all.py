"""Unit tests for tests/test_runner_all.py.

Only the pure parsing/rendering functions are exercised here: the
subprocess-invoking orchestration in main() is intentionally left to
manual/CI verification (it just shells out to the same `pytest` these
tests are running under, which would be circular).

This is a pytest test module, not a script: pytest finds and runs the
``test_*`` functions below by *collecting* this file, not by executing it
top-to-bottom. The normal way to run it is one of::

    pytest tests/test_test_runner_all.py
    pytest tests/          # collects this file along with everything else

Running it as ``python tests/test_test_runner_all.py`` used to just import
the module and exit silently, since nothing on the module's top level
triggers a test run. The ``if __name__ == "__main__"`` block at the bottom
makes that direct-invocation case work too, as a convenience for anyone
used to running a script with plain `python`, by handing off to
``pytest.main()`` explicitly.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

import test_runner_all as tra  # noqa: E402

SAMPLE_JUNIT_XML = """<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="4" failures="1" errors="1" skipped="1" time="1.230">
    <testcase classname="tests.coconut.test_coconut" name="test_a"
              file="tests/coconut/test_coconut.py" time="0.10" />
    <testcase classname="tests.coconut.test_coconut" name="test_b"
              file="tests/coconut/test_coconut.py" time="0.20">
      <failure message="assert 1 == 2">Traceback: assert 1 == 2</failure>
    </testcase>
    <testcase classname="tests.metaintegrator.test_scoring" name="test_c"
              file="tests/metaintegrator/test_scoring.py" time="0.30">
      <error message="ValueError">boom</error>
    </testcase>
    <testcase classname="tests.visualization.test_plotting" name="test_d"
              file="tests/visualization/test_plotting.py" time="0.05">
      <skipped message="not applicable" />
    </testcase>
  </testsuite>
</testsuites>
"""

SAMPLE_COVERAGE_XML = """<?xml version="1.0" ?>
<coverage line-rate="0.75" lines-covered="750" lines-valid="1000">
  <packages>
    <package name=".">
      <classes>
        <class name="__init__.py" filename="__init__.py">
          <lines/>
        </class>
      </classes>
    </package>
    <package name="coconut">
      <classes>
        <class name="coconut.py" filename="coconut/coconut.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="1"/>
            <line number="3" hits="0"/>
            <line number="4" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
    <package name="metaintegrator">
      <classes>
        <class name="scoring_core.py" filename="metaintegrator/scoring_core.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="1"/>
            <line number="3" hits="1"/>
          </lines>
        </class>
        <class name="labeling.py" filename="metaintegrator/labeling.py">
          <lines>
            <line number="1" hits="0"/>
            <line number="2" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
"""


def _write(tmp_path: Path, content: str, name: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


# ── JUnit parsing ───────────────────────────────────────────────────────


def test_parse_junit_xml_missing_file_returns_empty(tmp_path: Path) -> None:
    assert tra.parse_junit_xml(tmp_path / "nope.xml") == []


def test_parse_junit_xml_extracts_all_statuses(tmp_path: Path) -> None:
    xml_path = _write(tmp_path, SAMPLE_JUNIT_XML, "junit.xml")

    cases = tra.parse_junit_xml(xml_path)

    assert len(cases) == 4
    statuses = {c.name: c.status for c in cases}
    assert statuses == {
        "test_a": "passed",
        "test_b": "failed",
        "test_c": "error",
        "test_d": "skipped",
    }
    failed = next(c for c in cases if c.name == "test_b")
    assert failed.message == "assert 1 == 2"
    assert "assert 1 == 2" in failed.content


def test_testcase_full_name() -> None:
    case = tra.TestCase(
        classname="pkg.test_mod", name="test_x", file=None, time=0.0, status="passed"
    )
    assert case.full_name == "pkg.test_mod::test_x"


# ── Totals ──────────────────────────────────────────────────────────────


def test_totals_from_cases(tmp_path: Path) -> None:
    cases = tra.parse_junit_xml(_write(tmp_path, SAMPLE_JUNIT_XML, "junit.xml"))
    totals = tra.Totals.from_cases(cases)

    assert totals.total == 4
    assert totals.passed == 1
    assert totals.failed == 1
    assert totals.errors == 1
    assert totals.skipped == 1
    assert totals.failed_total == 2
    assert totals.success_rate == 25.0


def test_totals_success_rate_empty_is_zero() -> None:
    assert tra.Totals().success_rate == 0.0


# ── Coverage (project totals) ──────────────────────────────────────────


def test_parse_coverage_xml(tmp_path: Path) -> None:
    cov_path = _write(tmp_path, SAMPLE_COVERAGE_XML, "coverage.xml")

    cov = tra.parse_coverage_xml(cov_path)

    assert cov.percent == 75.0
    assert cov.lines_covered == 750
    assert cov.lines_valid == 1000


def test_parse_coverage_xml_missing_file(tmp_path: Path) -> None:
    cov = tra.parse_coverage_xml(tmp_path / "nope.xml")
    assert cov.line_rate is None
    assert cov.percent is None


def test_parse_coverage_xml_malformed_file_is_tolerated(tmp_path: Path) -> None:
    cov_path = _write(tmp_path, "<not-xml", "coverage.xml")
    cov = tra.parse_coverage_xml(cov_path)
    assert cov.percent is None


# ── Coverage by folder (TH-4) ────────────────────────────────────────────


def test_parse_folder_coverage(tmp_path: Path) -> None:
    cov_path = _write(tmp_path, SAMPLE_COVERAGE_XML, "coverage.xml")

    folders = tra.parse_folder_coverage(cov_path)
    by_name = {f.name: f for f in folders}

    assert by_name["coconut"].lines_covered == 2
    assert by_name["coconut"].lines_valid == 4
    assert by_name["coconut"].percent == 50.0

    assert by_name["metaintegrator"].lines_covered == 3
    assert by_name["metaintegrator"].lines_valid == 5
    assert by_name["metaintegrator"].percent == 60.0

    # The root package ("." in Cobertura) is relabeled for readability.
    assert by_name["(root)"].lines_valid == 0
    assert by_name["(root)"].percent == 100.0


def test_parse_folder_coverage_sorted_worst_first(tmp_path: Path) -> None:
    cov_path = _write(tmp_path, SAMPLE_COVERAGE_XML, "coverage.xml")
    folders = tra.parse_folder_coverage(cov_path)
    percents = [f.percent for f in folders]
    assert percents == sorted(percents)


def test_parse_folder_coverage_missing_file_returns_empty(tmp_path: Path) -> None:
    assert tra.parse_folder_coverage(tmp_path / "nope.xml") == []


# ── Coverage by file (TH-5) ──────────────────────────────────────────────


def test_parse_file_coverage(tmp_path: Path) -> None:
    cov_path = _write(tmp_path, SAMPLE_COVERAGE_XML, "coverage.xml")

    files = tra.parse_file_coverage(cov_path)
    by_path = {f.path: f for f in files}

    assert by_path["coconut/coconut.py"].percent == 50.0
    assert by_path["metaintegrator/scoring_core.py"].percent == 100.0
    assert by_path["metaintegrator/labeling.py"].percent == 0.0
    # An empty file (no measurable lines) is reported as fully covered,
    # matching coverage.py's own convention.
    assert by_path["__init__.py"].percent == 100.0


def test_parse_file_coverage_sorted_worst_first(tmp_path: Path) -> None:
    cov_path = _write(tmp_path, SAMPLE_COVERAGE_XML, "coverage.xml")
    files = tra.parse_file_coverage(cov_path)
    percents = [f.percent for f in files]
    assert percents == sorted(percents)


# ── Suite grouping ───────────────────────────────────────────────────────


def test_suite_label_for_test_subfolder() -> None:
    assert tra.suite_label_for("tests/coconut/test_coconut.py") == "tests/coconut"


def test_suite_label_for_tests_dir_with_no_subdir() -> None:
    assert tra.suite_label_for("tests/test_top_level.py") == "tests"


def test_suite_label_for_no_tests_segment_falls_back_to_parent() -> None:
    assert tra.suite_label_for("scripts/foo.py") == "scripts"


def test_group_by_suite_buckets_correctly(tmp_path: Path) -> None:
    cases = tra.parse_junit_xml(_write(tmp_path, SAMPLE_JUNIT_XML, "junit.xml"))

    suites = tra.group_by_suite(cases)
    labels = {s.label for s in suites}

    assert labels == {"tests/coconut", "tests/metaintegrator", "tests/visualization"}
    coconut_suite = next(s for s in suites if s.label == "tests/coconut")
    assert coconut_suite.totals.total == 2
    assert coconut_suite.totals.failed == 1


# ── HTML rendering ────────────────────────────────────────────────────────


def test_render_dashboard_html_escapes_hostile_content(tmp_path: Path) -> None:
    hostile_xml = SAMPLE_JUNIT_XML.replace(
        'message="assert 1 == 2"', 'message="&lt;script&gt;alert(1)&lt;/script&gt;"'
    )
    cases = tra.parse_junit_xml(_write(tmp_path, hostile_xml, "junit.xml"))
    suites = tra.group_by_suite(cases)

    out = tra.render_dashboard_html(suites, coverage=None, command="pytest")

    assert "<script>alert(1)</script>" not in out
    assert "GRAND TOTAL" in out


def test_render_dashboard_html_includes_coverage_card_when_present() -> None:
    coverage = tra.CoverageSummary(line_rate=0.75, lines_covered=750, lines_valid=1000)

    out = tra.render_dashboard_html([], coverage=coverage, command="pytest")

    assert "75.0%" in out
    assert "750/1000 lines" in out


def test_render_dashboard_html_omits_coverage_card_when_absent() -> None:
    out = tra.render_dashboard_html([], coverage=None, command="pytest")
    assert "Coverage" not in out


def test_render_dashboard_html_includes_folder_and_file_tables(tmp_path: Path) -> None:
    cov_path = _write(tmp_path, SAMPLE_COVERAGE_XML, "coverage.xml")
    folder_coverage = tra.parse_folder_coverage(cov_path)
    file_coverage = tra.parse_file_coverage(cov_path)

    out = tra.render_dashboard_html(
        [],
        coverage=None,
        command="pytest",
        folder_coverage=folder_coverage,
        file_coverage=file_coverage,
    )

    assert "Coverage by Source Folder" in out
    assert "Coverage by File" in out
    assert "coconut/coconut.py" in out


def test_render_dashboard_html_omits_folder_and_file_tables_when_absent() -> None:
    out = tra.render_dashboard_html([], coverage=None, command="pytest")
    assert "Coverage by Source Folder" not in out
    assert "Coverage by File" not in out


def test_render_failure_page_html_escapes_traceback() -> None:
    case = tra.TestCase(
        classname="pkg.test_mod",
        name="test_thing",
        file="pkg/test_mod.py",
        time=0.1,
        status="failed",
        message="<b>bold</b> assertion",
        content="Traceback (most recent call last):\n  assert a < b",
    )

    out = tra.render_failure_page_html("pkg/tests", [case])

    assert "<b>bold</b>" not in out
    assert "&lt;b&gt;bold&lt;/b&gt;" in out
    assert "assert a &lt; b" in out


def test_coverage_badge_class_thresholds() -> None:
    assert tra._coverage_badge_class(30.0) == "badge-danger"
    assert tra._coverage_badge_class(65.0) == "badge-warning"
    assert tra._coverage_badge_class(95.0) == "badge-success"


# ── pytest command construction ──────────────────────────────────────────


def test_build_pytest_command_includes_markers_and_coverage(tmp_path: Path) -> None:
    args = argparse.Namespace(
        input_dir="tests", markers="unit and not slow", coverage=True, extra=[]
    )

    cmd = tra.build_pytest_command(args, tmp_path / "junit.xml", tmp_path)

    assert cmd[3] == "tests"
    assert "-m" in cmd and "unit and not slow" in cmd
    assert any(c.startswith("--junit-xml=") for c in cmd)
    assert any(c.startswith("--cov-report=html:") for c in cmd)
    assert any(c.startswith("--cov-report=xml:") for c in cmd)
    assert "--no-cov" not in cmd


def test_build_pytest_command_no_coverage_passes_no_cov(tmp_path: Path) -> None:
    args = argparse.Namespace(input_dir="tests", markers=None, coverage=False, extra=["-k", "foo"])

    cmd = tra.build_pytest_command(args, tmp_path / "junit.xml", tmp_path)

    assert "--no-cov" in cmd
    assert not any(c.startswith("--cov-report=") for c in cmd)
    assert cmd[-2:] == ["-k", "foo"]


def test_build_pytest_command_uses_input_dir(tmp_path: Path) -> None:
    args = argparse.Namespace(input_dir="tests/coconut", markers=None, coverage=True, extra=[])

    cmd = tra.build_pytest_command(args, tmp_path / "junit.xml", tmp_path)

    assert cmd[3] == "tests/coconut"


# ── Direct-invocation convenience ────────────────────────────────────────
#
# This file is a pytest test module: pytest runs it by *collecting* the
# test_* functions above, not by executing top-level code. Without this
# block, `python tests/test_test_runner_all.py` would just import the
# module and exit -- no tests would run, and no output would appear. This
# hands that case off to pytest itself so the direct-invocation habit
# people have for `test_runner_all.py` (a real script) also works here.
if __name__ == "__main__":
    sys.exit(pytest.main([__file__, *sys.argv[1:]]))
