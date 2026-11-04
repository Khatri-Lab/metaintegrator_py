"""Keep plan IDs, session folders and personal paths out of committed files.

Why this test exists:

* A plan or session ID such as a numbered roadmap item, an observation
  number or a review-comment number means nothing to a reader of the
  repository. The plan it came from is not committed.
* A link into ``docs/planning/`` or a folder of AI session notes points at
  files that other checkouts do not have.
* A personal home-folder path leaks a user name, and it only works on one
  machine.

What it scans:

* Files that git tracks. If git is not available, it walks the tree and
  skips scratch folders, caches and virtual environments.
* Only code and doc files: ``.py``, ``.R``, ``.md``, ``.mmd``, ``.yml``,
  ``.yaml``, ``.toml``, ``.sh``, ``.ini``, ``.cfg``, ``Makefile`` and
  ``Dockerfile``. Data files (``.csv``, ``.txt``, ``.rda`` and so on) are
  never scanned, because gene symbols and sample names can look like IDs.
* This file is skipped. It has to spell out the patterns it looks for.

What it reports: every hit, with the file, the line number, the rule and
the line itself, so one run shows everything to fix.

How to fix a hit: say what the thing is called, not its number. Say what
happened, not when it was planned. Link only to committed files, such as
an ADR, the user guide or the changelog.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

# This file lives in ``tests/``, one level below the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]

# Files that exist in a host checkout but not inside the Docker image
# (``.dockerignore`` leaves out ``.github/`` and ``docs/``). If they are
# missing, the repository is not here, so the scan is skipped.
_REPO_MARKERS = ("pyproject.toml", "mkdocs.yml", ".github")

_REPO_PRESENT = all((REPO_ROOT / marker).exists() for marker in _REPO_MARKERS)
_REPO_MISSING_REASON = (
    "The repository checkout (pyproject.toml, mkdocs.yml, .github) is not present "
    "under REPO_ROOT. This scan checks committed files, which are not "
    "available inside a running container. Run it on the host with "
    "`pytest tests/test_repository_hygiene.py`."
)

# File types that are code or documentation. Never data files.
SCANNED_SUFFIXES = frozenset(
    {".py", ".r", ".md", ".mmd", ".yml", ".yaml", ".toml", ".sh", ".ini", ".cfg"}
)
SCANNED_NAMES = frozenset({"Makefile", "Dockerfile"})

# Folders the fallback tree walk skips. With git, only tracked files are
# listed, so none of this matters. Without git, these are scratch folders,
# caches and environments that a checkout can contain next to the code.
_SKIPPED_DIR_NAMES = frozenset(
    {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
        "site",
        ".idea",
    }
)
_SKIPPED_DIR_PREFIXES = ("_temp", "_docs_")
_SKIPPED_DIR_SUFFIXES = (".egg-info",)

_MAX_LINE_CHARS_IN_REPORT = 140


@dataclass(frozen=True)
class Rule:
    """One thing that must not appear in committed code or docs.

    Attributes:
        name: Short label shown next to each hit.
        pattern: Regular expression that finds the problem.
        advice: One sentence that says what to write instead.
    """

    name: str
    pattern: re.Pattern[str]
    advice: str


RULES: tuple[Rule, ...] = (
    Rule(
        name="plan or session ID",
        pattern=re.compile(r"\b(?:R[12]-\d+[a-z]?|OBS-\d+|RESP-\d+|COMMENT-\d+)\b"),
        advice="Use the feature's name, or describe what was observed.",
    ),
    Rule(
        name="session notes folder",
        pattern=re.compile(r"_temp_claude|temp4claude"),
        advice="Session notes are not committed. Remove the path and keep the reasoning.",
    ),
    Rule(
        name="planning docs link",
        pattern=re.compile(r"docs/planning/"),
        advice="Link to a committed ADR, the user guide or the changelog.",
    ),
    Rule(
        name="personal path",
        # ``/Users/you/`` is the placeholder used in help text. ``appuser``
        # is the container's service account and ``runner`` is the CI user.
        pattern=re.compile(
            r"/Users/(?!you(?![\w.-]))[A-Za-z0-9_.-]+"
            r"|/home/(?!(?:appuser|runner)(?![\w.-]))[A-Za-z0-9_.-]+"
        ),
        advice="Use a relative path or a placeholder such as /Users/you/.",
    ),
)


@dataclass(frozen=True)
class Violation:
    """One hit found in one line of one file.

    Attributes:
        path: File path relative to the repository root, with ``/`` separators.
        line: 1-based line number.
        rule: The :attr:`Rule.name` that matched.
        match: The text that matched.
        context: The line the match is on, shortened for the report.
    """

    path: str
    line: int
    rule: str
    match: str
    context: str


def is_scanned_file(relative_path: str) -> bool:
    """Say whether a file is code or documentation that this test scans.

    Args:
        relative_path: Path relative to the repository root.

    Returns:
        ``True`` for the code and doc file types listed in the module
        docstring, ``False`` for everything else (data files included).
    """
    path = Path(relative_path)
    if path.name in SCANNED_NAMES or path.name.startswith("Dockerfile."):
        return True
    return path.suffix.lower() in SCANNED_SUFFIXES


def find_violations(text: str, rules: Sequence[Rule] = RULES) -> list[tuple[int, Rule, str]]:
    """Find every rule hit in a block of text.

    Args:
        text: File content.
        rules: Rules to apply. Defaults to :data:`RULES`.

    Returns:
        One ``(line_number, rule, matched_text)`` tuple per hit, in line order.
    """
    hits: list[tuple[int, Rule, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule in rules:
            for match in rule.pattern.finditer(line):
                hits.append((number, rule, match.group(0)))
    return hits


def _git_tracked_files(root: Path) -> list[str] | None:
    """List the files git tracks under ``root``.

    Args:
        root: Repository root.

    Returns:
        Relative paths, or ``None`` when git is missing, ``root`` is not a
        git checkout, or the command fails for any reason.
    """
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(
            [git, "-C", str(root), "ls-files", "-z"],
            capture_output=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    names = result.stdout.decode("utf-8", errors="replace").split("\0")
    return [name for name in names if name]


def _walk_files(root: Path) -> list[str]:
    """List files under ``root`` without git, skipping scratch folders.

    Args:
        root: Repository root.

    Returns:
        Relative paths with ``/`` separators.
    """
    found: list[str] = []
    for current, dir_names, file_names in os.walk(root):
        dir_names[:] = [
            name
            for name in dir_names
            if name not in _SKIPPED_DIR_NAMES
            and not name.startswith(_SKIPPED_DIR_PREFIXES)
            and not name.endswith(_SKIPPED_DIR_SUFFIXES)
        ]
        folder = Path(current)
        for file_name in file_names:
            found.append((folder / file_name).relative_to(root).as_posix())
    return found


def list_scannable_files(root: Path) -> list[str]:
    """List the committed code and doc files to scan.

    Uses ``git ls-files`` when it works, and a tree walk with the same file
    type filter when it does not.

    Args:
        root: Repository root.

    Returns:
        Sorted relative paths of files to scan.
    """
    names = _git_tracked_files(root)
    if names is None:
        names = _walk_files(root)
    return sorted(name for name in names if is_scanned_file(name))


def _self_relative_path() -> str | None:
    """Return this file's path relative to the repository root, if inside it."""
    try:
        return Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return None


def scan_repository(root: Path, skip: Sequence[str] = ()) -> list[Violation]:
    """Scan the committed code and doc files under ``root``.

    Args:
        root: Repository root.
        skip: Relative paths to leave out, such as this test file.

    Returns:
        Every violation found, ordered by file and line.
    """
    violations: list[Violation] = []
    for relative in list_scannable_files(root):
        if relative in skip:
            continue
        path = root / relative
        if not path.is_file():
            continue  # tracked but deleted in the working tree
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        for number, rule, matched in find_violations(text):
            context = lines[number - 1].strip()
            if len(context) > _MAX_LINE_CHARS_IN_REPORT:
                context = context[: _MAX_LINE_CHARS_IN_REPORT - 3] + "..."
            violations.append(Violation(relative, number, rule.name, matched, context))
    return violations


def format_violations(violations: Sequence[Violation]) -> str:
    """Build the failure message: one line per hit, then advice per rule.

    Args:
        violations: Hits from :func:`scan_repository`.

    Returns:
        A multi-line message with file and line for every hit.
    """
    lines = [
        f"{len(violations)} repository hygiene problem(s) in committed files:",
        "",
    ]
    for item in violations:
        lines.append(f"  {item.path}:{item.line}: [{item.rule}] {item.match!r}")
        lines.append(f"      {item.context}")
    lines.append("")
    seen: set[str] = set()
    for rule in RULES:
        if any(item.rule == rule.name for item in violations) and rule.name not in seen:
            seen.add(rule.name)
            lines.append(f"{rule.name}: {rule.advice}")
    return "\n".join(lines)


# ── The scan of this repository ────────────────────────────────────────────


@pytest.mark.skipif(not _REPO_PRESENT, reason=_REPO_MISSING_REASON)
def test_committed_files_have_no_plan_ids_session_paths_or_personal_paths() -> None:
    skip = [path for path in (_self_relative_path(),) if path]
    violations = scan_repository(REPO_ROOT, skip=skip)
    assert not violations, format_violations(violations)


# ── The rules themselves ───────────────────────────────────────────────────


def _names_hit(text: str) -> list[str]:
    return [rule.name for _, rule, _ in find_violations(text)]


class TestPlanAndSessionIds:
    @pytest.mark.parametrize(
        "text",
        [
            "# R1-10: sample labels",
            "see R2-3 for details",
            "the R1-12a follow-up",
            "case that hid OBS-01",
            "(RESP-01, option 3)",
            "left COMMENT-3 here",
        ],
    )
    def test_plan_ids_are_flagged(self, text: str) -> None:
        assert _names_hit(text) == ["plan or session ID"]

    @pytest.mark.parametrize(
        "text",
        [
            "REG-12 is a row in the parity register",
            "ADR 0021, Decision 3",
            "migration 016",
            "gene symbol XR1-10 in a table",
            "R10-1 is not a plan ID",
            "R3-1 is not a plan ID either",
        ],
    )
    def test_committed_identifiers_are_allowed(self, text: str) -> None:
        assert _names_hit(text) == []


class TestSessionFoldersAndPlanningLinks:
    def test_session_notes_folder_is_flagged(self) -> None:
        assert _names_hit("see _temp_claude/notes.md") == ["session notes folder"]

    def test_local_project_folder_name_is_flagged(self) -> None:
        assert _names_hit("cd temp4claude_webApp_v01") == ["session notes folder"]

    def test_planning_docs_link_is_flagged(self) -> None:
        assert _names_hit("[design](docs/planning/x.md)") == ["planning docs link"]

    def test_adr_link_is_allowed(self) -> None:
        assert _names_hit("[ADR](docs/adr/0021-x.md)") == []


class TestPersonalPaths:
    @pytest.mark.parametrize(
        "text",
        [
            "/Users/someone/project",
            "cd /Users/someone",
            "/home/someone/.config",
            "C:/Users/someone/Desktop",
        ],
    )
    def test_personal_paths_are_flagged(self, text: str) -> None:
        assert _names_hit(text) == ["personal path"]

    @pytest.mark.parametrize(
        "text",
        [
            "make run DIR=/Users/you/data",
            "/home/appuser/.config/matplotlib",
            "/home/runner/work/repo",
            "/Users/<name>/ is a placeholder",
            "/home/<name>/ is a placeholder",
        ],
    )
    def test_placeholders_and_service_accounts_are_allowed(self, text: str) -> None:
        assert _names_hit(text) == []

    def test_a_longer_name_that_starts_with_you_is_flagged(self) -> None:
        assert _names_hit("/Users/youngster/data") == ["personal path"]


class TestFindViolations:
    def test_reports_line_numbers_and_every_hit(self) -> None:
        text = "clean line\nR1-1 and R1-2 here\n\n/Users/someone/x\n"
        hits = [(line, matched) for line, _, matched in find_violations(text)]
        assert hits == [(2, "R1-1"), (2, "R1-2"), (4, "/Users/someone")]


# ── Which files are scanned ────────────────────────────────────────────────


class TestIsScannedFile:
    @pytest.mark.parametrize(
        "path",
        [
            "backend/src/config.py",
            "backend/src/api/v1/r_scripts/plot_heatmap.R",
            "docs/adr/0021-x.md",
            "docs/diagrams/data-model.mmd",
            ".github/workflows/ci.yml",
            "config/settings.yaml",
            "pyproject.toml",
            "infra/run.sh",
            "pytest.ini",
            "setup.cfg",
            "Makefile",
            "backend/Dockerfile",
            "docker/Dockerfile.dev",
        ],
    )
    def test_code_and_doc_files_are_scanned(self, path: str) -> None:
        assert is_scanned_file(path)

    @pytest.mark.parametrize(
        "path",
        [
            "data/reference/ensembl_gene_symbols.txt",
            "data/immunoStatesMatrix.csv",
            "data/table.rda",
            "tutorials/notebook.ipynb",
            "docs/assets/extra.css",
            "logo.png",
            "data/old.csv_OBSOLETE",
        ],
    )
    def test_data_files_are_never_scanned(self, path: str) -> None:
        assert not is_scanned_file(path)


# ── Listing and scanning files ─────────────────────────────────────────────


def _write(root: Path, relative: str, text: str) -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


class TestTreeWalkFallback:
    """Without git, the walk lists the same kind of files and skips scratch."""

    def test_walk_skips_scratch_folders_and_data_files(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/app.py", "x = 1\n")
        _write(tmp_path, "docs/guide.md", "text\n")
        _write(tmp_path, "data/genes.txt", "R1-10\n")
        _write(tmp_path, "_temp_claude/notes.md", "R1-10\n")
        _write(tmp_path, "_docs_temp/old.md", "R1-10\n")
        _write(tmp_path, "src/__pycache__/app.py", "R1-10\n")
        _write(tmp_path, ".venv/lib/mod.py", "R1-10\n")
        _write(tmp_path, "pkg.egg-info/PKG.md", "R1-10\n")
        found = [name for name in _walk_files(tmp_path) if is_scanned_file(name)]
        assert sorted(found) == ["docs/guide.md", "src/app.py"]

    def test_a_folder_that_only_starts_with_underscore_is_still_scanned(
        self, tmp_path: Path
    ) -> None:
        _write(tmp_path, "_archived_page_stubs/stub.py", "x = 1\n")
        assert "_archived_page_stubs/stub.py" in list_scannable_files(tmp_path)

    def test_missing_git_falls_back_to_the_walk(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(shutil, "which", lambda _name: None)
        _write(tmp_path, "src/app.py", "x = 1\n")
        assert _git_tracked_files(tmp_path) is None
        assert list_scannable_files(tmp_path) == ["src/app.py"]

    def test_scan_finds_a_hit_without_git(self, tmp_path: Path) -> None:
        _write(tmp_path, "src/app.py", "# fine\n# R1-10 explains this\n")
        violations = scan_repository(tmp_path)
        assert [(v.path, v.line, v.rule) for v in violations] == [
            ("src/app.py", 2, "plan or session ID")
        ]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
class TestGitListing:
    def _init_repo(self, root: Path) -> None:
        subprocess.run(["git", "-C", str(root), "init", "-q"], check=True)

    def test_only_tracked_files_are_listed(self, tmp_path: Path) -> None:
        self._init_repo(tmp_path)
        _write(tmp_path, "tracked.py", "# R1-10\n")
        _write(tmp_path, "scratch.py", "# R1-10\n")
        subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.py"], check=True)
        assert list_scannable_files(tmp_path) == ["tracked.py"]

    def test_a_tracked_file_deleted_from_disk_is_skipped(self, tmp_path: Path) -> None:
        self._init_repo(tmp_path)
        _write(tmp_path, "gone.py", "# R1-10\n")
        subprocess.run(["git", "-C", str(tmp_path), "add", "gone.py"], check=True)
        (tmp_path / "gone.py").unlink()
        assert scan_repository(tmp_path) == []


class TestScanRepository:
    def test_skip_leaves_the_named_file_out(self, tmp_path: Path) -> None:
        _write(tmp_path, "tests/test_hygiene.py", "PATTERN = 'R1-10'\n")
        _write(tmp_path, "src/app.py", "x = 1\n")
        assert scan_repository(tmp_path, skip=["tests/test_hygiene.py"]) == []

    def test_every_hit_is_reported_with_file_and_line(self, tmp_path: Path) -> None:
        _write(tmp_path, "a.py", "ok\n# OBS-01 here\n")
        _write(tmp_path, "b.md", "see docs/planning/x.md\nand /Users/someone/x\n")
        violations = scan_repository(tmp_path)
        assert [(v.path, v.line, v.rule) for v in violations] == [
            ("a.py", 2, "plan or session ID"),
            ("b.md", 1, "planning docs link"),
            ("b.md", 2, "personal path"),
        ]


class TestFormatViolations:
    def test_message_lists_file_line_and_advice(self) -> None:
        message = format_violations(
            [Violation("src/app.py", 12, "plan or session ID", "R1-10", "# R1-10: x")]
        )
        assert "1 repository hygiene problem(s)" in message
        assert "src/app.py:12: [plan or session ID] 'R1-10'" in message
        assert "# R1-10: x" in message
        assert "Use the feature's name" in message
        assert "personal path" not in message
