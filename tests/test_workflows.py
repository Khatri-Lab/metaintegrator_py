"""Check the GitHub Actions files against the rules that keep CI safe.

Why this test exists:

* A floating runner label such as ``ubuntu-latest`` changes under the
  project without a commit, so a green run can turn red for no reason.
* A workflow that can write to Pages, or that has wide default
  permissions, gives every step in it more power than it needs.
* A job without ``timeout-minutes`` can hang for six hours and use up the
  monthly minutes.
* An action pinned to ``@main`` or ``@master`` runs whatever the author
  pushed last.

What it checks:

* Every file in ``.github/workflows/`` follows the common rules in
  :func:`find_problems`.
* ``ci.yml`` runs the format checks and the fast test selection, and keeps
  the job name ``test``.
* ``deploy-docs.yml`` builds on pull requests and deploys only when the
  variable ``DEPLOY_DOCS`` is ``true`` on ``main``.
* ``dependabot.yml`` tracks GitHub Actions only, once a month.
* ``requirements-dev.txt`` pins ``black`` and ``isort`` to exact versions.

What it does not check: the syntax of expressions or the inputs of an
action. Run actionlint for that, for example:
``docker run --rm -v "$(pwd):/repo" -w /repo rhysd/actionlint:latest``.

Where it runs: the Docker image leaves out ``.github/``, so the checks on
the real files are skipped there. They run on a full checkout, for example
in a container with the repository mounted. The tests of the rules
themselves run everywhere.

PyYAML reads the key ``on`` as the boolean ``True``. :func:`parse_yaml_mapping`
turns it back into the string ``"on"``, so the tests can write
``workflow["on"]``.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

# This file lives in ``tests/``, one level below the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
DEPENDABOT_FILE = REPO_ROOT / ".github" / "dependabot.yml"
DEV_REQUIREMENTS_FILE = REPO_ROOT / "requirements-dev.txt"

CI_WORKFLOW = "ci.yml"
DOCS_WORKFLOW = "deploy-docs.yml"

# The one job that may hold ``pages: write``.
DEPLOY_JOB = "deploy"
FLOATING_RUNNER = "ubuntu-latest"
_MOVING_REF = re.compile(r"@(?:main|master)$")

# Both parts must appear in the condition of every Pages step.
DEPLOY_CONDITION_PARTS = ("vars.DEPLOY_DOCS == 'true'", "github.ref == 'refs/heads/main'")
CANCEL_ONLY_PULL_REQUESTS = "${{ github.event_name == 'pull_request' }}"

# Tools whose version decides whether ``ci.yml`` passes.
PINNED_TOOLS = ("black", "isort")
_EXACT_PIN = re.compile(r"^==\d+(?:\.\d+){2}$")

_requires_workflows = pytest.mark.skipif(
    not WORKFLOWS_DIR.is_dir(),
    reason=(
        ".github/workflows is not present under REPO_ROOT. The Docker image leaves it out. "
        "Run this test on a full checkout, for example with the repository mounted."
    ),
)
_requires_dependabot = pytest.mark.skipif(
    not DEPENDABOT_FILE.is_file(), reason=".github/dependabot.yml is not present."
)
_requires_dev_requirements = pytest.mark.skipif(
    not DEV_REQUIREMENTS_FILE.is_file(), reason="requirements-dev.txt is not present."
)


# -- Reading the files --------------------------------------------------------


def parse_yaml_mapping(text: str) -> dict[str, Any]:
    """Parse YAML text into a mapping and repair the ``on`` key.

    PyYAML follows YAML 1.1, so an unquoted ``on:`` key becomes the boolean
    ``True``. Workflow files always write it unquoted. This function turns
    that key back into the string ``"on"``.

    Args:
        text: YAML source.

    Returns:
        The top-level mapping, with string keys.

    Raises:
        ValueError: If the document is not a mapping.
    """
    loaded = yaml.safe_load(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a YAML mapping, got {type(loaded).__name__}")
    return {"on" if key is True else str(key): value for key, value in loaded.items()}


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Read and parse a YAML file with :func:`parse_yaml_mapping`.

    Args:
        path: File to read.

    Returns:
        The top-level mapping.
    """
    return parse_yaml_mapping(path.read_text(encoding="utf-8"))


def parse_requirement_pins(text: str) -> dict[str, str]:
    """Map each requirement name to its version specifier.

    Comments, blank lines and option lines such as ``-r other.txt`` are
    skipped. Names follow PEP 503: lower case, with runs of ``-``, ``_`` and
    ``.`` treated as one hyphen.

    Args:
        text: Content of a requirements file.

    Returns:
        A mapping such as ``{"black": "==26.10.1", "pytest": ">=6.0.0"}``.
        A requirement without a specifier maps to an empty string.
    """
    pins: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:\[[^\]]*\])?\s*(.*)$", line)
        if match:
            name = re.sub(r"[-_.]+", "-", match.group(1)).lower()
            pins[name] = match.group(2).strip()
    return pins


# -- The common rules ---------------------------------------------------------


def _jobs(workflow: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return the jobs of a workflow as ``{job_id: job}``, or an empty dict."""
    jobs = workflow.get("jobs")
    if not isinstance(jobs, Mapping):
        return {}
    return {str(name): job for name, job in jobs.items() if isinstance(job, Mapping)}


def _steps(job: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the steps of a job that are mappings."""
    steps = job.get("steps")
    if not isinstance(steps, list):
        return []
    return [step for step in steps if isinstance(step, Mapping)]


def _iter_strings(node: Any) -> Iterator[str]:
    """Yield every string in a parsed YAML tree, keys and values alike."""
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for key, value in node.items():
            yield from _iter_strings(key)
            yield from _iter_strings(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_strings(item)


def _grants_pages_write(permissions: Any) -> bool:
    """Say whether a ``permissions`` value allows writing to Pages."""
    if isinstance(permissions, str):
        return permissions == "write-all"
    if isinstance(permissions, Mapping):
        return permissions.get("pages") == "write"
    return False


def _action_references(job: Mapping[str, Any]) -> list[str]:
    """List every ``uses`` value of a job: the job itself and its steps."""
    references = [job["uses"]] if isinstance(job.get("uses"), str) else []
    references.extend(step["uses"] for step in _steps(job) if isinstance(step.get("uses"), str))
    return references


def find_problems(workflow: Mapping[str, Any]) -> list[str]:
    """Apply the common workflow rules and list every violation.

    The rules:

    1. No job runs on ``ubuntu-latest``.
    2. The top-level ``permissions`` are exactly ``contents: read``.
    3. Only the ``deploy`` job may hold ``pages: write``.
    4. Every job has ``timeout-minutes``. A job that calls a reusable
       workflow with ``uses`` is exempt, because GitHub does not allow the
       key there.
    5. No action or reusable workflow is referenced as ``@main`` or
       ``@master``.

    Args:
        workflow: A parsed workflow, as returned by :func:`parse_yaml_mapping`.

    Returns:
        One message per violation. An empty list means the workflow is fine.
    """
    problems: list[str] = []
    jobs = _jobs(workflow)
    if not jobs:
        problems.append("defines no jobs")

    if any(FLOATING_RUNNER in text for text in _iter_strings(workflow)):
        problems.append(f"uses the floating runner label {FLOATING_RUNNER!r}; pin a version")

    permissions = workflow.get("permissions")
    if permissions != {"contents": "read"}:
        problems.append(
            f"top-level permissions must be exactly {{'contents': 'read'}}, got {permissions!r}"
        )

    for job_id, job in jobs.items():
        if job_id != DEPLOY_JOB and _grants_pages_write(job.get("permissions")):
            problems.append(f"job {job_id!r} may write to Pages; only {DEPLOY_JOB!r} may")
        if "timeout-minutes" not in job and "uses" not in job:
            problems.append(f"job {job_id!r} has no timeout-minutes")
        for reference in _action_references(job):
            if _MOVING_REF.search(reference.strip()):
                problems.append(f"job {job_id!r} uses {reference!r}, which follows a moving branch")
    return problems


# -- The rules themselves -----------------------------------------------------


def make_workflow() -> dict[str, Any]:
    """Build a small workflow that follows every common rule."""
    return {
        "name": "Example",
        "on": {"push": {"branches": ["main"]}},
        "permissions": {"contents": "read"},
        "jobs": {
            "build": {
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": 10,
                "steps": [{"uses": "actions/checkout@v7"}, {"run": "make build"}],
            },
            DEPLOY_JOB: {
                "needs": "build",
                "runs-on": "ubuntu-24.04",
                "timeout-minutes": 5,
                "permissions": {"pages": "write", "id-token": "write"},
                "steps": [{"uses": "actions/deploy-pages@v4"}],
            },
        },
    }


class TestParseYamlMapping:
    """The ``on`` key must come back as a string."""

    def test_unquoted_on_key_becomes_a_string(self) -> None:
        workflow = parse_yaml_mapping("on:\n  push:\n    branches: [main]\njobs: {}\n")
        assert workflow["on"] == {"push": {"branches": ["main"]}}
        assert True not in workflow

    def test_quoted_on_key_is_unchanged(self) -> None:
        assert parse_yaml_mapping('"on": [push]\n') == {"on": ["push"]}

    def test_other_keys_stay_as_they_are(self) -> None:
        assert parse_yaml_mapping("name: CI\njobs: {}\n") == {"name": "CI", "jobs": {}}

    @pytest.mark.parametrize("text", ["- a\n- b\n", "just text\n", ""])
    def test_a_document_that_is_not_a_mapping_is_rejected(self, text: str) -> None:
        with pytest.raises(ValueError, match="expected a YAML mapping"):
            parse_yaml_mapping(text)


class TestFindProblems:
    """Each rule flags the bad case and leaves the good case alone."""

    def test_a_clean_workflow_has_no_problems(self) -> None:
        assert find_problems(make_workflow()) == []

    def test_no_jobs_is_a_problem(self) -> None:
        workflow = make_workflow()
        workflow["jobs"] = {}
        assert find_problems(workflow) == ["defines no jobs"]

    # Rule 1: runner label.

    def test_floating_runner_is_flagged(self) -> None:
        workflow = make_workflow()
        workflow["jobs"]["build"]["runs-on"] = FLOATING_RUNNER
        assert len(find_problems(workflow)) == 1
        assert "floating runner" in find_problems(workflow)[0]

    def test_floating_runner_inside_a_matrix_is_flagged(self) -> None:
        workflow = make_workflow()
        workflow["jobs"]["build"]["strategy"] = {"matrix": {"os": [FLOATING_RUNNER, "macos-15"]}}
        assert any("floating runner" in problem for problem in find_problems(workflow))

    # Rule 2: default permissions.

    @pytest.mark.parametrize(
        "permissions",
        [None, {}, {"contents": "write"}, {"contents": "read", "pages": "write"}, "write-all"],
    )
    def test_top_level_permissions_must_be_contents_read(self, permissions: Any) -> None:
        workflow = make_workflow()
        if permissions is None:
            del workflow["permissions"]
        else:
            workflow["permissions"] = permissions
        problems = find_problems(workflow)
        assert len(problems) == 1
        assert "top-level permissions" in problems[0]

    # Rule 3: Pages write.

    def test_pages_write_is_allowed_in_the_deploy_job(self) -> None:
        assert find_problems(make_workflow()) == []

    @pytest.mark.parametrize("permissions", [{"pages": "write"}, "write-all"])
    def test_pages_write_in_another_job_is_flagged(self, permissions: Any) -> None:
        workflow = make_workflow()
        workflow["jobs"]["build"]["permissions"] = permissions
        problems = find_problems(workflow)
        assert len(problems) == 1
        assert "'build'" in problems[0]
        assert "Pages" in problems[0]

    def test_pages_read_in_another_job_is_fine(self) -> None:
        workflow = make_workflow()
        workflow["jobs"]["build"]["permissions"] = {"pages": "read"}
        assert find_problems(workflow) == []

    # Rule 4: timeouts.

    def test_a_job_without_a_timeout_is_flagged(self) -> None:
        workflow = make_workflow()
        del workflow["jobs"]["build"]["timeout-minutes"]
        assert find_problems(workflow) == ["job 'build' has no timeout-minutes"]

    def test_a_reusable_workflow_job_needs_no_timeout(self) -> None:
        workflow = make_workflow()
        workflow["jobs"]["shared"] = {"uses": "./.github/workflows/shared.yml"}
        assert find_problems(workflow) == []

    # Rule 5: moving references.

    @pytest.mark.parametrize("reference", ["actions/checkout@main", "actions/checkout@master"])
    def test_an_action_on_a_moving_branch_is_flagged(self, reference: str) -> None:
        workflow = make_workflow()
        workflow["jobs"]["build"]["steps"].append({"uses": reference})
        problems = find_problems(workflow)
        assert len(problems) == 1
        assert reference in problems[0]

    def test_a_reusable_workflow_on_a_moving_branch_is_flagged(self) -> None:
        workflow = make_workflow()
        workflow["jobs"]["shared"] = {"uses": "org/repo/.github/workflows/x.yml@main"}
        assert any("moving branch" in problem for problem in find_problems(workflow))

    @pytest.mark.parametrize(
        "reference",
        [
            "actions/checkout@v7",
            "actions/checkout@v7.1.0",
            "actions/checkout@" + "a" * 40,
            "owner/mainline-tool@v1",
            "docker://alpine:3.20",
        ],
    )
    def test_pinned_references_are_fine(self, reference: str) -> None:
        workflow = make_workflow()
        workflow["jobs"]["build"]["steps"].append({"uses": reference})
        assert find_problems(workflow) == []

    def test_every_violation_is_reported_at_once(self) -> None:
        workflow = make_workflow()
        workflow["jobs"]["build"]["runs-on"] = FLOATING_RUNNER
        del workflow["jobs"]["build"]["timeout-minutes"]
        workflow["permissions"] = {"contents": "write"}
        assert len(find_problems(workflow)) == 3


class TestParseRequirementPins:
    def test_exact_and_range_specifiers(self) -> None:
        text = "black==26.10.1\npytest>=6.0.0\nnumpy\n"
        expected = {"black": "==26.10.1", "pytest": ">=6.0.0", "numpy": ""}
        assert parse_requirement_pins(text) == expected

    def test_comments_blank_lines_and_options_are_skipped(self) -> None:
        text = "# header\n\n-r base.txt\nisort==9.0.2  # pinned\n"
        assert parse_requirement_pins(text) == {"isort": "==9.0.2"}

    def test_names_follow_pep_503(self) -> None:
        assert parse_requirement_pins("Pytest_Cov>=2.12\n") == {"pytest-cov": ">=2.12"}

    def test_extras_are_not_part_of_the_name(self) -> None:
        assert parse_requirement_pins("black[d]==26.10.1\n") == {"black": "==26.10.1"}


# -- The workflow files in this repository -----------------------------------

_WORKFLOW_FILES = (
    sorted([*WORKFLOWS_DIR.glob("*.yml"), *WORKFLOWS_DIR.glob("*.yaml")])
    if WORKFLOWS_DIR.is_dir()
    else []
)


@_requires_workflows
class TestWorkflowFiles:
    """The common rules, applied to every workflow file."""

    def test_the_expected_workflows_exist(self) -> None:
        names = {path.name for path in _WORKFLOW_FILES}
        assert {CI_WORKFLOW, DOCS_WORKFLOW} <= names

    @pytest.mark.parametrize("path", _WORKFLOW_FILES, ids=lambda path: path.name)
    def test_workflow_follows_the_common_rules(self, path: Path) -> None:
        problems = find_problems(load_yaml_file(path))
        assert not problems, f"{path.name}:\n  " + "\n  ".join(problems)

    @pytest.mark.parametrize("name", [CI_WORKFLOW, DOCS_WORKFLOW])
    def test_only_pull_request_runs_are_cancelled(self, name: str) -> None:
        """A run on main must never be cut off halfway."""
        workflow = load_yaml_file(WORKFLOWS_DIR / name)
        assert workflow["concurrency"]["cancel-in-progress"] == CANCEL_ONLY_PULL_REQUESTS


@pytest.fixture(scope="module")
def ci() -> dict[str, Any]:
    """The parsed ``ci.yml``."""
    return load_yaml_file(WORKFLOWS_DIR / CI_WORKFLOW)


@pytest.fixture(scope="module")
def docs() -> dict[str, Any]:
    """The parsed ``deploy-docs.yml``."""
    return load_yaml_file(WORKFLOWS_DIR / DOCS_WORKFLOW)


@pytest.fixture(scope="module")
def dependabot() -> dict[str, Any]:
    """The parsed ``dependabot.yml``."""
    return load_yaml_file(DEPENDABOT_FILE)


@_requires_workflows
class TestCiWorkflow:
    def test_the_job_is_named_test(self, ci: dict[str, Any]) -> None:
        """The name is how a required check would be selected later."""
        assert "test" in ci["jobs"]

    def test_runs_on_push_to_main_pull_requests_and_by_hand(self, ci: dict[str, Any]) -> None:
        triggers = ci["on"]
        assert triggers["push"]["branches"] == ["main"]
        assert "pull_request" in triggers
        assert "workflow_dispatch" in triggers

    def test_python_version_is_a_string(self, ci: dict[str, Any]) -> None:
        """Unquoted ``3.10`` would be read as the number 3.1."""
        setup = [
            step
            for step in _steps(ci["jobs"]["test"])
            if "actions/setup-python@" in step.get("uses", "")
        ]
        assert len(setup) == 1
        assert setup[0]["with"]["python-version"] == "3.11"

    def test_runs_the_format_checks_and_the_fast_tests(self, ci: dict[str, Any]) -> None:
        commands = [
            line.strip()
            for step in _steps(ci["jobs"]["test"])
            for line in str(step.get("run", "")).splitlines()
        ]
        assert "black --check src tests" in commands
        assert "isort --check src tests" in commands
        pytest_lines = [line for line in commands if line.startswith("pytest ")]
        assert len(pytest_lines) == 1
        assert '-m "not slow"' in pytest_lines[0]
        assert "--strict-markers" in pytest_lines[0]

    def test_installs_the_dev_requirements(self, ci: dict[str, Any]) -> None:
        scripts = "\n".join(str(step.get("run", "")) for step in _steps(ci["jobs"]["test"]))
        assert "-r requirements.txt" in scripts
        assert "-r requirements-dev.txt" in scripts


@_requires_workflows
class TestDocsWorkflow:
    def test_triggers(self, docs: dict[str, Any]) -> None:
        triggers = docs["on"]
        assert triggers["push"]["branches"] == ["main"]
        assert "pull_request" in triggers
        assert "workflow_dispatch" in triggers

    def test_push_and_pull_request_watch_the_same_paths(self, docs: dict[str, Any]) -> None:
        triggers = docs["on"]
        assert triggers["push"]["paths"] == triggers["pull_request"]["paths"]
        assert {
            "docs/**",
            "mkdocs.yml",
            "requirements-docs.txt",
            ".github/workflows/deploy-docs.yml",
        } <= set(triggers["push"]["paths"])

    def test_build_job_checks_links_with_a_strict_build(self, docs: dict[str, Any]) -> None:
        commands = [str(step.get("run", "")).strip() for step in _steps(docs["jobs"]["build"])]
        assert "mkdocs build --strict" in commands

    def test_deploy_runs_only_when_enabled_on_main(self, docs: dict[str, Any]) -> None:
        job = docs["jobs"][DEPLOY_JOB]
        for part in DEPLOY_CONDITION_PARTS:
            assert part in job["if"]
        assert job["needs"] == "build"
        assert job["permissions"] == {"pages": "write", "id-token": "write"}

    def test_the_pages_artifact_is_uploaded_only_when_deploying(self, docs: dict[str, Any]) -> None:
        uploads = [
            step
            for step in _steps(docs["jobs"]["build"])
            if "actions/upload-pages-artifact@" in step.get("uses", "")
        ]
        assert len(uploads) == 1
        for part in DEPLOY_CONDITION_PARTS:
            assert part in uploads[0]["if"]

    def test_build_job_has_no_job_level_permissions(self, docs: dict[str, Any]) -> None:
        """The build job keeps the read-only default."""
        assert "permissions" not in docs["jobs"]["build"]


@_requires_dependabot
class TestDependabot:
    def test_version_is_2(self, dependabot: dict[str, Any]) -> None:
        assert dependabot["version"] == 2

    def test_only_github_actions_are_tracked(self, dependabot: dict[str, Any]) -> None:
        """Python packages are pinned by hand on purpose."""
        ecosystems = [update["package-ecosystem"] for update in dependabot["updates"]]
        assert ecosystems == ["github-actions"]

    def test_updates_come_once_a_month_with_a_cap(self, dependabot: dict[str, Any]) -> None:
        update = dependabot["updates"][0]
        assert update["directory"] == "/"
        assert update["schedule"]["interval"] == "monthly"
        assert isinstance(update["open-pull-requests-limit"], int)
        assert 1 <= update["open-pull-requests-limit"] <= 5


@_requires_dev_requirements
class TestFormatToolPins:
    """The format checks in CI are only stable if the tool versions are."""

    @pytest.mark.parametrize("tool", PINNED_TOOLS)
    def test_tool_is_pinned_to_an_exact_version(self, tool: str) -> None:
        pins = parse_requirement_pins(DEV_REQUIREMENTS_FILE.read_text(encoding="utf-8"))
        assert tool in pins, f"{tool} is missing from requirements-dev.txt"
        assert _EXACT_PIN.match(pins[tool]), f"{tool} needs ==X.Y.Z, got {pins[tool]!r}"
