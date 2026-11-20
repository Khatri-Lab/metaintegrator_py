# Continuous Integration

Two GitHub Actions workflows check every change. This page says what they do and what to run before you push.

## What runs

| Workflow | File | Runs on | What it does |
|---|---|---|---|
| CI | `.github/workflows/ci.yml` | Pushes to `main`, pull requests, manual runs | Installs the package, checks formatting, and runs the fast tests. |
| Documentation | `.github/workflows/deploy-docs.yml` | Changes to the docs on `main` and in pull requests, manual runs | Builds the site with `mkdocs build --strict`. Deploys to GitHub Pages only when enabled. |

The CI job is named `test`. A pull request should finish in about five minutes.

The CI job runs these steps, in this order:

1. Install `requirements.txt`, `requirements-dev.txt` and the package on Python 3.11.
2. `black --check src tests`
3. `isort --check src tests`
4. `pytest -m "not slow"`

CI replaces the default pytest options from `pyproject.toml` with a shorter set. It keeps `--strict-markers`, so a misspelled marker fails the run. It does not collect coverage.

Tests marked `slow` do not run in CI. Run them by hand with `pytest -m slow`. CI also does not run mypy or flake8.

## Before you push

Run the same checks that CI runs, from the repository root:

```bash
black src tests
isort src tests
pytest -m "not slow"
```

The first two commands rewrite files. Add `--check` to only look. If all three pass, CI should pass. If you changed anything under `docs/` or `mkdocs.yml`, also run `mkdocs build --strict`, because a broken link fails the Documentation workflow.

Treat a red check as a blocker for a merge, even when GitHub would allow the merge.

## Reading a red run

Open the run on the Actions tab and look at the name of the failing step.

| Failing step | What to do |
|---|---|
| Format and import order | Run `black src tests` and `isort src tests`, then commit the result. |
| Tests | Read the failure summary at the end of the step, then run that test on your machine. |
| Install | A dependency changed or disappeared. Check `requirements.txt` and `requirements-dev.txt`. |
| Build documentation | MkDocs names the broken link or the missing page. Fix it and run `mkdocs build --strict`. |

## Tool versions

`black` and `isort` are pinned to exact versions in `requirements-dev.txt`. A newer release can format code differently, and that would turn a green run red without any change in the code. To update a pin, change it in a pull request of its own, run both tools on the whole tree, and commit the reformatting in the same pull request.

Dependabot proposes updates for the GitHub Actions in the workflows once a month. It does not track Python packages.

## Publishing the documentation

The Documentation workflow always builds the site, so it works as a link check. It deploys only when the repository variable `DEPLOY_DOCS` is `true` and the run is on `main`. To publish:

1. In the repository settings, open Pages and set the source to GitHub Actions.
2. In the repository settings, open Secrets and variables, then Actions, then Variables. Add `DEPLOY_DOCS` with the value `true`.

Until both are done, a push to `main` ends green and the `deploy` job shows as skipped.

## Related pages

- [Development Guide](development.md)
- [Contributing Guidelines](guidelines.md)
