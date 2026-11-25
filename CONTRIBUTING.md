# Contributing to metaintegrator_py

Thank you for your interest. The full guidelines are in [docs/contributing/guidelines.md](docs/contributing/guidelines.md). This page is the short version.

This is a research software project with a small team. Issues and pull requests are welcome, and they are reviewed on a best-effort basis.

## Before you start

- Search the existing issues first.
- For a bug, open an issue with the steps to reproduce, what you expected, what happened, your Python version and your operating system.
- For a larger change or a new feature, open an issue first and describe the idea.
- For a security problem, do not open an issue. Follow [SECURITY.md](SECURITY.md).

## Set up

You need Python 3.10 or later. CI uses Python 3.11.

```bash
git clone https://github.com/<your-account>/metaintegrator_py.git
cd metaintegrator_py
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

Fork the repository first and clone your fork. Create a branch from `main` for your change. More detail: [docs/contributing/development.md](docs/contributing/development.md).

## Before you push

Run the checks that CI runs, from the repository root:

```bash
black src tests
isort src tests
pytest -m "not slow"
```

The first two commands rewrite files. If you changed anything under `docs/` or `mkdocs.yml`, also run `mkdocs build --strict`. If your change can alter numeric results, read [docs/contributing/parity_testing.md](docs/contributing/parity_testing.md). More detail: [docs/contributing/continuous_integration.md](docs/contributing/continuous_integration.md).

## Pull requests

- Keep a pull request small and focused on one thing.
- Describe what changed and why, and link the issue.
- Add or update tests, and update the documentation.
- The `test` check must be green before a merge.
- A maintainer must approve the workflow runs on a pull request from an outside contributor. This is a GitHub safety setting, so the checks may start a little later.

## Your commit email

Commit messages and author emails are public and cannot be changed later. Check which address git will use before you commit:

```bash
git var GIT_AUTHOR_IDENT
```

GitHub offers a private "noreply" address in Settings, Emails.

## License

metaintegrator_py is released under the [Apache License 2.0](LICENSE). By contributing, you agree that your contribution is licensed under the same terms.
