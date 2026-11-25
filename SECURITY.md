# Security policy

## Supported versions

Security fixes go into the latest 1.x release. Older releases do not get fixes.

## Report a vulnerability

Please do not open a public issue for a security problem.

Use GitHub's private reporting:

1. Open the **Security** tab of this repository.
2. Open **Advisories**, then click **Report a vulnerability**.
3. Describe the problem. Only the maintainers can see the report.

A useful report has the version of metaintegrator_py and Python, your operating system, the steps to reproduce the problem, and the impact as far as you can tell.

This is a research software project. Reports are read on a best-effort basis, and we cannot promise a response time.

## What counts as a security problem

Examples:

- reading or writing files outside the folders a user chose;
- running code from a data file or a downloaded file;
- unsafe handling of data fetched from public services such as GEO or UCSC;
- secrets or credentials in the repository, the logs or the documentation.

A wrong statistical result, a crash on odd input or a slow analysis is not a security problem. Please report these as normal issues. A problem that exists only in a third-party package is best reported to that project.

## Safe use

metaintegrator_py runs analyses on your own computer. As with any data analysis tool, do not load files from sources you do not trust.
