# metaintegrator_py: Development / Test Container
# ──────────────────────────────────────────────────
# There is no "production" target here on purpose: this package isn't
# deployed as a service, it's mounted read-write into a container so its
# source can be exercised (pytest, test_runner_all.py, notebooks, parity
# comparisons against the web app) without installing anything on the
# host. See metaintegrator-web/docs/docker_mac_host_dev_setup.md for how
# this is wired up via docker-compose.metaintegrator-py.yml.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace/metaintegrator_py

# System deps for scipy/matplotlib/lxml wheels that need a compiler on
# some platforms, plus git (setuptools_scm reads version info from it).
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies from requirements files first so `docker compose
# build` only re-installs when these actually change, not on every source
# edit (the source itself is bind-mounted at runtime, not copied here).
COPY requirements.txt requirements-dev.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -r requirements-dev.txt

# The real source is bind-mounted over /workspace/metaintegrator_py by
# docker-compose.metaintegrator-py.yml: this COPY only makes `docker
# build` (and a plain `docker run` without the compose mount) usable on
# their own, e.g. in CI.
COPY . .
RUN pip install -e .

CMD ["sleep", "infinity"]
