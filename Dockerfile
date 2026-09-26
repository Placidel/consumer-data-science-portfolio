# Northstar Consumer: one image for the scoring API (default command) and the dashboard.
#
# Deterministic build: the base image is pinned by digest, every Python dependency by
# constraints.txt, the data is generated with the fixed default seed, and the models are trained
# from it (content-addressed versions). The container starts by loading those artifacts; it never
# trains at run time. No secrets are needed at build or run time, and none are baked in.
FROM python:3.12.14-slim-bookworm@sha256:392307d22300de8b5986851a12d9176dfc0fc073e65bf6523ebd7dcbeb23564e

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    NORTHSTAR_DATA_DIR=/app/data/raw \
    NORTHSTAR_MODEL_DIR=/app/models

WORKDIR /app

# Runtime dependencies first (pinned by constraints.txt), in their own layer so code changes do
# not reinstall them.
COPY pyproject.toml README.md LICENSE constraints.txt ./
RUN python -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && python -m pip install --constraint constraints.txt --requirement /tmp/requirements.txt

# Editable install keeps the repository layout (projects/, data/, models/) under /app, which is
# where the package resolves its default paths.
COPY src ./src
RUN python -m pip install --no-deps -e .

# Committed section outputs (read by the dashboard) and the dashboard theme.
COPY projects ./projects
COPY .streamlit ./.streamlit

# Build-time data and model training (about 2 minutes). The API only loads the result.
RUN northstar generate-data --skip-profile \
    && northstar train-models --no-generate

RUN useradd --create-home --uid 10001 northstar
USER northstar

EXPOSE 8000 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"]

CMD ["northstar", "serve", "--host", "0.0.0.0", "--port", "8000"]
