# syntax=docker/dockerfile:1
# Small-image strategy:
# - python:3.11-slim base (no build toolchain needed: pure-python deps + wheels)
# - torch installed from the CPU wheel index — the default PyPI build bundles
#   CUDA libraries and is several GB larger. The scoring environment is
#   limited-compute, so CPU is the safe default.
#   AMD GPU in the scoring env? Swap the index URL for ROCm, e.g.:
#     https://download.pytorch.org/whl/rocm6.2
#   (ROCm wheels are x86_64-only — build on/for linux/amd64, see Makefile.)
# - Layer order is chosen so the expensive layers never rebuild during rapid
#   iteration: deps → model-bake import chain → (optional bake) → full source.
# - The non-root user exists BEFORE anything large is written, and files are
#   copied with --chown. Never `chown -R` after a big COPY/RUN: overlayfs
#   copies every file up into a new layer, doubling the image.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN useradd --create-home agent && mkdir -p /app && chown agent:agent /app
WORKDIR /app

COPY requirements.txt .
# torch first, alone, from the CPU index (guarantees the +cpu build wins),
# then everything else from PyPI. The second install sees torch satisfied
# and skips it.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Only the model-loading import chain before the (optional) bake, so edits to
# router/main/etc. never invalidate a multi-GB bake layer.
COPY --chown=agent:agent config.py schemas.py local_model.py ./
USER agent

# OPTIONAL (kickoff day): bake the local model into the image so the scoring
# run needs no network and pays no cold-download. Requires network at BUILD
# time and AGENT_MOCK unset:
# RUN python -c "from local_model import LocalModel; LocalModel().load()"

COPY --chown=agent:agent . .

ENTRYPOINT ["python", "main.py"]
CMD ["--tasks", "tasks/sample_tasks.json"]
