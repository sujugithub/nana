# syntax=docker/dockerfile:1
# Image strategy:
# - python:3.11-slim base (no build toolchain needed: pure-python deps + wheels)
# - torch comes from TORCH_INDEX. Default is the ROCm wheel index: the
#   competition scores on AMD GPUs, and a ROCm torch build falls back to CPU
#   cleanly when no GPU is exposed (torch.cuda.is_available() → False), so
#   ROCm-by-default costs image size, never correctness. The ROCm userspace
#   is bundled in the wheel — the host only needs the amdgpu driver.
#   No-GPU / smallest-image build (multi-GB smaller):
#     docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu ...
#   (`make build-cpu`.) ROCm wheels are x86_64-only — build on/for
#   linux/amd64, see Makefile. If AMD Dev Cloud runs ROCm 7, bump the index
#   to .../whl/rocm7.0.
# - At RUNTIME the container must see the GPU devices:
#     docker run --device=/dev/kfd --device=/dev/dri ...   (`make docker-run-gpu`)
# - Layer order is chosen so the expensive layers never rebuild during rapid
#   iteration: deps → model-bake import chain → (optional bake) → full source.
# - The non-root user exists BEFORE anything large is written, and files are
#   copied with --chown. Never `chown -R` after a big COPY/RUN: overlayfs
#   copies every file up into a new layer, doubling the image.

FROM python:3.11-slim

# ROCm (AMD GPU) torch by default — see header comment for the CPU override.
ARG TORCH_INDEX=https://download.pytorch.org/whl/rocm6.4

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

RUN useradd --create-home agent && mkdir -p /app && chown agent:agent /app
WORKDIR /app

COPY requirements.txt .
# torch first, alone, from TORCH_INDEX (guarantees the +rocm / +cpu build
# wins over the default CUDA-bundling PyPI build), then everything else from
# PyPI. The second install sees torch satisfied and skips it.
RUN pip install torch --index-url ${TORCH_INDEX} \
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
