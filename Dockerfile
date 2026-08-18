# syntax=docker/dockerfile:1
# Image strategy:
# - python:3.11-slim base (no build toolchain needed: pure-python deps + wheels)
# - torch comes from TORCH_INDEX. Default is the ROCm wheel index: the
#   project targets AMD GPU hosts, and a ROCm torch build falls back to CPU
#   cleanly when no GPU is exposed (torch.cuda.is_available() → False), so
#   ROCm-by-default costs image size, never correctness. The ROCm userspace
#   is bundled in the wheel — the host only needs the amdgpu driver.
#   No-GPU / smallest-image build (multi-GB smaller):
#     docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu ...
#   (`make build-cpu`.) ROCm wheels are x86_64-only — build on/for
#   linux/amd64, see Makefile. On a ROCm 7 host, bump the index
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
# Local model BAKED into the image (default): a deployed run must download
# nothing — a cold multi-GB pull at startup is a run-killer. Size check:
# ROCm layers ~4.9 GB compressed + ~2.9 GB weights ≈ 7.9 GB. Dev escape hatch:
#   docker build --build-arg BAKE_MODEL="" ...   (small image, no weights)
ARG BAKE_MODEL=Qwen/Qwen2.5-1.5B-Instruct

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

# Only the model-loading import chain before the bake, so edits to
# router/main/etc. never invalidate the multi-GB bake layer.
COPY --chown=agent:agent config.py schemas.py local_model.py ./
USER agent

# Pin the runtime default to the baked model, so the image never loads a
# model it doesn't contain. Empty BAKE_MODEL → empty env → config.py falls
# back to its own default (dev images download on first use).
ENV LOCAL_MODEL_NAME=${BAKE_MODEL}

# snapshot_download, not LocalModel().load(): the bake needs the FILES in
# HF_HOME, not a full weight load — loading costs ~6 GB RAM and minutes
# under qemu emulation for zero extra benefit at build time.
RUN if [ -n "${BAKE_MODEL}" ]; then \
        python -c "from huggingface_hub import snapshot_download; snapshot_download('${BAKE_MODEL}')"; \
    fi

# AFTER the bake (which needs network): at RUNTIME load the baked model from
# the local cache and NEVER phone home. Without this, transformers issues a
# HEAD to huggingface.co on every load — verified on an AMD GPU host:
# with the HF host unreachable the tokenizer load raises and EVERY local task
# errors. Baking is meant to download nothing; this makes that actually true.
# The remote path uses plain `requests` to FIREWORKS_BASE_URL, unaffected.
# Override with -e HF_HUB_OFFLINE=0 to swap in an un-baked LOCAL_MODEL_NAME.
# MUST stay after the bake RUN — setting it earlier blocks the bake download.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY --chown=agent:agent . .

ENTRYPOINT ["python", "main.py"]
# Batch mode (the container default): read /input/tasks.json, write
# /output/results.json. Dev/mock runs override CMD, e.g.:
#   docker run ... hybrid-router-agent --tasks tasks/sample_tasks.json --mock
CMD ["--input", "/input/tasks.json", "--output", "/output/results.json"]
