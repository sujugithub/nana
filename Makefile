# Project shortcuts. `make test` before every commit; it needs no deps.
.PHONY: test mock run demo build build-cpu docker-run docker-run-gpu docker-run-harness \
	ghcr-login push push-cpu image-size

# Public container registry (check the package's visibility settings on GHCR
# after the first push).
IMAGE ?= ghcr.io/sujugithub/hybrid-token-routing-agent
TAG ?= latest

test:            ## offline wiring test — stdlib only, runs anywhere
	python3 test_harness.py

demo:            ## 🍌 banana demo: 8-category run + summary graph (real models)
	python3 scripts/banana.py --demo

mock:            ## run the sample task file in mock mode
	python3 main.py --tasks tasks/sample_tasks.json --mock

run:             ## real run — needs pip deps + FIREWORKS_API_KEY
	python3 main.py --tasks tasks/sample_tasks.json

# --platform pin: the deployment target is x86_64; without it, a build on an
# Apple Silicon Mac silently produces an arm64 image that dies with "exec
# format error" when shipped (and ROCm torch wheels are x86_64-only anyway).
# Default build = ROCm torch (for AMD GPU hosts); it still runs fine on
# CPU-only hosts, just bigger. `make build-cpu` for the small image.
build:
	docker build --platform=linux/amd64 -t hybrid-router-agent .

build-cpu:
	docker build --platform=linux/amd64 \
		--build-arg TORCH_INDEX=https://download.pytorch.org/whl/cpu \
		-t hybrid-router-agent-cpu .

# logs/ is mounted out so usage.jsonl survives --rm — it's the calibration
# audit trail. Image CMD is harness mode, so dev runs pass --tasks explicitly.
docker-run:
	mkdir -p logs
	docker run --rm --env-file .env -v "$$(pwd)/logs:/app/logs" \
		hybrid-router-agent --tasks tasks/sample_tasks.json

# On an AMD-GPU host: expose the GPU devices to
# the container. _pick_device() then sees torch.cuda.is_available() == True.
# /dev/kfd is group-owned by `render`, /dev/dri/card* by `video` — the
# container process must be in BOTH or torch.cuda.is_available() is False
# (verified 2026-07-07: `--group-add video` alone → cuda:False). GIDs vary by
# host, so resolve them at run time rather than hardcoding.
docker-run-gpu:
	mkdir -p logs
	docker run --rm --env-file .env -v "$$(pwd)/logs:/app/logs" \
		--device=/dev/kfd --device=/dev/dri \
		--group-add "$$(getent group render | cut -d: -f3)" \
		--group-add "$$(getent group video | cut -d: -f3)" \
		hybrid-router-agent --tasks tasks/sample_tasks.json

# Run batch mode in-container: /input + /output mounts, default CMD.
# Env comes from .env locally; inject it however your deployment does.
docker-run-harness:
	mkdir -p logs harness/input harness/output
	cp tasks/sample_tasks.json harness/input/tasks.json
	docker run --rm --env-file .env \
		-v "$$(pwd)/harness/input:/input:ro" \
		-v "$$(pwd)/harness/output:/output" \
		-v "$$(pwd)/logs:/app/logs" hybrid-router-agent
	@echo "── /output/results.json ──"
	@cat harness/output/results.json; echo

# ── Submission: GHCR (public) ────────────────────────────────────────────
# Needs `gh auth login` once (with the write:packages scope:
# `gh auth refresh -s write:packages`).
ghcr-login:
	gh auth token | docker login ghcr.io -u sujugithub --password-stdin

push:            ## push the ROCm image
	docker tag hybrid-router-agent $(IMAGE):$(TAG)
	docker push $(IMAGE):$(TAG)

push-cpu:        ## push the CPU fallback image as :cpu
	docker tag hybrid-router-agent-cpu $(IMAGE):cpu
	docker push $(IMAGE):cpu

# Compressed size ≈ what the registry actually stores.
# (docker images shows the UNcompressed size — not the number that counts.)
image-size:
	docker save hybrid-router-agent-cpu | gzip | wc -c | \
		awk '{printf "hybrid-router-agent-cpu compressed: %.2f GB\n", $$1/1e9}'
