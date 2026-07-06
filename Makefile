# Competition shortcuts. `make test` before every commit; it needs no deps.
.PHONY: test mock run build build-cpu docker-run docker-run-gpu docker-run-harness

test:            ## offline wiring test — stdlib only, runs anywhere
	python3 test_harness.py

mock:            ## run the sample task file in mock mode
	python3 main.py --tasks tasks/sample_tasks.json --mock

run:             ## real run — needs pip deps + FIREWORKS_API_KEY
	python3 main.py --tasks tasks/sample_tasks.json

# --platform pin: the scoring host is x86_64; without it, a build on an
# Apple Silicon Mac silently produces an arm64 image that dies with "exec
# format error" when shipped (and ROCm torch wheels are x86_64-only anyway).
# Default build = ROCm torch (the scoring env is AMD GPU); it still runs
# fine on CPU-only hosts, just bigger. `make build-cpu` for the small image.
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

# On an AMD-GPU host (e.g. AMD Developer Cloud): expose the GPU devices to
# the container. _pick_device() then sees torch.cuda.is_available() == True.
docker-run-gpu:
	mkdir -p logs
	docker run --rm --env-file .env -v "$$(pwd)/logs:/app/logs" \
		--device=/dev/kfd --device=/dev/dri --group-add video \
		hybrid-router-agent --tasks tasks/sample_tasks.json

# Simulate the scoring harness locally: /input + /output mounts, default CMD.
# The real harness injects FIREWORKS_* env itself; locally .env stands in.
docker-run-harness:
	mkdir -p logs harness/input harness/output
	cp tasks/sample_tasks.json harness/input/tasks.json
	docker run --rm --env-file .env \
		-v "$$(pwd)/harness/input:/input:ro" \
		-v "$$(pwd)/harness/output:/output" \
		-v "$$(pwd)/logs:/app/logs" hybrid-router-agent
	@echo "── /output/results.json ──"
	@cat harness/output/results.json; echo
