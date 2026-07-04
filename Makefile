# Competition shortcuts. `make test` before every commit; it needs no deps.
.PHONY: test mock run build docker-run

test:            ## offline wiring test — stdlib only, runs anywhere
	python3 test_harness.py

mock:            ## run the sample task file in mock mode
	python3 main.py --tasks tasks/sample_tasks.json --mock

run:             ## real run — needs pip deps + FIREWORKS_API_KEY
	python3 main.py --tasks tasks/sample_tasks.json

# --platform pin: the scoring host is x86_64; without it, a build on an
# Apple Silicon Mac silently produces an arm64 image that dies with "exec
# format error" when shipped (and ROCm torch wheels are x86_64-only anyway).
build:
	docker build --platform=linux/amd64 -t hybrid-router-agent .

# logs/ is mounted out so usage.jsonl survives --rm — it's the calibration
# audit trail.
docker-run:
	mkdir -p logs
	docker run --rm --env-file .env -v "$$(pwd)/logs:/app/logs" hybrid-router-agent
