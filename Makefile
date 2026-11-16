SHELL := /usr/bin/env bash -euo pipefail

CARGO         := cargo
CARGO_FLAGS   :=
RELEASE_FLAGS := --release
WORKSPACE     := --workspace

INTEGRATION_FEATURE := --features integration

FIXTURE_DIR   := crates/voicekit-core/fixtures
HELLO_WAV     := $(FIXTURE_DIR)/hello.wav
SAMPLE_RATE   := 16000
DURATION_SECS := 2

# Local model paths (override for non-standard layouts)
MODELS_DIR    ?= $(PWD)/models
REGISTRY      ?= $(PWD)/models.toml
LOCAL_CONFIG  ?= $(PWD)/config.local.toml

# ORT is loaded at runtime via dlopen (load-dynamic feature).
ORT_DYLIB_PATH ?= /usr/local/lib/python3.13/dist-packages/onnxruntime/capi/libonnxruntime.so.1.29.0
export ORT_DYLIB_PATH
-include .env
export
# Build

.PHONY: build
build:
	$(CARGO) build $(WORKSPACE) $(RELEASE_FLAGS)

.PHONY: build-dev
build-dev:
	$(CARGO) build $(WORKSPACE)

.PHONY: build-server
build-server:
	$(CARGO) build -p voicekit-server $(RELEASE_FLAGS)

# Format & lint

.PHONY: fmt
fmt:
	$(CARGO) fmt $(WORKSPACE)

.PHONY: fmt-check
fmt-check:
	$(CARGO) fmt $(WORKSPACE) -- --check

.PHONY: lint
lint:
	$(CARGO) clippy $(WORKSPACE) -- -D warnings

# Unit tests - no model files required, CI-safe

.PHONY: test
test: test-session test-vad test-sentence test-llm test-stt test-tts test_text_clean

.PHONY: test-session
test-session:
	$(CARGO) test -p voicekit-core --test session_test $(CARGO_FLAGS)

.PHONY: test-vad
test-vad:
	$(CARGO) test -p voicekit-core --test vad_test $(CARGO_FLAGS)

.PHONY: test-text-clean
test-text-clean:
	$(CARGO) test -p voicekit-core --test text_clean_test $(CARGO_FLAGS)

.PHONY: test-sentence
test-sentence:
	$(CARGO) test -p voicekit-core --test sentence_test $(CARGO_FLAGS)

# Integration tests - require model files, run after `make models`

.PHONY: test-integration
test-integration:
	VOICEKIT_INTEGRATION=1 \
	$(CARGO) test -p voicekit-core $(INTEGRATION_FEATURE) $(CARGO_FLAGS)

.PHONY: test-stt
test-stt:
	VOICEKIT_INTEGRATION=1 \
	$(CARGO) test -p voicekit-core --test stt_test \
	    $(INTEGRATION_FEATURE) $(CARGO_FLAGS)

.PHONY: test-llm
test-llm:
	VOICEKIT_INTEGRATION=1 \
	$(CARGO) test -p voicekit-core --test llm_test \
	    $(INTEGRATION_FEATURE) $(CARGO_FLAGS)

.PHONY: test-tts
test-tts:
	VOICEKIT_INTEGRATION=1 \
	$(CARGO) test -p voicekit-core --test tts_test \
	    $(INTEGRATION_FEATURE) $(CARGO_FLAGS)

.PHONY: test-session-integration
test-session-integration:
	VOICEKIT_INTEGRATION=1 \
	$(CARGO) test -p voicekit-core --test session_integration_test \
	    $(INTEGRATION_FEATURE) $(CARGO_FLAGS)

# E2E tests - full pipeline with real models
.PHONY: test-e2e
test-e2e:
	VOICEKIT_INTEGRATION=1 \
	VOICEKIT_CONFIG=$(LOCAL_CONFIG) \
	VOICEKIT_REGISTRY=$(REGISTRY) \
	VOICEKIT_MODELS_DIR=$(MODELS_DIR) \
	VOICEKIT_HELLO=$(PWD)/$(HELLO_WAV) \
	$(CARGO) test -p e2e --test pipeline --release \
	    $(INTEGRATION_FEATURE) $(CARGO_FLAGS) -- --nocapture

.PHONY: test-all
test-all: test

.PHONY: test-all-integration
test-all-integration: test test-integration test-session-integration test-e2e

# Fixtures & models

.PHONY: fixtures
fixtures: $(HELLO_WAV)

# Regenerate whenever the script or the paradee model changes.
$(HELLO_WAV): scripts/generate_fixtures.py models/paradee/onnx/paradee_int8.onnx
	@mkdir -p $(FIXTURE_DIR)
	PIP_BREAK_SYSTEM_PACKAGES=1 python3 scripts/generate_fixtures.py \
	    --out-dir     $(FIXTURE_DIR) \
	    --sample-rate $(SAMPLE_RATE) \
	    --duration    $(DURATION_SECS)

.PHONY: models
models:
	docker compose run --rm model-init

.PHONY: models-clean
models-clean:
	rm -rf models/

# Run (local — not Docker; uses local config and model paths)

.PHONY: run
run: build-server
	RUST_LOG=info ./target/release/voicekit-server \
	    --config     $(LOCAL_CONFIG) \
	    --registry   $(REGISTRY) \
	    --models-dir $(MODELS_DIR)

.PHONY: run-dev
run-dev:
	RUST_LOG=debug \
	$(CARGO) run -p voicekit-server -- \
	    --config     $(LOCAL_CONFIG) \
	    --registry   $(REGISTRY) \
	    --models-dir $(MODELS_DIR)

# Clean

.PHONY: clean
clean:
	$(CARGO) clean

.PHONY: clean-fixtures
clean-fixtures:
	rm -f $(FIXTURE_DIR)/*.wav $(FIXTURE_DIR)/*.bin

.PHONY: clean-orphans
clean-orphans:
	docker compose down --remove-orphans

# Docker image

IMAGE := ghcr.io/festuskipkoech/voicekit

.PHONY: docker-build
docker-build:
	docker build -t $(IMAGE):latest .

.PHONY: docker-push
docker-push:
	docker push $(IMAGE):latest

.PHONY: docker-release
docker-release: docker-build docker-push

# CI - no model files required

.PHONY: ci
ci: fmt-check lint test py-fmt-check py-lint py-test

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:' Makefile | \
	    awk -F':' '{printf "  %-30s\n", $$1}' | sort

# Python SDK

PY_SDK_DIR := sdks/voicekit-py

.PHONY: py-install
py-install:
	cd $(PY_SDK_DIR) && uv sync

.PHONY: py-test
py-test: py-install
	cd $(PY_SDK_DIR) && uv run pytest

.PHONY: py-lint
py-lint:
	cd $(PY_SDK_DIR) && uv run ruff check .

.PHONY: py-fmt
py-fmt:
	cd $(PY_SDK_DIR) && uv run ruff format .

.PHONY: py-fmt-check
py-fmt-check:
	cd $(PY_SDK_DIR) && uv run ruff format --check .