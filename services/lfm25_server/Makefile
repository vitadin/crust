# LFM25 server Makefile

ARGS   ?=
TARGET ?=

.PHONY: help install sync lock run run-main test test-one check deps clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  %-12s %s\n", $$1, $$2}'

install:  ## Install dependencies from lockfile (frozen)
	uv sync --frozen

sync:  ## Sync environment and allow lockfile updates
	uv sync

lock:  ## Update lockfile
	uv lock

run:  ## Run server  (pass extra flags via ARGS="--config config.yaml")
	uv run python -m lfm25_server $(ARGS)

run-main:  ## Run server via main.py  (pass extra flags via ARGS=)
	uv run python main.py $(ARGS)

test:  ## Run full unit test suite
	uv run python -m unittest discover -s tests -v

test-one:  ## Run one test target  (TARGET=tests.test_handler.HandlerTests)
	uv run python -m unittest -v $(TARGET)

check:  ## Compile-check sources without running inference
	uv run python -m compileall lfm25_server tests

deps:  ## Show project dependency tree
	uv tree

clean:  ## Remove caches and build artifacts
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov build dist .eggs *.egg-info
