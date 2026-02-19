# Crust Makefile

VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "dev")
BINARY_NAME = crust
BUILD_DIR = build
LDFLAGS = -ldflags "-s -w -X main.Version=$(VERSION)"

# On macOS, Homebrew LLVM clang cannot link against the macOS 15 SDK (.tbd files).
# Force the system Xcode clang instead.
ifeq ($(shell uname -s),Darwin)
export CC := /usr/bin/clang
endif

# Platforms for release
PLATFORMS = darwin/amd64 darwin/arm64 linux/amd64 linux/arm64

LFM_DIR := services/lfm25_server

.PHONY: all build clean test test-go test-python test-unit test-e2e test-sandbox-e2e test-daemon-e2e test-all test-data release install lint vulncheck semgrep help build-bpf install-bpf uninstall-bpf lfm-start

all: build

## Build for current platform
build:
	go build $(LDFLAGS) -o $(BINARY_NAME) .

## Run all unit tests (Go + Python)
test: test-unit

test-unit: test-go test-python

## Run Go unit tests
test-go:
	go test -v ./...

## Run Python unit tests (LFM25 server)
test-python:
	$(MAKE) -C $(LFM_DIR) test

## Run E2E sandbox tests
test-sandbox-e2e: test-data
	go test -v -tags=sandbox_e2e -timeout=5m ./internal/sandbox/...

## Run E2E daemon tests
test-daemon-e2e:
	go test -v ./internal/e2e/...

## Run all E2E tests
test-e2e: test-sandbox-e2e test-daemon-e2e

## Run all tests (Go unit + Python unit + E2E)
test-all: test-unit test-e2e
	@echo "All tests complete"

## Create test data for E2E tests
test-data:
	@if [ ! -d test-data ]; then \
		echo "Creating test data..."; \
		mkdir -p test-data/.ssh test-data/secrets test-data/project; \
		echo "SECRET=test" > test-data/.env; \
		echo "LOCAL=test" > test-data/.env.local; \
		echo "fake-rsa-key" > test-data/.ssh/id_rsa; \
		echo "fake-ed25519" > test-data/.ssh/id_ed25519; \
		echo '{"key":"secret"}' > test-data/secrets/credentials.json; \
		echo "password: test" > test-data/secrets/secrets.yaml; \
		echo "package main" > test-data/project/main.go; \
		echo "# README" > test-data/project/README.md; \
		echo "hello" > test-data/project/data.txt; \
		chmod -R 755 test-data; \
		echo "Test data created at ./test-data"; \
	fi

## Run Go linter
lint:
	golangci-lint run ./...

## Check dependencies for known vulnerabilities
vulncheck:
	govulncheck ./...

## Run semgrep SAST scan
semgrep:
	semgrep scan --config auto .

## Build bpf-helper binary (Linux only)
build-bpf:
	go build $(LDFLAGS) -o bpf-helper ./cmd/bpf-helper/

## Install bpf-helper and systemd service
install-bpf: build-bpf
	@if [ "$$(uname -s)" != "Linux" ]; then echo "bpf-helper is Linux-only"; exit 1; fi
	sudo install -d /usr/libexec/crust
	sudo install -m 755 bpf-helper /usr/libexec/crust/bpf-helper
	sudo cp init/crust-bpf@.service /etc/systemd/system/
	sudo systemctl daemon-reload
	@echo "Installed bpf-helper to /usr/libexec/crust/"
	@echo ""
	@echo "Enable for your user:"
	@echo "  sudo systemctl enable --now crust-bpf@$$(whoami).service"

## Uninstall bpf-helper and systemd service
uninstall-bpf:
	@if [ "$$(uname -s)" != "Linux" ]; then echo "bpf-helper is Linux-only"; exit 1; fi
	-sudo systemctl stop "crust-bpf@$$(whoami).service" 2>/dev/null
	-sudo systemctl disable "crust-bpf@$$(whoami).service" 2>/dev/null
	-sudo rm -f /usr/libexec/crust/bpf-helper
	-sudo rm -f /etc/systemd/system/crust-bpf@.service
	-sudo rm -rf "/etc/systemd/system/crust-bpf@$$(whoami).service.d"
	sudo systemctl daemon-reload
	@echo "Uninstalled bpf-helper"

## Clean build artifacts
clean:
	rm -f $(BINARY_NAME)
	rm -f bpf-helper
	rm -rf $(BUILD_DIR)
	rm -f coverage.out
	rm -rf test-data
	find . -name "*.test" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

## Install to /usr/local/bin
install: build
	sudo mv $(BINARY_NAME) /usr/local/bin/$(BINARY_NAME)
	@echo "Installed to /usr/local/bin/$(BINARY_NAME)"

## Build release tarball for current platform
release: clean
	@mkdir -p $(BUILD_DIR)
	@os=$$(go env GOOS); \
	arch=$$(go env GOARCH); \
	output=$(BUILD_DIR)/$(BINARY_NAME)-$(VERSION)-$$os-$$arch; \
	mkdir -p $$output; \
	echo "Building $$os/$$arch..."; \
	CGO_ENABLED=1 go build $(LDFLAGS) -o $$output/$(BINARY_NAME) .; \
	tar -czf $$output.tar.gz -C $$output $(BINARY_NAME); \
	rm -rf $$output; \
	echo "Created: $$output.tar.gz"

## Start the LFM25 inference server (requires uv env in services/lfm25_server/)
lfm-start:
	$(MAKE) -C $(LFM_DIR) run ARGS="--config config.yaml"

## Show help
help:
	@echo "Crust Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build             Build for current platform"
	@echo "  test              Run all unit tests (Go + Python)"
	@echo "  test-unit         Run all unit tests (Go + Python)"
	@echo "  test-go           Run Go unit tests only"
	@echo "  test-python       Run Python unit tests only (LFM25 server)"
	@echo "  test-e2e          Run all E2E tests (sandbox + daemon)"
	@echo "  test-sandbox-e2e  Run E2E sandbox tests"
	@echo "  test-daemon-e2e   Run E2E daemon tests"
	@echo "  test-all          Run all tests (unit + E2E)"
	@echo "  lfm-start         Start the LFM25 inference server"
	@echo "  lint              Run Go linter"
	@echo "  vulncheck         Check deps for known CVEs (govulncheck)"
	@echo "  semgrep           Run semgrep SAST scan"
	@echo "  clean             Clean build artifacts"
	@echo "  install           Install to /usr/local/bin"
	@echo "  build-bpf         Build bpf-helper (Linux only)"
	@echo "  install-bpf       Install bpf-helper + systemd service"
	@echo "  uninstall-bpf     Remove bpf-helper + systemd service"
	@echo "  release           Build release tarball"
	@echo ""
	@echo "Variables:"
	@echo "  VERSION    Release version (default: git tag or 'dev')"
