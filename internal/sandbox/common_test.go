package sandbox

import (
	"os"
	"path/filepath"
	"testing"
)

// setupSandboxHelperPath finds the sandbox helper binary.
// Returns the path to the helper or skips the test.
func setupSandboxHelperPath(t testing.TB) string {
	// Find project root by looking for go.mod
	dir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Failed to get working directory: %v", err)
	}

	// Walk up to find project root
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			break
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatal("Could not find project root (go.mod)")
		}
		dir = parent
	}

	// Check for sandbox helper binary
	// The path depends on where it's built. Common locations:
	candidates := []string{
		filepath.Join(dir, "cmd", "bakelens-sandbox", "target", "release", "bakelens-sandbox"),
		filepath.Join(dir, "target", "release", "bakelens-sandbox"),
		filepath.Join(dir, "bakelens-sandbox"),
	}

	var helperPath string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			helperPath = c
			break
		}
	}

	if helperPath == "" {
		t.Skipf("sandbox helper not found. build it with: make build-sandbox (or ensure it is in one of %v)", candidates)
	}

	// Set environment variable so findBakelensSandbox() can find it during tests
	os.Setenv("CRUST_SANDBOX_HELPER_PATH", helperPath)
	
	return helperPath
}
