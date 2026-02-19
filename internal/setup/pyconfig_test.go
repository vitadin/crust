package setup

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestReadPyConfig_BasicParsing(t *testing.T) {
	// Setup temp dir with a config file
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	content := []byte(`
# Comment
models:
  - id: lfm2.5
    backend: mlx
    model_path: ./models/LFM2.5
    enabled: true
`)
	if err := os.WriteFile(configPath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	root, entries, err := ReadPyConfig(dir)
	if err != nil {
		t.Fatalf("ReadPyConfig failed: %v", err)
	}

	if root == nil {
		t.Fatal("expected root node")
	}

	if len(entries) != 1 {
		t.Errorf("expected 1 model entry, got %d", len(entries))
	} else {
		m := entries[0]
		if m.ID != "lfm2.5" {
			t.Errorf("expected ID lfm2.5, got %s", m.ID)
		}
		if m.Backend != "mlx" {
			t.Errorf("expected backend mlx, got %s", m.Backend)
		}
		if m.ModelPath != "./models/LFM2.5" {
			t.Errorf("expected path ./models/LFM2.5, got %s", m.ModelPath)
		}
	}
}

func TestAddModelEntry(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	content := []byte("models: []\n")
	if err := os.WriteFile(configPath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	root, _, err := ReadPyConfig(dir)
	if err != nil {
		t.Fatal(err)
	}

	newEntry := PyModelEntry{
		ID:        "new-model",
		Backend:   "mlx",
		ModelPath: "/path/to/model",
		Enabled:   true,
	}

	if err := AddModelEntry(root, newEntry); err != nil {
		t.Fatalf("AddModelEntry failed: %v", err)
	}

	// Verify it was added by checking internal struct or marshalling
	// Let's write it and re-read it
	if err := WritePyConfig(dir, root); err != nil {
		t.Fatal(err)
	}

	_, entries, err := ReadPyConfig(dir)
	if err != nil {
		t.Fatal(err)
	}

	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].ID != "new-model" {
		t.Errorf("expected new-model, got %s", entries[0].ID)
	}

	// Test duplicate ID
	if err := AddModelEntry(root, newEntry); err == nil {
		t.Error("expected error adding duplicate model, got nil")
	}
}

func TestRemoveModelEntry(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	content := []byte(`
models:
  - id: keep-me
    enabled: true
  - id: remove-me
    enabled: true
`)
	if err := os.WriteFile(configPath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	root, _, err := ReadPyConfig(dir)
	if err != nil {
		t.Fatal(err)
	}

	if err := RemoveModelEntry(root, "remove-me"); err != nil {
		t.Fatalf("RemoveModelEntry failed: %v", err)
	}

	// Verify removal
	if err := WritePyConfig(dir, root); err != nil {
		t.Fatal(err)
	}
	_, entries, err := ReadPyConfig(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Errorf("expected 1 entry, got %d", len(entries))
	}
	if entries[0].ID != "keep-me" {
		t.Errorf("expected keep-me, got %s", entries[0].ID)
	}

	// Test removing non-existent
	if err := RemoveModelEntry(root, "non-existent"); err == nil {
		t.Error("expected error removing non-existent model, got nil")
	}
}

func TestWritePyConfig_PreservesComments(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	content := []byte(`# Top comment
models:
  # Model comment
  - id: m1
    enabled: true
# Bottom comment
`)
	if err := os.WriteFile(configPath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	root, _, err := ReadPyConfig(dir)
	if err != nil {
		t.Fatal(err)
	}

	// Modify something (add a model)
	newEntry := PyModelEntry{ID: "m2", Enabled: true}
	if err := AddModelEntry(root, newEntry); err != nil {
		t.Fatal(err)
	}

	if err := WritePyConfig(dir, root); err != nil {
		t.Fatal(err)
	}

	newContent, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	s := string(newContent)
	if !strings.Contains(s, "# Top comment") {
		t.Error("Top comment lost")
	}
	if !strings.Contains(s, "# Model comment") {
		t.Error("Model comment lost")
	}
	if !strings.Contains(s, "# Bottom comment") {
		t.Error("Bottom comment lost")
	}
}

func TestCollectModelPaths(t *testing.T) {
	dir := t.TempDir()
	configPath := filepath.Join(dir, "config.yaml")
	
	absPath := filepath.Join(dir, "external-model")
	relPath := "models/local-model" // relative to dir

	content := []byte(`
models:
  - id: m1
    model_path: ` + absPath + `
  - id: m2
    model_path: ./` + relPath + `
`)
	if err := os.WriteFile(configPath, content, 0o644); err != nil {
		t.Fatal(err)
	}

	paths, err := CollectModelPaths(dir)
	if err != nil {
		t.Fatalf("CollectModelPaths failed: %v", err)
	}

	if len(paths) != 2 {
		t.Fatalf("expected 2 paths, got %d", len(paths))
	}

	// Check if paths are absolute
	for _, p := range paths {
		if !filepath.IsAbs(p) {
			t.Errorf("path %s is not absolute", p)
		}
	}

	// Check specific paths (order is preserved from config)
	if paths[0] != absPath {
		t.Errorf("expected %s, got %s", absPath, paths[0])
	}
	
	expectedRel := filepath.Join(dir, relPath)
	if paths[1] != expectedRel {
		t.Errorf("expected %s, got %s", expectedRel, paths[1])
	}
}
