package setup_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/BakeLens/crust/internal/setup"
)

// ---------------------------------------------------------------------------
// DefaultServerDir
// ---------------------------------------------------------------------------

func TestDefaultServerDir_ContainsExpectedComponents(t *testing.T) {
	dir, err := setup.DefaultServerDir()
	if err != nil {
		t.Fatalf("DefaultServerDir: %v", err)
	}
	if dir == "" {
		t.Fatal("DefaultServerDir returned empty string")
	}
	// Must end with .crust/lfm25_server
	want := filepath.Join(".crust", "lfm25_server")
	if !strings.HasSuffix(dir, want) {
		t.Errorf("DefaultServerDir = %q, want suffix %q", dir, want)
	}
}

// ---------------------------------------------------------------------------
// NeedsSetup
// ---------------------------------------------------------------------------

func TestNeedsSetup_NoVersionFile(t *testing.T) {
	dir := t.TempDir()
	if !setup.NeedsSetup(dir, "1.0.0") {
		t.Error("NeedsSetup should return true when version file is missing")
	}
}

func TestNeedsSetup_VersionMatches(t *testing.T) {
	dir := t.TempDir()
	writeVersionFile(t, dir, "1.2.3")

	if setup.NeedsSetup(dir, "1.2.3") {
		t.Error("NeedsSetup should return false when version matches")
	}
}

func TestNeedsSetup_VersionMismatch(t *testing.T) {
	dir := t.TempDir()
	writeVersionFile(t, dir, "1.0.0")

	if !setup.NeedsSetup(dir, "1.1.0") {
		t.Error("NeedsSetup should return true when version differs")
	}
}

func TestNeedsSetup_VersionFileHasTrailingWhitespace(t *testing.T) {
	dir := t.TempDir()
	// Version files may have a trailing newline from os.WriteFile.
	if err := os.WriteFile(filepath.Join(dir, ".setup_version"), []byte("2.0.0\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if setup.NeedsSetup(dir, "2.0.0") {
		t.Error("NeedsSetup should trim whitespace when comparing versions")
	}
}

// ---------------------------------------------------------------------------
// IsServerReady
// ---------------------------------------------------------------------------

func TestIsServerReady_NoVersionFile(t *testing.T) {
	dir := t.TempDir()
	if setup.IsServerReady(dir) {
		t.Error("IsServerReady should return false when version file is absent")
	}
}

func TestIsServerReady_VersionFilePresent(t *testing.T) {
	dir := t.TempDir()
	writeVersionFile(t, dir, "any-version")

	if !setup.IsServerReady(dir) {
		t.Error("IsServerReady should return true when version file exists")
	}
}

func TestIsServerReady_EmptyVersionFile(t *testing.T) {
	dir := t.TempDir()
	// An empty version file still signals "setup ran at least once".
	if err := os.WriteFile(filepath.Join(dir, ".setup_version"), []byte(""), 0o644); err != nil {
		t.Fatal(err)
	}
	if !setup.IsServerReady(dir) {
		t.Error("IsServerReady should return true even for empty version file")
	}
}

// ---------------------------------------------------------------------------
// ModelExists
// ---------------------------------------------------------------------------

func TestModelExists_DirectoryMissing(t *testing.T) {
	dir := t.TempDir()
	if setup.ModelExists(dir) {
		t.Error("ModelExists should return false when models directory is absent")
	}
}

func TestModelExists_DirectoryEmpty(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", setup.ModelName)
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if setup.ModelExists(dir) {
		t.Error("ModelExists should return false for an empty model directory")
	}
}

func TestModelExists_DirectoryHasFiles(t *testing.T) {
	dir := t.TempDir()
	modelDir := filepath.Join(dir, "models", setup.ModelName)
	if err := os.MkdirAll(modelDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(modelDir, "config.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if !setup.ModelExists(dir) {
		t.Error("ModelExists should return true when model directory has files")
	}
}

func TestModelExists_WrongModelName(t *testing.T) {
	dir := t.TempDir()
	wrongDir := filepath.Join(dir, "models", "some-other-model")
	if err := os.MkdirAll(wrongDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(wrongDir, "weights.bin"), []byte("data"), 0o644); err != nil {
		t.Fatal(err)
	}
	if setup.ModelExists(dir) {
		t.Error("ModelExists should return false when only a different model is present")
	}
}

// ---------------------------------------------------------------------------
// ExtractFiles
// ---------------------------------------------------------------------------

func TestExtractFiles_BasicFiles(t *testing.T) {
	assets := fstest.MapFS{
		"services/lfm25_server/main.py":    {Data: []byte("# main\n")},
		"services/lfm25_server/config.yaml": {Data: []byte("host: localhost\n")},
	}

	dest := t.TempDir()
	if err := setup.ExtractFiles(assets, dest); err != nil {
		t.Fatalf("ExtractFiles: %v", err)
	}

	checkFile(t, dest, "main.py", "# main\n")
	checkFile(t, dest, "config.yaml", "host: localhost\n")
}

func TestExtractFiles_PreservesSubdirectory(t *testing.T) {
	assets := fstest.MapFS{
		"services/lfm25_server/lfm25_server/server.py": {Data: []byte("# server\n")},
	}

	dest := t.TempDir()
	if err := setup.ExtractFiles(assets, dest); err != nil {
		t.Fatalf("ExtractFiles: %v", err)
	}

	checkFile(t, dest, filepath.Join("lfm25_server", "server.py"), "# server\n")
}

func TestExtractFiles_IncludesUnderscorePrefixedFiles(t *testing.T) {
	// __init__.py and __main__.py are required for the Python package; the
	// embed directive uses all: prefix to include them.
	assets := fstest.MapFS{
		"services/lfm25_server/lfm25_server/__init__.py": {Data: []byte("# init\n")},
		"services/lfm25_server/lfm25_server/__main__.py": {Data: []byte("# main\n")},
	}

	dest := t.TempDir()
	if err := setup.ExtractFiles(assets, dest); err != nil {
		t.Fatalf("ExtractFiles: %v", err)
	}

	checkFile(t, dest, filepath.Join("lfm25_server", "__init__.py"), "# init\n")
	checkFile(t, dest, filepath.Join("lfm25_server", "__main__.py"), "# main\n")
}

func TestExtractFiles_SkipsPycacheDirectory(t *testing.T) {
	assets := fstest.MapFS{
		"services/lfm25_server/main.py": {Data: []byte("# main\n")},
		"services/lfm25_server/lfm25_server/__pycache__/server.cpython-311.pyc": {Data: []byte("bytecode")},
	}

	dest := t.TempDir()
	if err := setup.ExtractFiles(assets, dest); err != nil {
		t.Fatalf("ExtractFiles: %v", err)
	}

	// main.py must be extracted.
	checkFile(t, dest, "main.py", "# main\n")

	// __pycache__ directory must NOT be created.
	pycacheDir := filepath.Join(dest, "lfm25_server", "__pycache__")
	if _, err := os.Stat(pycacheDir); !os.IsNotExist(err) {
		t.Errorf("__pycache__ directory should not be extracted; stat error: %v", err)
	}
}

func TestExtractFiles_SkipsPycFiles(t *testing.T) {
	assets := fstest.MapFS{
		"services/lfm25_server/main.py":          {Data: []byte("# main\n")},
		"services/lfm25_server/cached.pyc":        {Data: []byte("bytecode")},
	}

	dest := t.TempDir()
	if err := setup.ExtractFiles(assets, dest); err != nil {
		t.Fatalf("ExtractFiles: %v", err)
	}

	checkFile(t, dest, "main.py", "# main\n")

	if _, err := os.Stat(filepath.Join(dest, "cached.pyc")); !os.IsNotExist(err) {
		t.Error(".pyc files should not be extracted")
	}
}

func TestExtractFiles_CreatesNestedDirectories(t *testing.T) {
	assets := fstest.MapFS{
		"services/lfm25_server/lfm25_server/prompts/lfm2.5/security/check.txt": {
			Data: []byte("prompt text\n"),
		},
	}

	dest := t.TempDir()
	if err := setup.ExtractFiles(assets, dest); err != nil {
		t.Fatalf("ExtractFiles: %v", err)
	}

	checkFile(t, dest,
		filepath.Join("lfm25_server", "prompts", "lfm2.5", "security", "check.txt"),
		"prompt text\n",
	)
}

func TestExtractFiles_OverwritesExistingFiles(t *testing.T) {
	assets := fstest.MapFS{
		"services/lfm25_server/main.py": {Data: []byte("# new content\n")},
	}

	dest := t.TempDir()

	// Write stale content.
	if err := os.WriteFile(filepath.Join(dest, "main.py"), []byte("# old content\n"), 0o644); err != nil {
		t.Fatal(err)
	}

	if err := setup.ExtractFiles(assets, dest); err != nil {
		t.Fatalf("ExtractFiles: %v", err)
	}

	checkFile(t, dest, "main.py", "# new content\n")
}

func TestExtractFiles_EmptyFS(t *testing.T) {
	// An FS with no files under the embed prefix should walk gracefully.
	// WalkDir returns an error when the root path doesn't exist, which
	// ExtractFiles should propagate.
	assets := fstest.MapFS{}

	dest := t.TempDir()
	err := setup.ExtractFiles(assets, dest)
	if err == nil {
		t.Error("ExtractFiles should return an error when the embed prefix does not exist in the FS")
	}
}

// ---------------------------------------------------------------------------
// PromptAndDownloadModel — stdin mocking
// ---------------------------------------------------------------------------

func TestPromptAndDownloadModel_UserDeclines(t *testing.T) {
	dir := t.TempDir()

	// Provide "n" on stdin so the function skips the download.
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.WriteString("n\n"); err != nil {
		t.Fatal(err)
	}
	w.Close()

	orig := os.Stdin
	os.Stdin = r
	t.Cleanup(func() { os.Stdin = orig })

	if err := setup.PromptAndDownloadModel(dir); err != nil {
		t.Errorf("PromptAndDownloadModel(decline) should not return an error: %v", err)
	}

	// No model directory should have been created.
	if setup.ModelExists(dir) {
		t.Error("ModelExists should be false after user declines")
	}
}

func TestPromptAndDownloadModel_UserDeclinesWithCapitalN(t *testing.T) {
	dir := t.TempDir()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.WriteString("N\n"); err != nil {
		t.Fatal(err)
	}
	w.Close()

	orig := os.Stdin
	os.Stdin = r
	t.Cleanup(func() { os.Stdin = orig })

	if err := setup.PromptAndDownloadModel(dir); err != nil {
		t.Errorf("PromptAndDownloadModel(decline) should not return an error: %v", err)
	}
}

func TestPromptAndDownloadModel_UserPressesEnter(t *testing.T) {
	// Default (empty) answer is treated as "no".
	dir := t.TempDir()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := w.WriteString("\n"); err != nil {
		t.Fatal(err)
	}
	w.Close()

	orig := os.Stdin
	os.Stdin = r
	t.Cleanup(func() { os.Stdin = orig })

	if err := setup.PromptAndDownloadModel(dir); err != nil {
		t.Errorf("PromptAndDownloadModel(empty) should not return an error: %v", err)
	}
}

func TestPromptAndDownloadModel_EOFTreatedAsNo(t *testing.T) {
	// If stdin is closed immediately (EOF), treat it as "no".
	dir := t.TempDir()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	w.Close() // immediate EOF

	orig := os.Stdin
	os.Stdin = r
	t.Cleanup(func() { os.Stdin = orig })

	if err := setup.PromptAndDownloadModel(dir); err != nil {
		t.Errorf("PromptAndDownloadModel(EOF) should not return an error: %v", err)
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

func writeVersionFile(t *testing.T, dir, version string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(dir, ".setup_version"), []byte(version), 0o644); err != nil {
		t.Fatalf("writeVersionFile: %v", err)
	}
}

func checkFile(t *testing.T, base, rel, wantContent string) {
	t.Helper()
	path := filepath.Join(base, rel)
	data, err := os.ReadFile(path)
	if err != nil {
		t.Errorf("expected file %q to exist: %v", path, err)
		return
	}
	if string(data) != wantContent {
		t.Errorf("file %q content = %q, want %q", path, data, wantContent)
	}
}
