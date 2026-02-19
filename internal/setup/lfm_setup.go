// Package setup handles auto-extraction and configuration of the embedded
// LFM25 inference server on first run (macOS only).
package setup

import (
	"bufio"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	serverSubDir = "lfm25_server"
	versionFile  = ".setup_version"
	embedPrefix  = "services/lfm25_server"
	// ModelName is the directory name of the bundled LFM2.5 model.
	ModelName = "LFM2.5-1.2B-Thinking-MLX-8bit"
)

// DefaultServerDir returns the absolute path to ~/.crust/lfm25_server.
func DefaultServerDir() (string, error) {
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("cannot determine home directory: %w", err)
	}
	return filepath.Join(home, ".crust", serverSubDir), nil
}

// NeedsSetup returns true if the server at serverDir has not been configured
// yet or if the installed version does not match version.
func NeedsSetup(serverDir, version string) bool {
	data, err := os.ReadFile(filepath.Join(serverDir, versionFile))
	if err != nil {
		return true
	}
	return strings.TrimSpace(string(data)) != version
}

// IsServerReady reports whether the server at serverDir has been set up at
// least once (version marker file exists), regardless of version.
func IsServerReady(serverDir string) bool {
	_, err := os.ReadFile(filepath.Join(serverDir, versionFile))
	return err == nil
}

// ExtractFiles walks assets (an fs.FS rooted at embedPrefix) and writes all
// files to destDir, stripping the embedPrefix from each path.
// __pycache__ directories and .pyc files are skipped.
func ExtractFiles(assets fs.FS, destDir string) error {
	return fs.WalkDir(assets, embedPrefix, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}

		rel, err := filepath.Rel(embedPrefix, path)
		if err != nil {
			return err
		}
		if rel == "." {
			return nil
		}

		base := filepath.Base(rel)
		if base == "__pycache__" {
			return fs.SkipDir
		}
		if strings.HasSuffix(base, ".pyc") {
			return nil
		}

		target := filepath.Join(destDir, rel)

		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}

		if err := os.MkdirAll(filepath.Dir(target), 0o755); err != nil {
			return err
		}

		data, err := fs.ReadFile(assets, path)
		if err != nil {
			return fmt.Errorf("reading embedded file %s: %w", path, err)
		}

		if err := os.WriteFile(target, data, 0o644); err != nil {
			return fmt.Errorf("writing %s: %w", target, err)
		}

		return nil
	})
}

// EnsureUV checks that the uv Python package manager is available and installs
// it via the official installer script if not found.
func EnsureUV() error {
	if uvPath() != "" {
		return nil
	}

	fmt.Println("  Installing uv (Python package manager)...")
	cmd := exec.Command("sh", "-c", "curl -LsSf https://astral.sh/uv/install.sh | sh") //nolint:gosec // official installer
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("uv installation failed: %w\n  Install manually: https://docs.astral.sh/uv/", err)
	}

	if uvPath() == "" {
		return fmt.Errorf("uv not found after installation; add ~/.local/bin to PATH and retry")
	}
	fmt.Println("  uv installed")
	return nil
}

// uvPath returns the absolute path to the uv binary, or "" if not found.
func uvPath() string {
	if p, err := exec.LookPath("uv"); err == nil {
		return p
	}
	home, _ := os.UserHomeDir()
	candidates := []string{
		filepath.Join(home, ".local", "bin", "uv"),
		filepath.Join(home, ".cargo", "bin", "uv"),
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			os.Setenv("PATH", filepath.Dir(c)+string(os.PathListSeparator)+os.Getenv("PATH"))
			return c
		}
	}
	return ""
}

// InstallDeps runs `uv sync --frozen` inside serverDir to install Python
// dependencies using the pinned lock file.
func InstallDeps(serverDir string) error {
	uv := uvPath()
	if uv == "" {
		uv = "uv"
	}
	cmd := exec.Command(uv, "sync", "--frozen")
	cmd.Dir = serverDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("uv sync --frozen failed: %w", err)
	}
	return nil
}

// ModelExists reports whether the default LFM2.5 model directory is present.
func ModelExists(serverDir string) bool {
	def := DefaultCatalogModel()
	if def == nil {
		return false
	}
	return ModelExistsByDir(serverDir, def.DirName)
}

// ModelExistsByDir reports whether a specific model directory exists and is non-empty.
func ModelExistsByDir(serverDir, dirName string) bool {
	modelDir := filepath.Join(serverDir, "models", dirName)
	entries, err := os.ReadDir(modelDir)
	return err == nil && len(entries) > 0
}

// DownloadModel downloads the default catalog model.
func DownloadModel(serverDir string) error {
	def := DefaultCatalogModel()
	if def == nil {
		return fmt.Errorf("no default model in catalog")
	}
	return DownloadCatalogModel(serverDir, *def)
}

// DownloadCatalogModel downloads a specific model from the catalog using uv.
func DownloadCatalogModel(serverDir string, model CatalogModel) error {
	uv := uvPath()
	if uv == "" {
		uv = "uv"
	}

	// Ensure the models directory exists
	if err := os.MkdirAll(filepath.Join(serverDir, "models"), 0o755); err != nil {
		return fmt.Errorf("creating models directory: %w", err)
	}

	localDir := filepath.Join("models", model.DirName)
	cmd := exec.Command(uv, "run", "python", "download_model.py",
		"--repo-id", model.HFRepoID,
		"--local-dir", localDir,
	)
	cmd.Dir = serverDir
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("downloading %s: %w", model.ID, err)
	}
	return nil
}

// PromptAndDownloadCatalogModel asks the user whether to download the model
// and, if confirmed, downloads it and registers it in the config.
func PromptAndDownloadCatalogModel(serverDir string, model CatalogModel) error {
	fmt.Println()
	fmt.Printf("  The %s model is required for local inference (%s download).\n", model.DisplayName, model.SizeDescription)
	fmt.Print("  Download now? [y/N] ")

	reader := bufio.NewReader(os.Stdin)
	answer, err := reader.ReadString('\n')
	if err != nil && err != io.EOF {
		return fmt.Errorf("reading input: %w", err)
	}
	answer = strings.TrimSpace(strings.ToLower(answer))

	if answer != "y" {
		fmt.Println()
		fmt.Println("  Skipping model download. To download later, run:")
		fmt.Printf("    crust install-model %s\n", model.ID)
		return nil
	}

	fmt.Printf("  Downloading %s from HuggingFace...\n", model.ID)
	if err := DownloadCatalogModel(serverDir, model); err != nil {
		return err
	}

	// Register in config.yaml
	root, _, err := ReadPyConfig(serverDir)
	if err != nil {
		return fmt.Errorf("updating config: %w", err)
	}

	entry := PyModelEntry{
		ID:            model.ID,
		Backend:       model.Backend,
		ModelPath:     filepath.Join("models", model.DirName), // Relative path
		PromptModelID: model.PromptModelID,
		Enabled:       true,
	}

	if err := AddModelEntry(root, entry); err != nil {
		// Ignore "already exists" error
		if !strings.Contains(err.Error(), "already exists") {
			return fmt.Errorf("adding model to config: %w", err)
		}
	} else {
		if err := WritePyConfig(serverDir, root); err != nil {
			return fmt.Errorf("saving config: %w", err)
		}
	}

	fmt.Println("  Model downloaded and configured successfully")
	return nil
}

// PromptAndDownloadModel acts as a backward-compatible wrapper.
func PromptAndDownloadModel(serverDir string) error {
	def := DefaultCatalogModel()
	if def == nil {
		return fmt.Errorf("no default model")
	}
	return PromptAndDownloadCatalogModel(serverDir, *def)
}

// writeVersionMarker records the current version so subsequent runs skip setup.
func writeVersionMarker(serverDir, version string) error {
	return os.WriteFile(filepath.Join(serverDir, versionFile), []byte(version), 0o644)
}

// RunAutoSetup orchestrates the full first-run setup of the LFM25 server:
// extract embedded files → ensure uv → install deps → optionally download model.
// It is a no-op when the installed version already matches version.
func RunAutoSetup(assets fs.FS, version string) error {
	dir, err := DefaultServerDir()
	if err != nil {
		return err
	}

	if !NeedsSetup(dir, version) {
		return nil
	}

	fmt.Println("Setting up LFM25 inference server...")

	// 1. Extract embedded Python source files.
	fmt.Println("  Extracting server files...")
	if err := ExtractFiles(assets, dir); err != nil {
		return fmt.Errorf("extracting server files: %w", err)
	}

	// 2. Ensure uv is available.
	if err := EnsureUV(); err != nil {
		return err
	}

	// 3. Install Python dependencies.
	fmt.Println("  Installing Python dependencies...")
	if err := InstallDeps(dir); err != nil {
		return err
	}

	// 4. Write version marker — deps are ready even if model download is skipped.
	if err := writeVersionMarker(dir, version); err != nil {
		return fmt.Errorf("writing version marker: %w", err)
	}

	// 5. Prompt for model download if not already present.
	if !ModelExists(dir) {
		if err := PromptAndDownloadModel(dir); err != nil {
			// Non-fatal: user can download with `crust install-model`.
			fmt.Fprintf(os.Stderr, "  Warning: model download failed: %v\n", err)
		}
	}

	fmt.Println()
	fmt.Println("LFM25 server ready.")
	return nil
}
