package setup

import (
	"fmt"
	"os"
	"path/filepath"

	"gopkg.in/yaml.v3"
)

// PyModelEntry mirrors one entry in the Python server's config.yaml "models:" list.
type PyModelEntry struct {
	ID                 string         `yaml:"id"`
	Backend            string         `yaml:"backend"`
	ModelPath          string         `yaml:"model_path"`
	PromptModelID      string         `yaml:"prompt_model_id"`
	Enabled            bool           `yaml:"enabled"`
	GenerationDefaults map[string]any `yaml:"generation_defaults"`
}

// PyConfigPath returns the absolute path to the Python server's config.yaml.
func PyConfigPath(serverDir string) string {
	return filepath.Join(serverDir, "config.yaml")
}

// ReadPyConfig parses the Python server's config.yaml and returns:
//   - root: the yaml.Node document root for comment-preserving round-trips
//   - entries: decoded model entries from the "models:" sequence
func ReadPyConfig(serverDir string) (*yaml.Node, []PyModelEntry, error) {
	path := PyConfigPath(serverDir)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, fmt.Errorf("reading %s: %w", path, err)
	}

	var root yaml.Node
	if err := yaml.Unmarshal(data, &root); err != nil {
		return nil, nil, fmt.Errorf("parsing %s: %w", path, err)
	}

	if root.Kind != yaml.DocumentNode || len(root.Content) == 0 {
		return nil, nil, fmt.Errorf("unexpected YAML structure in %s", path)
	}

	modelsNode := findMappingValue(root.Content[0], "models")
	if modelsNode == nil {
		// No models key — valid but empty.
		return &root, nil, nil
	}
	if modelsNode.Kind != yaml.SequenceNode {
		return nil, nil, fmt.Errorf("'models' key in %s is not a sequence", path)
	}

	entries := make([]PyModelEntry, 0, len(modelsNode.Content))
	for _, child := range modelsNode.Content {
		var entry PyModelEntry
		if err := child.Decode(&entry); err != nil {
			return nil, nil, fmt.Errorf("decoding model entry in %s: %w", path, err)
		}
		entries = append(entries, entry)
	}

	return &root, entries, nil
}

// WritePyConfig writes the modified yaml.Node root back to config.yaml,
// preserving all comments from the original document.
func WritePyConfig(serverDir string, root *yaml.Node) error {
	data, err := yaml.Marshal(root)
	if err != nil {
		return fmt.Errorf("marshaling config: %w", err)
	}
	path := PyConfigPath(serverDir)
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("writing %s: %w", path, err)
	}
	return nil
}

// AddModelEntry appends a new model entry to the "models:" sequence in root.
// Returns an error if a model with the same ID already exists.
func AddModelEntry(root *yaml.Node, entry PyModelEntry) error {
	mappingNode := root
	if mappingNode.Kind == yaml.DocumentNode {
		if len(mappingNode.Content) == 0 {
			return fmt.Errorf("empty YAML document")
		}
		mappingNode = mappingNode.Content[0]
	}

	modelsNode := findMappingValue(mappingNode, "models")
	if modelsNode == nil {
		return fmt.Errorf("config.yaml has no 'models' key")
	}
	if modelsNode.Kind != yaml.SequenceNode {
		return fmt.Errorf("'models' key is not a sequence")
	}

	// Check for duplicate ID.
	for _, child := range modelsNode.Content {
		var existing PyModelEntry
		if err := child.Decode(&existing); err == nil {
			if existing.ID == entry.ID {
				return fmt.Errorf("model %q already exists in config", entry.ID)
			}
		}
	}

	// Marshal the entry to a yaml.Node for appending.
	entryBytes, err := yaml.Marshal(entry)
	if err != nil {
		return fmt.Errorf("encoding model entry: %w", err)
	}
	var entryDoc yaml.Node
	if err := yaml.Unmarshal(entryBytes, &entryDoc); err != nil {
		return fmt.Errorf("re-parsing model entry: %w", err)
	}
	if entryDoc.Kind != yaml.DocumentNode || len(entryDoc.Content) == 0 {
		return fmt.Errorf("unexpected structure when encoding model entry")
	}

	modelsNode.Content = append(modelsNode.Content, entryDoc.Content[0])
	return nil
}

// RemoveModelEntry removes the model entry with the given ID from the "models:"
// sequence in root. Returns an error if the ID is not found.
func RemoveModelEntry(root *yaml.Node, id string) error {
	mappingNode := root
	if mappingNode.Kind == yaml.DocumentNode {
		if len(mappingNode.Content) == 0 {
			return fmt.Errorf("empty YAML document")
		}
		mappingNode = mappingNode.Content[0]
	}

	modelsNode := findMappingValue(mappingNode, "models")
	if modelsNode == nil {
		return fmt.Errorf("config.yaml has no 'models' key")
	}
	if modelsNode.Kind != yaml.SequenceNode {
		return fmt.Errorf("'models' key is not a sequence")
	}

	found := false
	newContent := make([]*yaml.Node, 0, len(modelsNode.Content))
	for _, child := range modelsNode.Content {
		var entry PyModelEntry
		if err := child.Decode(&entry); err == nil && entry.ID == id {
			found = true
			continue // skip — this is the entry to remove
		}
		newContent = append(newContent, child)
	}

	if !found {
		return fmt.Errorf("model %q not found in config", id)
	}

	modelsNode.Content = newContent
	return nil
}

// CollectModelPaths returns the resolved absolute paths of all model_path values
// in the Python server's config.yaml. Relative paths (e.g. "./models/...") are
// resolved relative to serverDir.
func CollectModelPaths(serverDir string) ([]string, error) {
	_, entries, err := ReadPyConfig(serverDir)
	if err != nil {
		return nil, err
	}

	paths := make([]string, 0, len(entries))
	for _, entry := range entries {
		p := entry.ModelPath
		if !filepath.IsAbs(p) {
			p = filepath.Join(serverDir, p)
		}
		abs, err := filepath.Abs(p)
		if err != nil {
			// Skip paths we can't resolve.
			continue
		}
		// Resolve symlinks when possible; fall back to Abs path.
		if resolved, err := filepath.EvalSymlinks(abs); err == nil {
			abs = resolved
		}
		paths = append(paths, abs)
	}
	return paths, nil
}

// findMappingValue returns the value node for a key in a YAML MappingNode,
// or nil if the key is not found or the node is not a mapping.
func findMappingValue(mapping *yaml.Node, key string) *yaml.Node {
	if mapping == nil || mapping.Kind != yaml.MappingNode {
		return nil
	}
	for i := 0; i+1 < len(mapping.Content); i += 2 {
		if mapping.Content[i].Value == key {
			return mapping.Content[i+1]
		}
	}
	return nil
}
