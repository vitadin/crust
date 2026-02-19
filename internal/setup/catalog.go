// Package setup handles auto-extraction and configuration of the embedded
// LFM25 inference server on first run (macOS only).
package setup

// CatalogModel describes a model available for download from HuggingFace.
type CatalogModel struct {
	ID              string // short ID used in config and CLI, e.g. "lfm2.5"
	DisplayName     string // human-readable name shown in CLI output
	HFRepoID        string // HuggingFace repository ID for download
	DirName         string // local directory name under <serverDir>/models/
	Backend         string // inference backend, e.g. "mlx"
	PromptModelID   string // maps to prompts/<id>/ directory in the server
	SizeDescription string // approximate download size, e.g. "~1.2 GB"
	Default         bool   // if true, downloaded on first-run auto-setup
}

// Catalog lists all models available for download.
// Add new entries here to extend multi-model support.
var Catalog = []CatalogModel{
	{
		ID:              "lfm2.5",
		DisplayName:     "LFM2.5 1.2B Thinking (MLX 8-bit)",
		HFRepoID:        "LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit",
		DirName:         "LFM2.5-1.2B-Thinking-MLX-8bit",
		Backend:         "mlx",
		PromptModelID:   "lfm2.5",
		SizeDescription: "~1.2 GB",
		Default:         true,
	},
	// Future: add more models here
}

// LookupCatalogModel returns the catalog entry for the given ID, or nil if not found.
func LookupCatalogModel(id string) *CatalogModel {
	for i := range Catalog {
		if Catalog[i].ID == id {
			return &Catalog[i]
		}
	}
	return nil
}

// DefaultCatalogModel returns the catalog entry marked as Default == true.
// Falls back to the first entry if none is marked default.
// Returns nil if the catalog is empty.
func DefaultCatalogModel() *CatalogModel {
	for i := range Catalog {
		if Catalog[i].Default {
			return &Catalog[i]
		}
	}
	if len(Catalog) > 0 {
		return &Catalog[0]
	}
	return nil
}

// CatalogIDs returns the list of model IDs in catalog order.
func CatalogIDs() []string {
	ids := make([]string, len(Catalog))
	for i, m := range Catalog {
		ids[i] = m.ID
	}
	return ids
}
