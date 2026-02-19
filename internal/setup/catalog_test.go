package setup

import (
	"testing"
)

func TestLookupCatalogModel(t *testing.T) {
	// Test existing model
	m := LookupCatalogModel("lfm2.5")
	if m == nil {
		t.Errorf("expected lfm2.5 to be found")
	} else if m.ID != "lfm2.5" {
		t.Errorf("expected ID lfm2.5, got %s", m.ID)
	}

	// Test non-existent model
	m = LookupCatalogModel("non-existent")
	if m != nil {
		t.Errorf("expected nil for non-existent model")
	}
}

func TestDefaultCatalogModel(t *testing.T) {
	m := DefaultCatalogModel()
	if m == nil {
		t.Fatal("expected a default model")
	}
	if !m.Default {
		t.Errorf("expected Default to be true for %s", m.ID)
	}
}

func TestCatalogIDs(t *testing.T) {
	ids := CatalogIDs()
	if len(ids) != len(Catalog) {
		t.Errorf("expected %d IDs, got %d", len(Catalog), len(ids))
	}
	for i, id := range ids {
		if id != Catalog[i].ID {
			t.Errorf("expected ID at index %d to be %s, got %s", i, Catalog[i].ID, id)
		}
	}
}
