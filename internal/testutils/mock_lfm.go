package testutils

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
)

// MockLFM is a simple LFM security check mock server
type MockLFM struct {
	server *httptest.Server
	// verdict to return
	verdict string
}

// NewMockLFM creates a new mock LFM server
func NewMockLFM(verdict string) *MockLFM {
	m := &MockLFM{
		verdict: verdict,
	}
	m.server = httptest.NewServer(http.HandlerFunc(m.handleRequest))
	return m
}

// URL returns the mock server URL
func (m *MockLFM) URL() string {
	return m.server.URL
}

// Close stops the mock server
func (m *MockLFM) Close() {
	m.server.Close()
}

func (m *MockLFM) handleRequest(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/security/check" {
		w.WriteHeader(http.StatusNotFound)
		return
	}

	response := map[string]interface{}{
		"verdict":    m.verdict,
		"risk_level": "low",
		"reason":     "Mock LFM verdict",
	}
	if m.verdict == "block" {
		response["risk_level"] = "high"
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(response)
}
