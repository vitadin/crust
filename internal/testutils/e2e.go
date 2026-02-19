package testutils

import (
	"context"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/BakeLens/crust/internal/config"
	"github.com/BakeLens/crust/internal/proxy"
	"github.com/BakeLens/crust/internal/rules"
	"github.com/BakeLens/crust/internal/security"
	"github.com/BakeLens/crust/internal/types"
)

// E2EStack holds all components of a full Crust stack for testing
type E2EStack struct {
	Config      *config.Config
	RuleEngine  *rules.Engine
	Manager     *security.Manager
	ProxyServer *httptest.Server
	MockUpstream *MockUpstream
	MockLFM      *MockLFM
	TempDir     string
}

// SetupE2EStack initializes a full Crust stack in a temporary directory
func SetupE2EStack(apiType types.APIType) (*E2EStack, error) {
	tempDir, err := os.MkdirTemp("", "crust-e2e-*")
	if err != nil {
		return nil, err
	}

	// Create subdirectories
	rulesDir := filepath.Join(tempDir, "rules.d")
	os.MkdirAll(rulesDir, 0755)
	lfmDir := filepath.Join(tempDir, "lfm25_server")
	os.MkdirAll(filepath.Join(lfmDir, "models"), 0755)

	// Mock infrastructure
	mockUpstream := NewMockUpstream()
	mockLFM := NewMockLFM("allow")

	// Config
	cfg := config.DefaultConfig()
	cfg.Rules.UserDir = rulesDir
	cfg.Storage.DBPath = filepath.Join(tempDir, "crust.db")
	cfg.Upstream.URL = mockUpstream.URL()
	cfg.LFMSecurity.Enabled = true
	cfg.LFMSecurity.Endpoint = mockLFM.URL()

	// Initialize Rules Engine
	engineCfg := rules.EngineConfig{
		UserRulesDir:   rulesDir,
		DisableBuiltin: false,
		APIPort:        9091,
		ModelPaths:     []string{lfmDir},
	}
	ruleEngine, err := rules.NewEngine(engineCfg)
	if err != nil {
		return nil, err
	}
	rules.SetGlobalEngine(ruleEngine)

	// Initialize Manager
	managerCfg := security.Config{
		DBPath:          cfg.Storage.DBPath,
		SecurityEnabled: true,
		LFMEnabled:      true,
		LFMEndpoint:     mockLFM.URL(),
		LFMTimeoutMs:    1000,
	}
	manager, err := security.Init(managerCfg)
	if err != nil {
		return nil, err
	}

	// Initialize Proxy
	proxyHandler, err := proxy.NewProxy(cfg.Upstream.URL, "", 5*time.Second, nil)
	if err != nil {
		return nil, err
	}

	proxyServer := httptest.NewServer(proxyHandler)

	return &E2EStack{
		Config:       cfg,
		RuleEngine:   ruleEngine,
		Manager:      manager,
		ProxyServer:  proxyServer,
		MockUpstream: mockUpstream,
		MockLFM:      mockLFM,
		TempDir:      tempDir,
	}, nil
}

// Close shuts down the stack and cleans up
func (s *E2EStack) Close() {
	if s.ProxyServer != nil {
		s.ProxyServer.Close()
	}
	if s.Manager != nil {
		s.Manager.Shutdown(context.Background())
	}
	if s.MockUpstream != nil {
		s.MockUpstream.Close()
	}
	if s.MockLFM != nil {
		s.MockLFM.Close()
	}
	os.RemoveAll(s.TempDir)
}

// ContainsIgnoreCase checks if s contains substr (case-insensitive)
func ContainsIgnoreCase(s, substr string) bool {
	return strings.Contains(strings.ToLower(s), strings.ToLower(substr))
}
