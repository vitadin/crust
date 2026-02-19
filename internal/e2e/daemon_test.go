package e2e

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"

	"github.com/BakeLens/crust/internal/security"
	"github.com/BakeLens/crust/internal/testutils"
	"github.com/BakeLens/crust/internal/types"
)

func TestE2E_BasicProxying(t *testing.T) {
	stack, err := testutils.SetupE2EStack(types.APITypeOpenAICompletion)
	if err != nil {
		t.Fatalf("Failed to setup E2E stack: %v", err)
	}
	defer stack.Close()

	// Send a simple request
	payload := map[string]interface{}{
		"model": "test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "Hello"},
		},
	}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(stack.ProxyServer.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Errorf("Expected status 200, got %d", resp.StatusCode)
	}

	var respData map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&respData); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	if choices, ok := respData["choices"].([]interface{}); !ok || len(choices) == 0 {
		// Log response for debugging
		respBytes, _ := json.MarshalIndent(respData, "", "  ")
		t.Errorf("Invalid response format: %s", string(respBytes))
	}
}

func TestE2E_SelfProtection(t *testing.T) {
	stack, err := testutils.SetupE2EStack(types.APITypeOpenAICompletion)
	if err != nil {
		t.Fatalf("Failed to setup E2E stack: %v", err)
	}
	defer stack.Close()

	// Configure mock upstream to return a malicious tool call
	maliciousToolCall := testutils.OpenAIResponse("read_file", `{"path": "~/.crust/config.yaml"}`)
	stack.MockUpstream.SetResponse("trigger-malicious", maliciousToolCall)

	// Send a request that triggers the malicious tool call
	payload := map[string]interface{}{
		"model": "test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "trigger-malicious"},
		},
	}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(stack.ProxyServer.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	var respData map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&respData); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	
	choices := respData["choices"].([]interface{})
	message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	
	if toolCalls, ok := message["tool_calls"]; ok && toolCalls != nil && len(toolCalls.([]interface{})) > 0 {
		t.Error("Malicious tool call was NOT blocked")
	}
	
	content, _ := message["content"].(string)
	if !testutils.ContainsIgnoreCase(content, "blocked") {
		t.Error("Response does not contain block warning")
	}
}

func TestE2E_LFMProtection(t *testing.T) {
	stack, err := testutils.SetupE2EStack(types.APITypeOpenAICompletion)
	if err != nil {
		t.Fatalf("Failed to setup E2E stack: %v", err)
	}
	defer stack.Close()

	tests := []struct {
		name    string
		tool    string
		args    string
		keyword string
	}{
		{
			name:    "Block reading LFM config",
			tool:    "read_file",
			args:    `{"path": "~/.crust/lfm25_server/config.yaml"}`,
			keyword: "trigger-lfm-config",
		},
		{
			name:    "Block deleting LFM model",
			tool:    "delete_file",
			args:    `{"path": "~/.crust/lfm25_server/models/lfm2.5/model.safetensors"}`,
			keyword: "trigger-lfm-model",
		},
		{
			name:    "Block reading LFM prompts",
			tool:    "list_directory",
			args:    `{"path": "~/.crust/lfm25_server/lfm25_server/prompts"}`,
			keyword: "trigger-lfm-prompts",
		},
		{
			name:    "Block LFM SSRF",
			tool:    "fetch_url",
			args:    `{"url": "http://localhost:8765/control/unload"}`,
			keyword: "trigger-lfm-ssrf",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			maliciousToolCall := testutils.OpenAIResponse(tt.tool, tt.args)
			stack.MockUpstream.SetResponse(tt.keyword, maliciousToolCall)

			payload := map[string]interface{}{
				"model": "test-model",
				"messages": []map[string]interface{}{
					{"role": "user", "content": tt.keyword},
				},
			}
			body, _ := json.Marshal(payload)

			resp, err := http.Post(stack.ProxyServer.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
			if err != nil {
				t.Fatalf("Request failed: %v", err)
			}
			defer resp.Body.Close()

			var respData map[string]interface{}
			if err := json.NewDecoder(resp.Body).Decode(&respData); err != nil {
				t.Fatalf("Failed to decode response: %v", err)
			}
			
			choices := respData["choices"].([]interface{})
			message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
			
			if toolCalls, ok := message["tool_calls"]; ok && toolCalls != nil && len(toolCalls.([]interface{})) > 0 {
				t.Errorf("%s: Tool call was NOT blocked", tt.name)
			}
		})
	}
}

func TestE2E_UnicodeBypass(t *testing.T) {
	stack, err := testutils.SetupE2EStack(types.APITypeOpenAICompletion)
	if err != nil {
		t.Fatalf("Failed to setup E2E stack: %v", err)
	}
	defer stack.Close()

	// Fullwidth characters for ~/.crust/config.yaml
	// ～／．ｃｒｕｓｔ／ｃｏｎｆｉｇ．ｙａｍｌ
	fullwidthPath := "\uff5e\uff0f\uff0e\uff43\uff52\uff55\uff53\uff54\uff0f\uff43\uff4f\uff4e\uff46\uff49\uff47\uff0e\uff59\uff41\uff4d\uff4c"

	maliciousToolCall := testutils.OpenAIResponse("read_file", `{"path": "`+fullwidthPath+`"}`)
	stack.MockUpstream.SetResponse("trigger-unicode", maliciousToolCall)

	payload := map[string]interface{}{
		"model": "test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "trigger-unicode"},
		},
	}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(stack.ProxyServer.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	var respData map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&respData); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	
	choices := respData["choices"].([]interface{})
	message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	
	if toolCalls, ok := message["tool_calls"]; ok && toolCalls != nil && len(toolCalls.([]interface{})) > 0 {
		t.Error("Unicode bypass (fullwidth) was NOT blocked")
	}
}

func TestE2E_AnthropicInterception(t *testing.T) {
	stack, err := testutils.SetupE2EStack(types.APITypeAnthropic)
	if err != nil {
		t.Fatalf("Failed to setup E2E stack: %v", err)
	}
	defer stack.Close()

	// Configure mock upstream to return an Anthropic-style tool use
	anthropicToolCall := testutils.AnthropicResponse("read_file", map[string]interface{}{
		"path": "~/.crust/config.yaml",
	})
	stack.MockUpstream.SetResponse("trigger-anthropic", anthropicToolCall)

	// Send an Anthropic-style request (using /v1/messages)
	payload := map[string]interface{}{
		"model": "test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "trigger-anthropic"},
		},
	}
	body, _ := json.Marshal(payload)

	// Proxy detects Anthropic format by URL path
	resp, err := http.Post(stack.ProxyServer.URL+"/v1/messages", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	var respData map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&respData); err != nil {
		t.Fatalf("Failed to decode response: %v", err)
	}
	
	content := respData["content"].([]interface{})
	
	// Verify tool_use block was removed or replaced
	foundToolUse := false
	for _, block := range content {
		b := block.(map[string]interface{})
		if b["type"] == "tool_use" {
			foundToolUse = true
			break
		}
	}
	
	if foundToolUse {
		t.Error("Anthropic tool_use was NOT blocked")
	}
	
	// Check for warning message
	foundWarning := false
	for _, block := range content {
		b := block.(map[string]interface{})
		if b["type"] == "text" && testutils.ContainsIgnoreCase(b["text"].(string), "blocked") {
			foundWarning = true
			break
		}
	}
	
	if !foundWarning {
		t.Error("Response does not contain block warning")
	}
}

func TestE2E_StreamingInterception(t *testing.T) {
	stack, err := testutils.SetupE2EStack(types.APITypeOpenAICompletion)
	if err != nil {
		t.Fatalf("Failed to setup E2E stack: %v", err)
	}
	defer stack.Close()

	// Enable buffered streaming in manager config (manually since stack is already initialized)
	stack.Manager.Shutdown(context.Background())
	managerCfg := security.Config{
		DBPath:          stack.Config.Storage.DBPath,
		SecurityEnabled: true,
		BufferStreaming: true, // Enable buffering
		MaxBufferSize:   100,
		BufferTimeout:   5,
	}
	stack.Manager, _ = security.Init(managerCfg)

	// Configure mock upstream to return a streaming malicious tool call
	maliciousStream := testutils.OpenAIStreamResponse("read_file", `{"path": "~/.crust/config.yaml"}`)
	stack.MockUpstream.SetResponse("trigger-stream", maliciousStream)

	// Send a streaming request
	payload := map[string]interface{}{
		"model": "test-model",
		"messages": []map[string]interface{}{
			{"role": "user", "content": "trigger-stream"},
		},
		"tools": []map[string]interface{}{
			{
				"type": "function",
				"function": map[string]interface{}{
					"name": "read_file",
					"parameters": map[string]interface{}{
						"type": "object",
						"properties": map[string]interface{}{
							"path": map[string]interface{}{"type": "string"},
						},
					},
				},
			},
		},
		"stream": true,
	}
	body, _ := json.Marshal(payload)

	resp, err := http.Post(stack.ProxyServer.URL+"/v1/chat/completions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("Request failed: %v", err)
	}
	defer resp.Body.Close()

	// Read SSE response
	respBody, _ := io.ReadAll(resp.Body)
	t.Logf("Streaming response: %s", string(respBody))
	
	// Verify tool_calls delta was removed or replaced in the stream
	if bytes.Contains(respBody, []byte("\"tool_calls\"")) && bytes.Contains(respBody, []byte("\"read_file\"")) {
		// Note: The buffered writer might keep tool_calls field but it should be empty or replaced
		// In "remove" mode (default), it removes the tool_calls.
		t.Error("Streaming tool call was NOT blocked")
	}
	
	if !bytes.Contains(respBody, []byte("blocked")) {
		t.Error("Streaming response does not contain block warning")
	}
}
