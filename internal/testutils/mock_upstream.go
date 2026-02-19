package testutils

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
)

// MockUpstream is a simple OpenAI/Anthropic compatible mock server
type MockUpstream struct {
	server *httptest.Server
	// Map of prompt keyword to response
	responses map[string]interface{}
}

// NewMockUpstream creates a new mock LLM server
func NewMockUpstream() *MockUpstream {
	m := &MockUpstream{
		responses: make(map[string]interface{}),
	}
	m.server = httptest.NewServer(http.HandlerFunc(m.handleRequest))
	return m
}

// URL returns the mock server URL
func (m *MockUpstream) URL() string {
	return m.server.URL
}

// Close stops the mock server
func (m *MockUpstream) Close() {
	m.server.Close()
}

// SetResponse sets a mock response for a specific keyword found in the request
func (m *MockUpstream) SetResponse(keyword string, response interface{}) {
	m.responses[keyword] = response
}

func (m *MockUpstream) handleRequest(w http.ResponseWriter, r *http.Request) {
	// Read request body to find keyword
	defer r.Body.Close()
	body := make([]byte, 4096)
	n, _ := r.Body.Read(body)
	requestStr := string(body[:n])

	var foundResponse interface{}
	for keyword, resp := range m.responses {
		if strings.Contains(requestStr, keyword) {
			foundResponse = resp
			break
		}
	}

	if foundResponse == nil {
		// Default response: simple assistant message
		foundResponse = map[string]interface{}{
			"id":      "chatcmpl-123",
			"object":  "chat.completion",
			"created": 1677652288,
			"model":   "gpt-3.5-turbo-0613",
			"choices": []map[string]interface{}{
				{
					"index": 0,
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": "Hello! How can I help you today?",
					},
					"finish_reason": "stop",
				},
			},
		}
	}

	// If response is a slice of strings, treat as SSE
	if stream, ok := foundResponse.([]string); ok {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		for _, data := range stream {
			fmt.Fprintf(w, "data: %s\n\n", data)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(foundResponse)
}

// OpenAIStreamResponse returns a slice of data for SSE streaming
func OpenAIStreamResponse(toolName string, arguments string) []string {
	// Properly escape arguments for inclusion in a JSON string
	escapedArgs, _ := json.Marshal(arguments)
	escapedArgsStr := string(escapedArgs)
	// Remove outer quotes added by json.Marshal
	escapedArgsStr = escapedArgsStr[1 : len(escapedArgsStr)-1]

	return []string{
		`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"delta":{"role":"assistant","content":null,"tool_calls":[{"index":0,"id":"call_abc123","type":"function","function":{"name":"` + toolName + `","arguments":"` + escapedArgsStr + `"}}]},"finish_reason":null}]}`,
		`{"id":"chatcmpl-123","object":"chat.completion.chunk","created":1677652288,"model":"gpt-3.5-turbo-0613","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`,
		"[DONE]",
	}
}

// OpenAIResponse returns a mock OpenAI chat completion response with tool calls
func OpenAIResponse(toolName string, arguments string) map[string]interface{} {
	return map[string]interface{}{
		"id":      "chatcmpl-123",
		"object":  "chat.completion",
		"created": 1677652288,
		"model":   "gpt-3.5-turbo-0613",
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": nil,
					"tool_calls": []map[string]interface{}{
						{
							"id":   "call_abc123",
							"type": "function",
							"function": map[string]interface{}{
								"name":      toolName,
								"arguments": arguments,
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
	}
}

// AnthropicResponse returns a mock Anthropic message response with tool use
func AnthropicResponse(toolName string, input map[string]interface{}) map[string]interface{} {
	return map[string]interface{}{
		"id":    "msg_0123456789",
		"type":  "message",
		"role":  "assistant",
		"model": "claude-3-opus-20240229",
		"content": []map[string]interface{}{
			{
				"type":  "tool_use",
				"id":    "toolu_0123456789",
				"name":  toolName,
				"input": input,
			},
		},
		"stop_reason":   "tool_use",
		"stop_sequence": nil,
		"usage": map[string]interface{}{
			"input_tokens":  10,
			"output_tokens": 5,
		},
	}
}
