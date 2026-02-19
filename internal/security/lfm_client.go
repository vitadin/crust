package security

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// LFMCheckRequest is the JSON body sent to /security/check.
type LFMCheckRequest struct {
	ToolName  string          `json:"tool_name"`
	Arguments json.RawMessage `json:"arguments"`
	TraceID   string          `json:"trace_id,omitempty"`
	SessionID string          `json:"session_id,omitempty"`
	Model     string          `json:"model,omitempty"`
}

// LFMCheckResponse is the JSON response from /security/check.
type LFMCheckResponse struct {
	Verdict  string `json:"verdict"`  // "allow" or "block"
	Reason   string `json:"reason"`
	Thinking string `json:"thinking,omitempty"`
}

// LFMClient sends tool calls to the LFM25 security check endpoint.
// A nil LFMClient means the feature is disabled.
type LFMClient struct {
	endpoint string
	failOpen bool
	client   *http.Client
}

// NewLFMClient creates a new LFMClient with the given endpoint, timeout, and fail-open behavior.
func NewLFMClient(endpoint string, timeoutMs int, failOpen bool) *LFMClient {
	return &LFMClient{
		endpoint: endpoint,
		failOpen: failOpen,
		client: &http.Client{
			Timeout: time.Duration(timeoutMs) * time.Millisecond,
		},
	}
}

// CheckToolCall calls /security/check and returns (allow bool, reason string).
// On any error (network, timeout, bad parse): returns failOpen value as allow.
func (c *LFMClient) CheckToolCall(toolName string, arguments json.RawMessage, traceID, sessionID, model string) (allow bool, reason string) {
	req := LFMCheckRequest{
		ToolName:  toolName,
		Arguments: arguments,
		TraceID:   traceID,
		SessionID: sessionID,
		Model:     model,
	}

	body, err := json.Marshal(req)
	if err != nil {
		return c.failOpen, fmt.Sprintf("LFM check failed: marshal error: %v", err)
	}

	resp, err := c.client.Post(c.endpoint+"/security/check", "application/json", bytes.NewReader(body)) //nolint:noctx
	if err != nil {
		return c.failOpen, fmt.Sprintf("LFM check failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return c.failOpen, fmt.Sprintf("LFM check failed: HTTP %d", resp.StatusCode)
	}

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return c.failOpen, fmt.Sprintf("LFM check failed: read error: %v", err)
	}

	var checkResp LFMCheckResponse
	if err := json.Unmarshal(respBody, &checkResp); err != nil {
		return c.failOpen, fmt.Sprintf("LFM check failed: parse error: %v", err)
	}

	verdict := strings.ToLower(strings.TrimSpace(checkResp.Verdict))
	if verdict == "block" {
		return false, checkResp.Reason
	}
	// Anything other than "block" is treated as allow
	return true, checkResp.Reason
}
