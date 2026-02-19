package security

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// helper: build a JSON response body matching LFMCheckResponse
func jsonVerdict(verdict, reason string) string {
	b, _ := json.Marshal(map[string]string{"verdict": verdict, "reason": reason})
	return string(b)
}

func TestCheckToolCall_Allow(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(jsonVerdict("allow", "safe")))
	}))
	defer srv.Close()

	c := NewLFMClient(srv.URL, 2000, false)
	allow, reason := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if !allow {
		t.Errorf("expected allow=true, got false (reason: %q)", reason)
	}
}

func TestCheckToolCall_Block(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(jsonVerdict("block", "exfil")))
	}))
	defer srv.Close()

	c := NewLFMClient(srv.URL, 2000, false)
	allow, reason := c.CheckToolCall("exec", json.RawMessage(`{}`), "", "", "")
	if allow {
		t.Errorf("expected allow=false, got true (reason: %q)", reason)
	}
	if reason != "exfil" {
		t.Errorf("expected reason=%q, got %q", "exfil", reason)
	}
}

func TestCheckToolCall_NetworkError_FailOpen(t *testing.T) {
	// Use a server that is immediately closed so connection is refused.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close()

	c := NewLFMClient(srv.URL, 500, true)
	allow, _ := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if !allow {
		t.Error("expected fail-open (allow=true) on network error")
	}
}

func TestCheckToolCall_NetworkError_FailClosed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close()

	c := NewLFMClient(srv.URL, 500, false)
	allow, _ := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if allow {
		t.Error("expected fail-closed (allow=false) on network error")
	}
}

func TestCheckToolCall_Timeout_FailOpen(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond) // longer than timeout
	}))
	defer srv.Close()

	c := NewLFMClient(srv.URL, 50, true) // 50 ms timeout
	allow, _ := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if !allow {
		t.Error("expected fail-open (allow=true) on timeout")
	}
}

func TestCheckToolCall_Timeout_FailClosed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(200 * time.Millisecond)
	}))
	defer srv.Close()

	c := NewLFMClient(srv.URL, 50, false)
	allow, _ := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if allow {
		t.Error("expected fail-closed (allow=false) on timeout")
	}
}

func TestCheckToolCall_HTTP500_FailClosed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal error", http.StatusInternalServerError)
	}))
	defer srv.Close()

	c := NewLFMClient(srv.URL, 2000, false)
	allow, _ := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if allow {
		t.Error("expected fail-closed (allow=false) on HTTP 500")
	}
}

func TestCheckToolCall_BadJSON_FailClosed(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`not valid json`))
	}))
	defer srv.Close()

	c := NewLFMClient(srv.URL, 2000, false)
	allow, _ := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if allow {
		t.Error("expected fail-closed (allow=false) on bad JSON response")
	}
}

func TestCheckToolCall_UnknownVerdict(t *testing.T) {
	// Any verdict that is not "block" is treated as allow.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(jsonVerdict("maybe", "")))
	}))
	defer srv.Close()

	c := NewLFMClient(srv.URL, 2000, false)
	allow, _ := c.CheckToolCall("read_file", json.RawMessage(`{}`), "", "", "")
	if !allow {
		t.Error("expected allow=true for unknown (non-block) verdict")
	}
}
