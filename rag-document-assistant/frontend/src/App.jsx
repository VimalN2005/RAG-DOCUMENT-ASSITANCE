import { useState, useRef, useEffect } from "react";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ── Session helpers ──────────────────────────
// Each browser tab keeps its own session id in localStorage.
// The backend creates a fresh session automatically if none/invalid is sent.
function getSessionId() {
  return localStorage.getItem("docrag_session_id") || "";
}
function setSessionId(id) {
  if (id) localStorage.setItem("docrag_session_id", id);
}
function sessionHeaders(extra = {}) {
  const sid = getSessionId();
  return sid ? { "X-Session-Id": sid, ...extra } : { ...extra };
}

const FileIcon = () => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
    <polyline points="14 2 14 8 20 8"/>
  </svg>
);

const SendIcon = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <line x1="22" y1="2" x2="11" y2="13"/>
    <polygon points="22 2 15 22 11 13 2 9 22 2"/>
  </svg>
);

const UploadIcon = () => (
  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
    <polyline points="16 16 12 12 8 16"/>
    <line x1="12" y1="12" x2="12" y2="21"/>
    <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
  </svg>
);

const BotIcon = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
    <rect x="3" y="11" width="18" height="11" rx="2" ry="2"/>
    <path d="M7 11V7a5 5 0 0 1 10 0v4"/>
    <circle cx="9" cy="16" r="1" fill="white"/>
    <circle cx="15" cy="16" r="1" fill="white"/>
  </svg>
);

export default function App() {
  const [messages, setMessages]     = useState([]);
  const [input, setInput]           = useState("");
  const [uploading, setUploading]   = useState(false);
  const [thinking, setThinking]     = useState(false);
  const [stats, setStats]           = useState(null);
  const [dragOver, setDragOver]     = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const fileRef   = useRef();
  const bottomRef = useRef();

  useEffect(() => {
    fetchStats();
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  async function fetchStats() {
    try {
      const r = await fetch(`${API}/stats`, { headers: sessionHeaders() });
      const d = await r.json();
      if (d.session_id) setSessionId(d.session_id);
      setStats(d);
    } catch {}
  }

  async function handleUpload(file) {
    if (!file) return;
    const allowed = [".pdf", ".txt", ".docx", ".doc"];
    const ext = "." + file.name.split(".").pop().toLowerCase();
    if (!allowed.includes(ext)) {
      alert("Only PDF, TXT, DOCX files allowed!");
      return;
    }
    setUploading(true);
    const fd = new FormData();
    fd.append("file", file);
    try {
      const r = await fetch(`${API}/upload`, {
        method: "POST",
        headers: sessionHeaders(),
        body: fd,
      });
      const d = await r.json();
      if (d.session_id) setSessionId(d.session_id);
      if (d.status === "ok") {
        setUploadedFiles(prev => [...prev, file.name]);
        setMessages(prev => [...prev, {
          role: "system",
          text: `✅ "${file.name}" indexed — ${d.chunks} chunks ready.`,
        }]);
        fetchStats();
      } else if (d.status === "skipped") {
        setMessages(prev => [...prev, {
          role: "system",
          text: `ℹ️ "${file.name}" was already indexed.`,
        }]);
      }
    } catch (e) {
      setMessages(prev => [...prev, { role: "system", text: `❌ Upload failed: ${e.message}` }]);
    } finally {
      setUploading(false);
    }
  }

  async function handleSend() {
    const q = input.trim();
    if (!q || thinking) return;
    setInput("");
    setMessages(prev => [...prev, { role: "user", text: q }]);
    setThinking(true);
    try {
      const r = await fetch(`${API}/chat`, {
        method: "POST",
        headers: sessionHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({ question: q, top_k: 5 }),
      });
      const d = await r.json();
      setMessages(prev => [...prev, {
        role: "assistant",
        text: d.answer,
        sources: d.sources,
      }]);
    } catch (e) {
      setMessages(prev => [...prev, {
        role: "assistant",
        text: `Error: ${e.message}`,
        sources: [],
      }]);
    } finally {
      setThinking(false);
    }
  }

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleUpload(file);
  };

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0a0f",
      display: "flex",
      fontFamily: "'IBM Plex Mono', monospace",
      color: "#e2e8f0",
    }}>
      {/* ── Sidebar ── */}
      <div style={{
        width: 260,
        background: "#0f0f1a",
        borderRight: "1px solid #1e2035",
        display: "flex",
        flexDirection: "column",
        padding: "24px 16px",
        gap: 24,
        flexShrink: 0,
      }}>
        {/* Logo */}
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
            <div style={{
              width: 32, height: 32, borderRadius: 8,
              background: "linear-gradient(135deg, #6366f1, #a855f7)",
              display: "flex", alignItems: "center", justifyContent: "center",
              fontSize: 16,
            }}>🔍</div>
            <span style={{ fontWeight: 700, fontSize: 15, letterSpacing: 1 }}>DocRAG</span>
          </div>
          <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, textTransform: "uppercase" }}>
            Document Intelligence
          </div>
        </div>

        {/* Upload zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          onClick={() => fileRef.current?.click()}
          style={{
            border: `2px dashed ${dragOver ? "#6366f1" : "#2a2a3e"}`,
            borderRadius: 10,
            padding: "20px 12px",
            textAlign: "center",
            cursor: "pointer",
            transition: "all .2s",
            background: dragOver ? "#1a1a2e" : "transparent",
          }}
        >
          <input
            ref={fileRef}
            type="file"
            accept=".pdf,.txt,.docx,.doc"
            style={{ display: "none" }}
            onChange={e => handleUpload(e.target.files[0])}
          />
          <div style={{ color: uploading ? "#6366f1" : "#4b5563", marginBottom: 8 }}>
            <UploadIcon />
          </div>
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            {uploading ? "Indexing…" : "Drop file or click"}
          </div>
          <div style={{ fontSize: 10, color: "#374151", marginTop: 4 }}>PDF · TXT · DOCX</div>
        </div>

        {/* Stats */}
        {stats && (
          <div style={{
            background: "#13131f",
            borderRadius: 8,
            padding: 12,
            border: "1px solid #1e2035",
          }}>
            <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, marginBottom: 10, textTransform: "uppercase" }}>Index Stats</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              {[
                ["Chunks", stats.total_chunks],
                ["Docs", stats.documents],
              ].map(([label, val]) => (
                <div key={label} style={{ display: "flex", justifyContent: "space-between", fontSize: 12 }}>
                  <span style={{ color: "#6b7280" }}>{label}</span>
                  <span style={{ color: "#a855f7", fontWeight: 600 }}>{val}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Uploaded files */}
        {uploadedFiles.length > 0 && (
          <div>
            <div style={{ fontSize: 10, color: "#4b5563", letterSpacing: 2, marginBottom: 8, textTransform: "uppercase" }}>Indexed</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
              {uploadedFiles.map((f, i) => (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 6,
                  fontSize: 11, color: "#6366f1", padding: "4px 0",
                }}>
                  <FileIcon />
                  <span style={{ overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{f}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        <div style={{ marginTop: "auto", fontSize: 10, color: "#374151" }}>
          FAISS · sentence-transformers<br/>
          Groq LLaMA3 · FastAPI
        </div>
      </div>

      {/* ── Main Chat ── */}
      <div style={{
        flex: 1,
        display: "flex",
        flexDirection: "column",
        maxWidth: 800,
        margin: "0 auto",
        width: "100%",
      }}>
        {/* Header */}
        <div style={{
          padding: "20px 28px",
          borderBottom: "1px solid #1e2035",
          display: "flex",
          alignItems: "center",
          gap: 12,
        }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: "#10b981",
            boxShadow: "0 0 8px #10b981",
          }} />
          <span style={{ fontSize: 13, color: "#9ca3af" }}>Ask anything about your documents</span>
        </div>

        {/* Messages */}
        <div style={{
          flex: 1,
          overflowY: "auto",
          padding: "28px",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}>
          {messages.length === 0 && (
            <div style={{
              textAlign: "center",
              marginTop: 80,
              color: "#374151",
            }}>
              <div style={{ fontSize: 48, marginBottom: 16 }}>📄</div>
              <div style={{ fontSize: 16, color: "#4b5563", marginBottom: 8 }}>Upload a document to get started</div>
              <div style={{ fontSize: 12, color: "#374151" }}>Drag & drop PDF, TXT, or DOCX in the sidebar</div>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} style={{
              display: "flex",
              flexDirection: msg.role === "user" ? "row-reverse" : "row",
              gap: 12,
              alignItems: "flex-start",
            }}>
              {msg.role !== "system" && (
                <div style={{
                  width: 30, height: 30, borderRadius: 8, flexShrink: 0,
                  background: msg.role === "user"
                    ? "linear-gradient(135deg, #6366f1, #a855f7)"
                    : "#1a1a2e",
                  border: msg.role === "assistant" ? "1px solid #2a2a3e" : "none",
                  display: "flex", alignItems: "center", justifyContent: "center",
                  fontSize: msg.role === "user" ? 12 : 14,
                  color: msg.role === "assistant" ? "#6366f1" : "white",
                }}>
                  {msg.role === "user" ? "V" : <BotIcon />}
                </div>
              )}
              <div style={{
                maxWidth: msg.role === "system" ? "100%" : "75%",
                width: msg.role === "system" ? "100%" : undefined,
              }}>
                {msg.role === "system" ? (
                  <div style={{
                    fontSize: 12, color: "#4b5563",
                    background: "#0f0f1a",
                    borderRadius: 6,
                    padding: "8px 12px",
                    border: "1px solid #1e2035",
                    textAlign: "center",
                  }}>{msg.text}</div>
                ) : (
                  <>
                    <div style={{
                      background: msg.role === "user" ? "linear-gradient(135deg, #312e81, #4c1d95)" : "#13131f",
                      border: msg.role === "assistant" ? "1px solid #1e2035" : "none",
                      borderRadius: msg.role === "user" ? "14px 14px 4px 14px" : "14px 14px 14px 4px",
                      padding: "12px 16px",
                      fontSize: 13,
                      lineHeight: 1.7,
                      whiteSpace: "pre-wrap",
                    }}>
                      {msg.text}
                    </div>
                    {msg.sources?.length > 0 && (
                      <div style={{ display: "flex", gap: 6, marginTop: 6, flexWrap: "wrap" }}>
                        {msg.sources.map((s, j) => (
                          <span key={j} style={{
                            fontSize: 10,
                            padding: "2px 8px",
                            background: "#1a1a2e",
                            border: "1px solid #2a2a3e",
                            borderRadius: 4,
                            color: "#6366f1",
                          }}>📎 {s}</span>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          ))}

          {thinking && (
            <div style={{ display: "flex", gap: 12, alignItems: "flex-start" }}>
              <div style={{
                width: 30, height: 30, borderRadius: 8,
                background: "#1a1a2e",
                border: "1px solid #2a2a3e",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: "#6366f1",
              }}>
                <BotIcon />
              </div>
              <div style={{
                background: "#13131f",
                border: "1px solid #1e2035",
                borderRadius: "14px 14px 14px 4px",
                padding: "14px 18px",
                display: "flex", gap: 6, alignItems: "center",
              }}>
                {[0, 1, 2].map(i => (
                  <div key={i} style={{
                    width: 6, height: 6, borderRadius: "50%",
                    background: "#6366f1",
                    animation: `bounce 1.2s ${i * 0.2}s infinite`,
                  }} />
                ))}
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Input */}
        <div style={{
          padding: "16px 28px 24px",
          borderTop: "1px solid #1e2035",
        }}>
          <div style={{
            display: "flex",
            gap: 10,
            background: "#0f0f1a",
            border: "1px solid #2a2a3e",
            borderRadius: 12,
            padding: "4px 4px 4px 16px",
            alignItems: "center",
          }}>
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && !e.shiftKey && handleSend()}
              placeholder="Ask about your documents…"
              style={{
                flex: 1,
                background: "transparent",
                border: "none",
                outline: "none",
                color: "#e2e8f0",
                fontSize: 13,
                fontFamily: "inherit",
                padding: "10px 0",
              }}
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || thinking}
              style={{
                width: 38, height: 38, borderRadius: 9,
                background: input.trim() && !thinking
                  ? "linear-gradient(135deg, #6366f1, #a855f7)"
                  : "#1a1a2e",
                border: "none",
                cursor: input.trim() && !thinking ? "pointer" : "not-allowed",
                display: "flex", alignItems: "center", justifyContent: "center",
                color: input.trim() && !thinking ? "white" : "#374151",
                transition: "all .2s",
                flexShrink: 0,
              }}
            >
              <SendIcon />
            </button>
          </div>
          <div style={{ fontSize: 10, color: "#374151", textAlign: "center", marginTop: 8 }}>
            Press Enter to send · Answers grounded in your documents
          </div>
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        * { box-sizing: border-box; margin: 0; padding: 0; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: #0a0a0f; }
        ::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 2px; }
        @keyframes bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: .4; }
          40% { transform: translateY(-6px); opacity: 1; }
        }
      `}</style>
    </div>
  );
}
