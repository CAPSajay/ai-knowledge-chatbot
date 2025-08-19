import React, { useEffect, useRef, useState } from "react";
import {
  fetchDocuments,
  uploadFiles,
  deleteDocument,
  sendChat,
  createSession,
  listSessions,
  deleteSession,
} from "./api";

export default function App() {
  const [sessions, setSessions] = useState([]); // array of {session_id, title}
  const [activeSession, setActiveSession] = useState(null);

  const [docs, setDocs] = useState([]);
  const [messagesBySession, setMessagesBySession] = useState({}); // {sessionId: [{role,text,sources?}]}
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  const fileRef = useRef(null);
  const scroller = useRef(null);

  const messages = messagesBySession[activeSession] || [];
  const activeSessionTitle =
    sessions.find((s) => s.session_id === activeSession)?.title || "New Chat";

  // Helpers
  async function refreshDocs(sessionId) {
    try {
      const data = await fetchDocuments(sessionId);
      setDocs(data);
    } catch (e) {
      console.error(e);
    }
  }

  async function initSessions() {
    try {
      const list = await listSessions();
      if (list.length === 0) {
        const result = await createSession();
        const newSession = {
          session_id: result.session_id,
          title: result.title, // ✅ use backend title directly
        };
        setSessions([newSession]);
        setActiveSession(result.session_id);
      } else {
        setSessions(list);
        setActiveSession(list[0].session_id);
      }
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    initSessions();
  }, []);

  useEffect(() => {
    if (activeSession) {
      refreshDocs(activeSession);
      // ensure message bucket exists
      setMessagesBySession((prev) =>
        prev[activeSession] ? prev : { ...prev, [activeSession]: [] }
      );
    }
  }, [activeSession]);

  useEffect(() => {
    if (scroller.current)
      scroller.current.scrollTop = scroller.current.scrollHeight;
  }, [messages, busy]);

  // Session actions
  async function onNewSession() {
    try {
      const result = await createSession();
      const newSession = {
        session_id: result.session_id,
        title: result.title, // ✅ use backend title directly
      };
      setSessions((s) => [newSession, ...s]); // Add to beginning (newest first)
      setActiveSession(result.session_id);
    } catch (e) {
      console.error(e);
    }
  }

  async function onDeleteSession(sessionId) {
    try {
      await deleteSession(sessionId);
      setSessions((s) => s.filter((x) => x.session_id !== sessionId));
      setMessagesBySession((prev) => {
        const copy = { ...prev };
        delete copy[sessionId];
        return copy;
      });
      if (activeSession === sessionId) {
        const remaining = sessions.filter((x) => x.session_id !== sessionId);
        setActiveSession(remaining[0]?.session_id || null);
      }
    } catch (e) {
      console.error(e);
    }
  }

  // File actions
  async function onUpload(e) {
    const files = Array.from(e.target.files || []);
    if (!files.length || !activeSession) return;
    setBusy(true);
    try {
      await uploadFiles(files, activeSession);
      await refreshDocs(activeSession);
    } catch (e) {
      console.error(e);
    }
    setBusy(false);
    e.target.value = "";
  }

  async function onDeleteFile(name) {
    if (!activeSession) return;
    setBusy(true);
    try {
      await deleteDocument(name, activeSession);
      await refreshDocs(activeSession);
    } catch (e) {
      console.error(e);
    }
    setBusy(false);
  }

  // Chat actions
  async function onSend() {
    const text = input.trim();
    if (!text || !activeSession) return;

    const nextMsgs = [...messages, { role: "user", text }];
    setMessagesBySession((prev) => ({ ...prev, [activeSession]: nextMsgs }));
    setInput("");
    setBusy(true);

    try {
      const resp = await sendChat(
        text,
        nextMsgs.map((m) => ({
          role: m.role === "user" ? "user" : "assistant",
          content: m.text,
        })),
        activeSession
      );

      console.log("Chat response:", resp); // Debug log

      const finalMsgs = [
        ...nextMsgs,
        { role: "bot", text: resp.answer, sources: resp.sources },
      ];
      setMessagesBySession((prev) => ({ ...prev, [activeSession]: finalMsgs }));

      // Update session title if it was changed (first message)
      if (resp.session_title) {
        console.log("Updating session title to:", resp.session_title); // Debug log
        setSessions((prev) => {
          const updated = prev.map((s) =>
            s.session_id === activeSession
              ? { ...s, title: resp.session_title }
              : s
          );
          console.log("Sessions after title update:", updated); // Debug log
          return updated;
        });
      }
    } catch (e) {
      console.error("Chat error:", e);
      const finalMsgs = [
        ...nextMsgs,
        { role: "bot", text: "⚠️ Error contacting server." },
      ];
      setMessagesBySession((prev) => ({ ...prev, [activeSession]: finalMsgs }));
    }
    setBusy(false);
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="side-title">Chat Sessions</div>
        <button
          className="new-chat"
          onClick={onNewSession}
          title="Create new chat"
        >
          + New Chat
        </button>

        <div className="sessions">
          {sessions.length === 0 && (
            <div className="file-card muted">No sessions yet.</div>
          )}
          {sessions.map((session) => (
            <div
              key={session.session_id}
              className={`session ${
                session.session_id === activeSession ? "active" : ""
              }`}
              onClick={() => setActiveSession(session.session_id)}
              title={session.title}
            >
              <span>🗂️ {session.title}</span>
              <button
                className="session-del"
                onClick={(e) => {
                  e.stopPropagation();
                  onDeleteSession(session.session_id);
                }}
                title="Delete session"
              >
                ×
              </button>
            </div>
          ))}
        </div>

        <div className="side-sub">Files in this chat</div>
        <div className="files">
          {docs.length === 0 && (
            <div className="file-card muted">No files uploaded yet.</div>
          )}
          {docs.map((d) => (
            <div className="file-card" key={d.filename}>
              <div className="file-name">📄 {d.filename}</div>
              <div className="file-meta">Pages: {d.page_count ?? "—"}</div>
              <button
                className="file-del"
                onClick={() => onDeleteFile(d.filename)}
                title="Remove"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="main">
        <header className="title">
          DOCBOT{" "}
          {activeSession ? (
            <small style={{ opacity: 0.7 }}>• {activeSessionTitle}</small>
          ) : null}
        </header>

        <section className="chat" ref={scroller}>
          {messages.map((m, i) => (
            <div
              key={i}
              className={`bubble ${m.role === "user" ? "user" : "bot"}`}
            >
              <div>{m.text}</div>
              {m.sources?.length ? (
                <div className="sources">
                  {m.sources.map((s, j) => (
                    <span className="source" key={j}>
                      {s.filename} • p.{s.page}
                    </span>
                  ))}
                </div>
              ) : null}
            </div>
          ))}
          {busy && <div className="typing">Thinking…</div>}
        </section>

        <div className="composer">
          <button
            className="attach"
            onClick={() => fileRef.current?.click()}
            title="Attach files"
            disabled={!activeSession}
          >
            📎
          </button>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept=".pdf,.docx,.txt"
            style={{ display: "none" }}
            onChange={onUpload}
          />
          <input
            className="input"
            placeholder={
              activeSession ? "Type a message…" : "Create a chat to start…"
            }
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") onSend();
            }}
            disabled={!activeSession}
          />
          <button
            className="send"
            onClick={onSend}
            title="Send"
            disabled={!activeSession}
          >
            →
          </button>
        </div>
      </main>
    </div>
  );
}
