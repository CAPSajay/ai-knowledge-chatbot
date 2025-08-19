const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

// Helper function to handle API responses
async function handleResponse(response) {
  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }
  return response.json();
}

// Session management
export async function createSession() {
  const response = await fetch(`${API}/sessions`, {
    method: "POST",
  });
  return handleResponse(response);
}

export async function listSessions() {
  const response = await fetch(`${API}/sessions`);
  return handleResponse(response);
}

export async function deleteSession(sessionId) {
  const response = await fetch(`${API}/sessions/${sessionId}`, {
    method: "DELETE",
  });
  return handleResponse(response);
}

// Document management
export async function fetchDocuments(sessionId) {
  const response = await fetch(`${API}/documents?session_id=${sessionId}`);
  return handleResponse(response);
}

export async function uploadFiles(files, sessionId) {
  const results = [];

  for (const file of files) {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch(`${API}/upload?session_id=${sessionId}`, {
      method: "POST",
      body: formData,
    });

    const result = await handleResponse(response);
    results.push(result);
  }

  return results;
}

export async function deleteDocument(filename, sessionId) {
  const response = await fetch(
    `${API}/documents/${encodeURIComponent(filename)}?session_id=${sessionId}`,
    {
      method: "DELETE",
    }
  );
  return handleResponse(response);
}

// Chat
export async function sendChat(message, history, sessionId) {
  const response = await fetch(`${API}/chat?session_id=${sessionId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      history,
      conversation_id: sessionId,
    }),
  });
  return handleResponse(response);
}
