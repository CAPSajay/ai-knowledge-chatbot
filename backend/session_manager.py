# backend/utils/session_manager.py
import os
import uuid
import shutil
from pathlib import Path

BASE_UPLOAD_DIR = "uploads"
BASE_FAISS_DIR = "faiss_index"
BASE_METADATA_DIR = "metadata"

# Ensure directories exist
for folder in [BASE_UPLOAD_DIR, BASE_FAISS_DIR, BASE_METADATA_DIR]:
    Path(folder).mkdir(parents=True, exist_ok=True)


def create_session():
    """Create a new chat session with unique session_id."""
    session_id = str(uuid.uuid4())

    # Make directories for uploads, FAISS, metadata
    Path(f"{BASE_UPLOAD_DIR}/{session_id}").mkdir(parents=True, exist_ok=True)

    # File paths
    faiss_path = f"{BASE_FAISS_DIR}/{session_id}.index"
    metadata_path = f"{BASE_METADATA_DIR}/document_metadata_{session_id}.json"

    return {
        "session_id": session_id,
        "faiss_path": faiss_path,
        "metadata_path": metadata_path,
        "upload_dir": f"{BASE_UPLOAD_DIR}/{session_id}"
    }


def delete_session(session_id: str):
    """Delete an entire session with files, index, metadata."""
    upload_dir = f"{BASE_UPLOAD_DIR}/{session_id}"
    faiss_path = f"{BASE_FAISS_DIR}/{session_id}.index"
    metadata_path = f"{BASE_METADATA_DIR}/document_metadata_{session_id}.json"

    # Remove files and directories if they exist
    if os.path.exists(upload_dir):
        shutil.rmtree(upload_dir)
    if os.path.exists(faiss_path):
        os.remove(faiss_path)
    if os.path.exists(metadata_path):
        os.remove(metadata_path)

    return {"message": f"Session {session_id} deleted successfully"}


def list_sessions():
    """Return list of all session IDs (based on metadata files)."""
    sessions = []
    for f in os.listdir(BASE_METADATA_DIR):
        if f.startswith("document_metadata_") and f.endswith(".json"):
            session_id = f.replace("document_metadata_",
                                   "").replace(".json", "")
            sessions.append(session_id)
    return sessions
