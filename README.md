# AI Knowledge Chatbot

A powerful document-based chatbot that allows users to upload documents (PDF, DOCX, TXT) and ask questions about their content. The system uses vector embeddings and semantic search to provide accurate, context-aware responses.

## Features

- 📄 **Multi-format Document Support**: Upload PDF, DOCX, and TXT files
- 🔍 **Semantic Search**: Advanced vector-based document retrieval using FAISS
- 💬 **Conversational AI**: Powered by Google's Gemini AI model
- 📚 **Session Management**: Organize conversations and documents in separate sessions
- 🎯 **Source Attribution**: Every answer includes references to source documents
- 🚀 **Modern UI**: Clean, responsive React frontend with real-time chat interface

## Tech Stack

### Backend

- **FastAPI**: High-performance Python web framework
- **Google Gemini**: Advanced language model for chat responses
- **FAISS**: Facebook AI Similarity Search for vector operations
- **Sentence Transformers**: Text embedding generation
- **PyMuPDF**: PDF text extraction
- **python-docx**: DOCX document processing
- **LangChain**: Text splitting and processing utilities

### Frontend

- **React**: Modern JavaScript UI library
- **Vite**: Fast build tool and development server
- **CSS Grid/Flexbox**: Responsive layout system

## Prerequisites

- Python 3.8+
- Node.js 16+
- Google Gemini API key

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ai-knowledge-chatbot
```

### 2. Backend Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install fastapi uvicorn python-multipart
pip install PyMuPDF python-docx
pip install numpy faiss-cpu
pip install langchain sentence-transformers
pip install google-generativeai
pip install python-dotenv
```

### 3. Frontend Setup

```bash
# Install Node.js dependencies
npm install
```

### 4. Environment Configuration

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
SECRET_KEY=your_secret_key_here
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE_MB=50
ALLOWED_EXTENSIONS=pdf,docx,txt
FAISS_INDEX_PATH=faiss_index
METADATA_PATH=document_metadata.json
UPLOAD_DIR=uploads
```

**Get your Gemini API key:**

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy it to your `.env` file

## Usage

### 1. Start the Backend Server

```bash
# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Start FastAPI server
cd backend
python main.py
```

The API will be available at `http://localhost:8000`

### 2. Start the Frontend Development Server

```bash
# In a new terminal
npm run dev
```

The web interface will be available at `http://localhost:5173`

### 3. Using the Application

1. **Create a Session**: Click "New Chat" to start a new conversation
2. **Upload Documents**: Use the 📎 button to upload PDF, DOCX, or TXT files
3. **Ask Questions**: Type questions about your uploaded documents
4. **View Sources**: Each response includes source references with page numbers
5. **Manage Sessions**: Switch between different chat sessions in the sidebar

## API Endpoints

### Session Management

- `POST /sessions` - Create a new chat session
- `GET /sessions` - List all sessions
- `DELETE /sessions/{session_id}` - Delete a session

### Document Management

- `POST /upload?session_id={id}` - Upload documents to a session
- `GET /documents?session_id={id}` - List documents in a session
- `DELETE /documents/{filename}?session_id={id}` - Remove a document

### Chat

- `POST /chat?session_id={id}` - Send a message and get AI response

### Health Check

- `GET /health` - Check API status

## Configuration

### File Upload Limits

- Maximum file size: 50MB (configurable via `MAX_FILE_SIZE_MB`)
- Supported formats: PDF, DOCX, TXT (configurable via `ALLOWED_EXTENSIONS`)

### Text Processing

- Chunk size: 1000 characters
- Chunk overlap: 200 characters
- Embedding model: `all-MiniLM-L6-v2`

### AI Model

- Default model: `gemini-1.5-flash`
- Context window: Last 6 conversation turns
- Retrieval: Top 4 most relevant document chunks

## Project Structure

```
ai-knowledge-chatbot/
├── backend/
│   ├── main.py              # FastAPI application
│   └── .env                 # Environment variables
├── src/
│   ├── api.js              # API client functions
│   ├── app.jsx             # Main React component
│   ├── main.jsx            # React entry point
│   └── styles/
│       └── style.css       # Application styles
├── sessions/               # Session data storage
├── package.json           # Node.js dependencies
└── README.md             # This file
```

## Development

### Backend Development

```bash
# Run with auto-reload
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
# Run with hot reload
npm run dev
```

### Building for Production

```bash
# Build frontend
npm run build

# Serve built files
npm run preview
```

## Troubleshooting

### Common Issues

1. **"Gemini API key not configured"**

   - Ensure your `.env` file contains a valid `GEMINI_API_KEY`
   - Verify the API key is active in Google AI Studio

2. **File upload fails**

   - Check file size is under the limit (50MB default)
   - Ensure file format is supported (PDF, DOCX, TXT)

3. **No relevant context found**

   - Upload documents to the current session
   - Ensure documents contain text (not just images)

4. **CORS errors**
   - Verify the backend is running on the expected port
   - Check the `VITE_API_URL` environment variable

### Performance Tips

- For large documents, consider splitting them into smaller files
- Use specific, detailed questions for better results
- Upload documents in supported formats for optimal text extraction

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

