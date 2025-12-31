# Course Materials RAG System

A Retrieval-Augmented Generation (RAG) system designed to answer questions about course materials using semantic search and AI-powered responses.

## Overview

This application is a full-stack web application that enables users to query course materials and receive intelligent, context-aware responses. It uses ChromaDB for vector storage, Anthropic's Claude for AI generation, and provides a web interface for interaction.

## Technology Stack

| Layer | Technologies |
|-------|-------------|
| **Backend** | Python 3.13+, FastAPI, Uvicorn |
| **AI/ML** | Anthropic Claude (claude-sonnet-4-20250514), Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector DB** | ChromaDB (embedded, persistent) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript, Marked.js |
| **Package Manager** | uv |

## Project Structure

```
├── backend/                    # FastAPI application
│   ├── app.py                 # Main API endpoints & startup
│   ├── rag_system.py          # Core RAG orchestrator
│   ├── vector_store.py        # ChromaDB wrapper for semantic search
│   ├── ai_generator.py        # Claude API integration with tool use
│   ├── document_processor.py  # Document parsing & chunking
│   ├── search_tools.py        # Search tool definitions for Claude
│   ├── session_manager.py     # Conversation history management
│   ├── models.py              # Pydantic data models
│   ├── config.py              # Configuration settings
│   └── chroma_db/             # ChromaDB persistent storage
├── frontend/                   # Web interface
│   ├── index.html             # Main page
│   ├── script.js              # Client-side logic
│   └── style.css              # Styling
├── docs/                       # Course material documents
├── main.py                    # Entry point
├── pyproject.toml             # Dependencies
└── run.sh                     # Startup script
```

## Prerequisites

- Python 3.13 or higher
- uv (Python package manager)
- An Anthropic API key (for Claude AI)
- **For Windows**: Use Git Bash to run the application commands - [Download Git for Windows](https://git-scm.com/downloads/win)

## Installation

1. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Install Python dependencies**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

## Running the Application

### Quick Start

Use the provided shell script:
```bash
chmod +x run.sh
./run.sh
```

### Manual Start

```bash
cd backend
uv run uvicorn app:app --reload --port 8000
```

The application will be available at:
- Web Interface: `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`

## Architecture

### Core Components

#### RAGSystem (`backend/rag_system.py`)
The central orchestrator that coordinates all subsystems:
- Initializes and manages document processor, vector store, AI generator, and session manager
- Loads course documents on startup
- Implements the main `query()` method that processes user questions

#### DocumentProcessor (`backend/document_processor.py`)
Handles document ingestion:
- Parses course files extracting metadata (title, instructor, course link)
- Identifies lessons using `Lesson N: Title` format
- Chunks text into overlapping segments (800 chars with 100 char overlap)

#### VectorStore (`backend/vector_store.py`)
Manages ChromaDB for semantic search:
- **course_catalog** collection: Course metadata for fuzzy name matching
- **course_content** collection: Text chunks with embeddings
- Uses Sentence Transformers (`all-MiniLM-L6-v2`) for embeddings

#### AIGenerator (`backend/ai_generator.py`)
Wraps Anthropic's Claude API:
- Implements tool calling for intelligent search decisions
- Claude autonomously decides when to search vs. answer directly
- Maintains conversation context (max 2 exchanges)

#### SessionManager (`backend/session_manager.py`)
Manages conversation state:
- Creates per-user sessions with unique IDs
- Maintains conversation history for multi-turn conversations

### Document Ingestion Flow

```
docs/*.txt files
       ↓
DocumentProcessor.process_course_document()  → Course object with metadata
       ↓
DocumentProcessor (chunking)                 → List of CourseChunk (800 char segments)
       ↓
VectorStore.add_course_metadata()            → course_catalog collection
VectorStore.add_course_content()             → course_content collection
       ↓
ChromaDB (auto-embeds using Sentence Transformers)
```

### User Query Flow

```
┌─────────┐     ┌──────────────┐     ┌─────────┐     ┌───────────┐     ┌─────────────┐     ┌──────────┐     ┌─────────┐
│  User   │     │   Frontend   │     │ FastAPI │     │ RAGSystem │     │ AIGenerator │     │  Claude  │     │ChromaDB │
│         │     │  script.js   │     │  app.py │     │           │     │             │     │   API    │     │         │
└────┬────┘     └──────┬───────┘     └────┬────┘     └─────┬─────┘     └──────┬──────┘     └────┬─────┘     └────┬────┘
     │                 │                  │                │                  │                 │                │
     │  Type question  │                  │                │                  │                 │                │
     │  + click Send   │                  │                │                  │                 │                │
     │────────────────>│                  │                │                  │                 │                │
     │                 │                  │                │                  │                 │                │
     │                 │ POST /api/query  │                │                  │                 │                │
     │                 │ {query,session_id}                │                  │                 │                │
     │                 │─────────────────>│                │                  │                 │                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │ rag_system.query(query, session_id)                 │                │
     │                 │                  │───────────────>│                  │                 │                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │  get_conversation_history()       │                │
     │                 │                  │                │──────┐           │                 │                │
     │                 │                  │                │      │           │                 │                │
     │                 │                  │                │<─────┘           │                 │                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │ generate_response(query, history, tools)            │
     │                 │                  │                │─────────────────>│                 │                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │                  │ messages.create()               │
     │                 │                  │                │                  │ (with tools)    │                │
     │                 │                  │                │                  │────────────────>│                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │                  │   stop_reason:  │                │
     │                 │                  │                │                  │   "tool_use"    │                │
     │                 │                  │                │                  │<────────────────│                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │                  │ tool_manager.execute_tool()     │
     │                 │                  │                │                  │─────────────────────────────────>│
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │                  │                 │  semantic      │
     │                 │                  │                │                  │                 │  search        │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │                  │   search results (chunks)       │
     │                 │                  │                │                  │<─────────────────────────────────│
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │                  │ messages.create()               │
     │                 │                  │                │                  │ (with tool results)             │
     │                 │                  │                │                  │────────────────>│                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │                  │  final answer   │                │
     │                 │                  │                │                  │<────────────────│                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │                │    response text │                 │                │
     │                 │                  │                │<─────────────────│                 │                │
     │                 │                  │                │                  │                 │                │
     │                 │                  │  (answer, sources)                │                 │                │
     │                 │                  │<───────────────│                  │                 │                │
     │                 │                  │                │                  │                 │                │
     │                 │ {answer, sources, session_id}    │                  │                 │                │
     │                 │<─────────────────│                │                  │                 │                │
     │                 │                  │                │                  │                 │                │
     │  Display answer │                  │                │                  │                 │                │
     │  + sources      │                  │                │                  │                 │                │
     │<────────────────│                  │                │                  │                 │                │
     │                 │                  │                │                  │                 │                │
```

### Query Flow Step-by-Step

| Step | Component | File | What Happens |
|------|-----------|------|--------------|
| 1 | Frontend | `frontend/script.js` | User types question, `sendMessage()` POSTs to `/api/query` |
| 2 | FastAPI | `backend/app.py` | `query_documents()` receives request, creates session if needed |
| 3 | RAGSystem | `backend/rag_system.py` | `query()` gets conversation history, calls AI generator |
| 4 | AIGenerator | `backend/ai_generator.py` | Builds prompt with system instructions, sends to Claude with tools |
| 5 | Claude API | External | Claude decides whether to use search tool based on question |
| 6 | Tool Execution | `backend/ai_generator.py` | If `stop_reason: "tool_use"`, executes search via `tool_manager` |
| 7 | ChromaDB | `backend/vector_store.py` | Semantic search returns relevant course chunks |
| 8 | Claude API | External | Second call with search results generates final answer |
| 9 | Response | `backend/app.py` | Returns `{answer, sources, session_id}` to frontend |
| 10 | Frontend | `frontend/script.js` | `addMessage()` renders response with markdown + sources |

## API Endpoints

### POST `/api/query`
Process a user question and return an AI-generated response.

**Request:**
```json
{
  "query": "What is covered in Lesson 1?",
  "session_id": "session_123"  // optional
}
```

**Response:**
```json
{
  "answer": "Lesson 1 covers...",
  "sources": ["Course Title - Lesson 1"],
  "session_id": "session_123"
}
```

### GET `/api/courses`
Get course catalog statistics.

**Response:**
```json
{
  "total_courses": 4,
  "course_titles": ["Course 1", "Course 2", "Course 3", "Course 4"]
}
```

## Configuration

Configuration is managed in `backend/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `ANTHROPIC_API_KEY` | From `.env` | API key for Claude |
| `ANTHROPIC_MODEL` | `claude-sonnet-4-20250514` | Claude model to use |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `CHUNK_SIZE` | 800 | Characters per text chunk |
| `CHUNK_OVERLAP` | 100 | Overlap between chunks |
| `MAX_RESULTS` | 5 | Search results to return |
| `MAX_HISTORY` | 2 | Conversation exchanges to retain |
| `CHROMA_PATH` | `./chroma_db` | Vector store location |

## Course Document Format

Documents in `docs/` should follow this structure:

```
Course Title: Introduction to Python
Course Link: https://example.com/course
Course Instructor: Jane Doe

Lesson 0: Getting Started
Lesson Link: https://example.com/lesson0
Welcome to the course! In this lesson...

Lesson 1: Variables and Data Types
Lesson Link: https://example.com/lesson1
In this lesson, we'll cover...
```

The system parses this format to extract:
- Course metadata (title, instructor, link)
- Lesson structure (number, title, link)
- Content for each lesson (chunked for embedding)

## Data Models

### Course
```python
class Course(BaseModel):
    title: str              # Unique identifier
    course_link: Optional[str]
    instructor: Optional[str]
    lessons: List[Lesson]
```

### Lesson
```python
class Lesson(BaseModel):
    lesson_number: int
    title: str
    lesson_link: Optional[str]
```

### CourseChunk
```python
class CourseChunk(BaseModel):
    content: str            # Text content
    course_title: str       # Parent course
    lesson_number: Optional[int]
    chunk_index: int        # Position in sequence
```

