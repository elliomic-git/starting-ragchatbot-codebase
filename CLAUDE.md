# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run

```bash
# Quick start
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

- Web UI: http://localhost:8000
- API docs: http://localhost:8000/docs

Environment: Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`.

## Architecture

### Query Flow
User Query → FastAPI (`app.py`) → RAGSystem → AIGenerator (Claude) → Tool Use → VectorStore Search → Response

### Key Components

**RAGSystem** (`backend/rag_system.py`): Main orchestrator
- `query(query, session_id)` coordinates the RAG pipeline
- `add_course_folder(path)` loads docs with duplicate detection via course title matching

**AIGenerator** (`backend/ai_generator.py`): Claude integration with tool use
- Claude decides when to search (no hardcoded heuristics)
- Two-step process: initial request → tool execution → final answer
- System prompt guides Claude to search only for course-specific questions

**VectorStore** (`backend/vector_store.py`): ChromaDB wrapper
- Two collections: `course_catalog` (metadata), `course_content` (chunks)
- Course name filtering uses semantic search for fuzzy matching (e.g., "MCP" matches full title)
- Embedding: `all-MiniLM-L6-v2`

**DocumentProcessor** (`backend/document_processor.py`): Parsing & chunking
- Parses structured format: `Course Title:`, `Lesson N:` headers
- Chunks: 800 chars, 100 overlap, sentence-aware splitting

**SessionManager** (`backend/session_manager.py`): Conversation state
- Max 4 messages (2 exchanges) per session
- Auto-generates session IDs

### API Endpoints

```
POST /api/query    { "query": "...", "session_id": "..." } → { "answer", "sources", "session_id" }
GET  /api/courses  → { "total_courses", "course_titles" }
```

## Configuration (`backend/config.py`)

- Model: `claude-sonnet-4-20250514` (temperature=0)
- Chunk size: 800, overlap: 100
- Max search results: 5
- ChromaDB path: `./chroma_db`

## Document Format

```
Course Title: ...
Course Link: ...
Course Instructor: ...

Lesson 0: Title
Lesson Link: ...
Content...
```
