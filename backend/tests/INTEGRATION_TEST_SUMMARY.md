# Integration Tests Summary

## Overview
Comprehensive integration test suite for the RAG Chatbot backend testing component interactions and API endpoints.

## Test Statistics
- **Total Integration Tests**: 24
- **All Passing**: ✅ 24/24 (100%)
- **Test Execution Time**: ~5 seconds

## Test Coverage by Category

### 1. RAG Pipeline Integration (4 tests)
**File**: `test_rag_pipeline.py`

Tests the complete flow through multiple components:

#### test_document_ingestion_to_query_flow
- **What it tests**: End-to-end document processing → storage → query → response
- **Components tested**: DocumentProcessor, VectorStore (mocked ChromaDB), AIGenerator (mocked API), RAGSystem
- **Flow**:
  1. Create real course document with lessons
  2. Process and ingest into mocked vector store
  3. Query the system with a question
  4. Verify AI generates response using tools
  5. Confirm data was properly stored and retrieved

#### test_session_continuity_across_queries
- **What it tests**: Conversation history maintained across multiple queries
- **Components tested**: SessionManager, RAGSystem, AIGenerator
- **Verifies**:
  - Session ID persists across queries
  - Conversation history accumulates (respecting MAX_HISTORY limit)
  - History is passed to AI for context
  - Subsequent queries have access to previous context

#### test_error_handling_in_pipeline
- **What it tests**: Graceful error handling throughout pipeline
- **Scenarios**:
  - Invalid document path → returns None, 0
  - Empty document → handles gracefully
  - Missing files → doesn't crash

#### test_chunking_preserves_context
- **What it tests**: Document chunking maintains lesson/course context
- **Verifies**:
  - Long content is split into multiple chunks
  - Each chunk contains course/lesson identifiers
  - Context is preserved across chunk boundaries

---

### 2. FastAPI Endpoint Integration (14 tests)
**File**: `test_api_endpoints.py`

Tests HTTP API endpoints with FastAPI TestClient:

#### POST /api/query Tests (9 tests)

**test_query_endpoint_success**
- Valid query with session_id
- Verifies response structure (answer, sources, session_id)
- Confirms RAG system called correctly

**test_query_endpoint_creates_session**
- Query without session_id
- Verifies new session created automatically

**test_query_endpoint_with_empty_sources**
- Query that returns no sources
- Verifies sources array is empty

**test_query_endpoint_error_handling**
- RAG system raises exception
- Verifies 500 error returned with error details

**test_query_endpoint_validation**
- Missing required query field → 422 validation error
- Empty query string → processes normally (200)

**test_multiple_queries_same_session**
- Multiple queries with same session_id
- Verifies session maintained across requests

**test_cors_headers_present**
- Checks CORS middleware configured
- Origin headers handled

**test_query_endpoint_with_sources**
- Response with multiple sources
- Verifies source formatting (text, url)
- Handles sources with null URLs

**test_query_special_characters**
- Unicode and special characters in query/response
- Verifies UTF-8 handling (é, ñ, 中文)

#### GET /api/courses Tests (3 tests)

**test_courses_endpoint_success**
- Returns total_courses and course_titles
- Verifies analytics structure

**test_courses_endpoint_empty_catalog**
- Empty course catalog
- Returns 0 courses, empty array

**test_courses_endpoint_error_handling**
- Vector store error
- Returns 500 with error details

#### Other Tests (2 tests)

**test_endpoint_returns_proper_content_type**
- Verifies application/json content-type
- Tests both query and courses endpoints

**test_startup_event_loads_documents**
- Smoke test for startup event
- Mocks document loading on app initialization

---

### 3. Multi-Document Scenarios (6 tests)
**File**: `test_multi_document.py`

Tests handling multiple course documents:

#### test_multiple_courses_ingestion
- **What it tests**: Ingesting multiple course documents
- **Verifies**:
  - All courses stored with unique IDs
  - Each course processed independently
  - No interference between courses

#### test_duplicate_course_prevention
- **What it tests**: Duplicate detection when loading folder
- **Scenario**: Folder contains multiple files with same course title
- **Verifies**:
  - Only unique courses are added
  - Duplicates detected by title matching
  - Total count reflects unique courses only

#### test_cross_course_search
- **What it tests**: Searching across multiple courses without filters
- **Verifies**:
  - Search returns results from multiple courses
  - No course filter applied
  - AI can synthesize cross-course information

#### test_course_specific_search
- **What it tests**: Filtering search to specific course
- **Verifies**:
  - Course name resolution works
  - Search limited to specified course
  - Results only from target course

#### test_analytics_with_multiple_courses
- **What it tests**: Getting statistics for multiple courses
- **Verifies**:
  - Accurate course count
  - All course titles returned
  - Analytics reflect actual catalog state

#### test_mixed_file_types_in_folder
- **What it tests**: Folder ingestion with various file types
- **File types tested**:
  - ✅ .txt - processed
  - ✅ .pdf - processed
  - ✅ .docx - processed
  - ❌ .md - skipped
  - ❌ .png - skipped
  - ❌ .csv - skipped
- **Verifies**: Only supported file types processed

---

## Mocking Strategy

### External Services Mocked
1. **ChromaDB** - Full mock of PersistentClient and collections
2. **Anthropic API** - Mock client responses (direct and tool_use flows)
3. **Static Files** - Mock StaticFiles for FastAPI tests

### Components Tested with Real Implementation
- DocumentProcessor (actual text processing and chunking)
- SessionManager (actual conversation history management)
- RAGSystem orchestration logic

### Why This Approach?
- **Fast**: No real API calls or database operations
- **Reliable**: No external dependencies
- **Comprehensive**: Tests actual business logic while mocking I/O
- **Realistic**: Uses real document processing and session management

---

## Running Integration Tests

### Run All Integration Tests
```bash
uv run pytest tests/integration/ -v
```

### Run Specific Test File
```bash
uv run pytest tests/integration/test_rag_pipeline.py -v
uv run pytest tests/integration/test_api_endpoints.py -v
uv run pytest tests/integration/test_multi_document.py -v
```

### Run Specific Test
```bash
uv run pytest tests/integration/test_rag_pipeline.py::TestRAGPipelineIntegration::test_document_ingestion_to_query_flow -v
```

### Run with Coverage
```bash
uv run pytest tests/integration/ --cov=backend --cov-report=term-missing
```

---

## Key Testing Patterns

### 1. Multi-Layer Mocking
```python
@patch('rag_system.CourseSearchTool')
@patch('rag_system.ToolManager')
@patch('ai_generator.anthropic.Anthropic')
@patch('vector_store.chromadb.PersistentClient')
def test_example(self, mock_chroma, mock_anthropic, ...):
    # Setup mocks to simulate entire pipeline
    ...
```

### 2. Real Document Processing
```python
# Create actual course document
course_content = """Course Title: Test
Lesson 1: Introduction
Content here..."""

with tempfile.NamedTemporaryFile(...) as f:
    f.write(course_content)
    # Process with real DocumentProcessor
    course, chunks = rag.add_course_document(f.name)
```

### 3. Tool Use Flow Simulation
```python
# Mock tool use request
tool_block = Mock()
tool_block.type = "tool_use"
tool_block.name = "search_course_content"
tool_block.input = {"query": "test"}

initial_response.content = [tool_block]
initial_response.stop_reason = "tool_use"

# Then mock final response
final_response.content = [Mock(text="Answer", type="text")]
```

### 4. FastAPI Test Client
```python
@pytest.fixture
def mock_app():
    """Fixture that provides app with mocked static files"""
    with patch('app.StaticFiles'):
        from app import app
        yield app

def test_endpoint(mock_app):
    client = TestClient(mock_app)
    response = client.post("/api/query", json={...})
    assert response.status_code == 200
```

---

## Integration Test Characteristics

| Aspect | Details |
|--------|---------|
| **Speed** | ~5 seconds for all 24 tests |
| **Isolation** | Each test independent, no shared state |
| **Realism** | Uses real document processing, mocks I/O |
| **Coverage** | All major user flows covered |
| **Maintenance** | Clear, well-documented test cases |

---

## Test Data

### Sample Course Documents
Located in test methods as strings:
- Well-formed course with multiple lessons
- Malformed documents (missing metadata)
- Empty documents
- Long documents (testing chunking)
- Documents with special characters

### Mock Responses
- Anthropic API responses (tool_use, end_turn)
- ChromaDB query results
- Session management scenarios

---

## Future Enhancements

### Potential Additional Tests
1. **Performance Tests**
   - Large document processing
   - Many concurrent queries
   - Memory usage profiling

2. **Error Recovery Tests**
   - Network failures
   - Partial document processing
   - Database connection issues

3. **Advanced Scenarios**
   - Course updates/versioning
   - Incremental indexing
   - Search ranking validation

---

## Troubleshooting

### Common Issues

**Import Errors**
```bash
# Always use uv run
uv run pytest tests/integration/
```

**Mock Not Applied**
- Check patch decorator order (bottom-up execution)
- Verify patch path matches import path

**Fixture Issues**
- Don't call fixtures directly
- Use as function parameters

**File Paths**
- Use tempfile for test documents
- Clean up temp files in finally blocks

---

## Success Metrics

✅ **24/24 tests passing**
✅ **~5 second execution**
✅ **100% critical paths covered**
✅ **Zero external dependencies**
✅ **Clear, maintainable code**

---

## Resources

- [pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [TEST_SUMMARY.md](./TEST_SUMMARY.md) - Unit tests summary
