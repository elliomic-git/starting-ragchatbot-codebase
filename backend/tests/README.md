# Backend Tests

This directory contains the test suite for the RAG Chatbot backend.

## Quick Start

```bash
# Install dev dependencies (from project root)
uv sync --dev

# Run all tests (unit + integration)
uv run pytest tests/ -v

# Run only unit tests
uv run pytest tests/unit/ -v

# Run only integration tests
uv run pytest tests/integration/ -v

# Run with coverage
uv run pytest tests/ --cov=backend --cov-report=term-missing
```

## Test Structure

```
tests/
├── README.md                      # This file
├── TEST_SUMMARY.md               # Detailed test coverage report
├── conftest.py                   # Shared fixtures and mocks
├── pytest.ini                    # Pytest configuration
├── fixtures/                     # Test data (future)
├── unit/                         # Unit tests (115 tests)
│   ├── test_session_manager.py  # Session/conversation tests
│   ├── test_document_processor.py # Document parsing tests
│   ├── test_vector_store.py     # ChromaDB integration tests
│   ├── test_ai_generator.py     # Anthropic API tests
│   ├── test_search_tools.py     # Tool system tests
│   └── test_rag_system.py       # Main orchestrator tests
└── integration/                  # Integration tests (future)
```

## Running Tests

### By Category
```bash
# Run only unit tests
uv run pytest backend/tests/unit/ -v

# Run only integration tests (when implemented)
uv run pytest backend/tests/integration/ -v

# Run tests with specific marker
uv run pytest -m unit -v
```

### By Component
```bash
# Test a specific component
uv run pytest backend/tests/unit/test_session_manager.py -v

# Test a specific class
uv run pytest backend/tests/unit/test_rag_system.py::TestRAGSystem -v

# Test a specific method
uv run pytest backend/tests/unit/test_rag_system.py::TestRAGSystem::test_query_with_session -v
```

### With Coverage
```bash
# Coverage report in terminal
uv run pytest backend/tests/unit/ --cov=backend --cov-report=term-missing

# HTML coverage report
uv run pytest backend/tests/unit/ --cov=backend --cov-report=html
# Then open htmlcov/index.html in browser
```

### Other Options
```bash
# Show print statements
uv run pytest backend/tests/unit/ -v -s

# Stop on first failure
uv run pytest backend/tests/unit/ -x

# Run only failed tests from last run
uv run pytest backend/tests/unit/ --lf

# Run in parallel (faster)
uv run pytest backend/tests/unit/ -n auto  # requires pytest-xdist
```

## Test Statistics

- **Total Tests**: 139 (115 unit + 24 integration)
- **Unit Test Coverage**: 93%
- **All Tests Passing**: ✅ 139/139
- **Unit Test Execution**: ~2-4 seconds
- **Integration Test Execution**: ~5 seconds
- **Total Execution Time**: ~5 seconds

## Component Coverage

### Unit Tests
| Component | Tests | Coverage |
|-----------|-------|----------|
| SessionManager | 17 | 100% |
| AIGenerator | 17 | 100% |
| RAGSystem | 18 | 100% |
| DocumentProcessor | 22 | 99% |
| SearchTools | 27 | 97% |
| VectorStore | 21 | 75% |

### Integration Tests
| Test Suite | Tests | Focus |
|------------|-------|-------|
| RAG Pipeline | 4 | End-to-end document flow |
| API Endpoints | 14 | FastAPI REST API |
| Multi-Document | 6 | Multiple course scenarios |

## Writing New Tests

### Test File Template

```python
import pytest
from unittest.mock import Mock, patch

@pytest.mark.unit
class TestYourComponent:
    """Unit tests for YourComponent"""

    def test_basic_functionality(self):
        """Test description"""
        # Arrange
        component = YourComponent()

        # Act
        result = component.do_something()

        # Assert
        assert result == expected
```

### Using Fixtures

Common fixtures are defined in `conftest.py`:
- `sample_course` - Sample Course object
- `sample_course_chunks` - Sample CourseChunk list
- `sample_course_document` - Sample document content
- `mock_config` - Mock configuration object
- `mock_chroma_client` - Mock ChromaDB client
- `mock_anthropic_client` - Mock Anthropic client

Example:
```python
def test_with_fixture(self, sample_course, mock_config):
    """Test using fixtures"""
    rag = RAGSystem(mock_config)
    # Use sample_course in test
```

### Mocking External Services

```python
@patch('module.ExternalService')
def test_with_mock(self, mock_service):
    """Test with mocked service"""
    mock_service.return_value.method.return_value = "result"
    # Test code
```

## Continuous Integration

Tests should be run in CI/CD pipeline:

```yaml
# Example GitHub Actions
- name: Run tests
  run: |
    uv sync --dev
    uv run pytest backend/tests/unit/ -v --cov=backend
```

## Troubleshooting

### Import Errors
If you get import errors, make sure you're running tests with `uv run`:
```bash
# Wrong - may fail with imports
pytest backend/tests/unit/

# Correct
uv run pytest backend/tests/unit/
```

### ChromaDB Warnings
You may see ChromaDB warnings during tests - these are expected and filtered in the backend code but may appear in tests.

### Slow Tests
If tests are slow:
1. Check for real API calls (should all be mocked)
2. Check for large file operations
3. Use pytest-xdist for parallel execution

## Documentation

- **[README.md](./README.md)** - This file, quick start guide
- **[TEST_SUMMARY.md](./TEST_SUMMARY.md)** - Comprehensive unit test coverage report
- **[INTEGRATION_TEST_SUMMARY.md](./INTEGRATION_TEST_SUMMARY.md)** - Integration test documentation

## Future Work

### Performance Tests
Planned performance testing:
- Load testing with many documents
- Concurrent query handling
- Memory usage profiling
- Large document processing benchmarks

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [Pytest-cov](https://pytest-cov.readthedocs.io/)
- [TEST_SUMMARY.md](./TEST_SUMMARY.md) - Detailed coverage report
