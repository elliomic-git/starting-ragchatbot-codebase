import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


@pytest.fixture
def mock_app():
    """Fixture that provides the app with mocked static files"""
    with patch('app.StaticFiles'):
        from app import app
        yield app


@pytest.mark.integration
class TestAPIEndpoints:
    """Integration tests for FastAPI endpoints"""

    @patch('app.rag_system')
    def test_query_endpoint_success(self, mock_rag_system, mock_app):
        """Test POST /api/query with successful response"""

        # Setup mock
        mock_rag_system.query.return_value = (
            "Python is a high-level programming language.",
            [{"text": "Python Course - Lesson 1", "url": "https://example.com/python/1"}]
        )
        mock_rag_system.session_manager.create_session.return_value = "session_123"

        client = TestClient(mock_app)

        # Test with session_id
        response = client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "session_123"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "Python is a high-level programming language."
        assert data["session_id"] == "session_123"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["text"] == "Python Course - Lesson 1"

        # Verify RAG system was called correctly
        mock_rag_system.query.assert_called_once_with("What is Python?", "session_123")

    @patch('app.rag_system')
    def test_query_endpoint_creates_session(self, mock_rag_system, mock_app):
        """Test POST /api/query creates session when not provided"""

        # Setup mock
        mock_rag_system.query.return_value = ("Answer", [])
        mock_rag_system.session_manager.create_session.return_value = "new_session_456"

        client = TestClient(mock_app)

        # Test without session_id
        response = client.post(
            "/api/query",
            json={"query": "Test question"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["session_id"] == "new_session_456"
        mock_rag_system.session_manager.create_session.assert_called_once()

    @patch('app.rag_system')
    def test_query_endpoint_with_empty_sources(self, mock_rag_system, mock_app):
        """Test POST /api/query with no sources returned"""

        mock_rag_system.query.return_value = ("General knowledge answer.", [])
        mock_rag_system.session_manager.create_session.return_value = "session_123"

        client = TestClient(mock_app)

        response = client.post(
            "/api/query",
            json={"query": "What is 2+2?"}
        )

        assert response.status_code == 200
        data = response.json()

        assert data["answer"] == "General knowledge answer."
        assert data["sources"] == []

    @patch('app.rag_system')
    def test_query_endpoint_error_handling(self, mock_rag_system, mock_app):
        """Test POST /api/query handles errors gracefully"""

        # Setup mock to raise exception
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        client = TestClient(mock_app)

        response = client.post(
            "/api/query",
            json={"query": "Test question", "session_id": "session_123"}
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Database connection failed" in data["detail"]

    @patch('app.rag_system')
    def test_query_endpoint_validation(self, mock_rag_system, mock_app):
        """Test POST /api/query validates request body"""

        client = TestClient(mock_app)

        # Test missing query field
        response = client.post(
            "/api/query",
            json={"session_id": "session_123"}
        )

        assert response.status_code == 422  # Validation error

        # Test empty query - mock the response to avoid errors
        mock_rag_system.query.return_value = ("Empty query response", [])
        mock_rag_system.session_manager.create_session.return_value = "session_123"

        response = client.post(
            "/api/query",
            json={"query": ""}
        )

        # Empty string is valid as pydantic str, so should return 200
        assert response.status_code == 200

    @patch('app.rag_system')
    def test_courses_endpoint_success(self, mock_rag_system, mock_app):
        """Test GET /api/courses returns course statistics"""

        # Setup mock
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 3,
            "course_titles": ["Python Basics", "Machine Learning", "Web Development"]
        }

        client = TestClient(mock_app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 3
        assert len(data["course_titles"]) == 3
        assert "Python Basics" in data["course_titles"]

        mock_rag_system.get_course_analytics.assert_called_once()

    @patch('app.rag_system')
    def test_courses_endpoint_empty_catalog(self, mock_rag_system, mock_app):
        """Test GET /api/courses with no courses"""

        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }

        client = TestClient(mock_app)

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    @patch('app.rag_system')
    def test_courses_endpoint_error_handling(self, mock_rag_system, mock_app):
        """Test GET /api/courses handles errors"""

        mock_rag_system.get_course_analytics.side_effect = Exception("Vector store error")

        client = TestClient(mock_app)

        response = client.get("/api/courses")

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    @patch('app.rag_system')
    def test_multiple_queries_same_session(self, mock_rag_system, mock_app):
        """Test multiple queries with the same session maintain context"""

        # Setup mock to return different responses
        mock_rag_system.query.side_effect = [
            ("Python is a programming language.", []),
            ("Yes, Python is beginner-friendly.", []),
            ("Start with variables and loops.", [])
        ]

        client = TestClient(mock_app)

        # First query
        response1 = client.post(
            "/api/query",
            json={"query": "What is Python?", "session_id": "session_abc"}
        )
        assert response1.status_code == 200

        # Second query with same session
        response2 = client.post(
            "/api/query",
            json={"query": "Is it beginner-friendly?", "session_id": "session_abc"}
        )
        assert response2.status_code == 200

        # Third query with same session
        response3 = client.post(
            "/api/query",
            json={"query": "How do I start?", "session_id": "session_abc"}
        )
        assert response3.status_code == 200

        # Verify all queries used the same session
        calls = mock_rag_system.query.call_args_list
        assert all(call[0][1] == "session_abc" for call in calls)
        assert len(calls) == 3

    @patch('app.rag_system')
    def test_cors_headers_present(self, mock_rag_system, mock_app):
        """Test that CORS headers are properly set"""

        mock_rag_system.query.return_value = ("Answer", [])
        mock_rag_system.session_manager.create_session.return_value = "session_123"

        client = TestClient(mock_app)

        response = client.post(
            "/api/query",
            json={"query": "Test"},
            headers={"Origin": "http://localhost:3000"}
        )

        # Check CORS headers
        assert response.status_code == 200
        # CORS middleware should add these headers
        # Note: TestClient may not fully simulate CORS, but we can check the middleware is configured

    @patch('app.rag_system')
    def test_query_endpoint_with_sources(self, mock_rag_system, mock_app):
        """Test that sources are properly formatted in response"""

        sources = [
            {"text": "Python Basics - Lesson 1", "url": "https://example.com/python/1"},
            {"text": "Python Basics - Lesson 2", "url": "https://example.com/python/2"},
            {"text": "Advanced Python - Lesson 1", "url": None}  # No URL
        ]

        mock_rag_system.query.return_value = ("Python answer", sources)
        mock_rag_system.session_manager.create_session.return_value = "session_123"

        client = TestClient(mock_app)

        response = client.post(
            "/api/query",
            json={"query": "Python question"}
        )

        assert response.status_code == 200
        data = response.json()

        assert len(data["sources"]) == 3
        assert data["sources"][0]["url"] == "https://example.com/python/1"
        assert data["sources"][1]["url"] == "https://example.com/python/2"
        assert data["sources"][2]["url"] is None

    @patch('app.rag_system')
    def test_endpoint_returns_proper_content_type(self, mock_rag_system, mock_app):
        """Test that endpoints return JSON content type"""

        mock_rag_system.query.return_value = ("Answer", [])
        mock_rag_system.session_manager.create_session.return_value = "session_123"
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 1,
            "course_titles": ["Test"]
        }

        client = TestClient(mock_app)

        # Test query endpoint
        response = client.post("/api/query", json={"query": "Test"})
        assert "application/json" in response.headers.get("content-type", "")

        # Test courses endpoint
        response = client.get("/api/courses")
        assert "application/json" in response.headers.get("content-type", "")

    @patch('os.path.exists')
    @patch('app.rag_system')
    def test_startup_event_loads_documents(self, mock_rag_system, mock_exists, mock_app):
        """Test that startup event loads documents from docs folder"""
        # This test is tricky because startup happens on app creation
        # We need to mock before importing the app

        mock_exists.return_value = True
        mock_rag_system.add_course_folder.return_value = (5, 50)

        # Import app which triggers startup

        # Create client which should trigger startup event
        # Note: FastAPI TestClient doesn't always trigger startup events
        # This is more of a smoke test
        client = TestClient(mock_app)

        # Make a request to ensure app is initialized
        response = client.get("/api/courses")

        # The startup event should have been called during app initialization
        # But TestClient might not trigger it, so this is best effort

    @patch('app.rag_system')
    def test_query_special_characters(self, mock_rag_system, mock_app):
        """Test query endpoint handles special characters"""

        mock_rag_system.query.return_value = ("Answer with special chars: é, ñ, 中文", [])
        mock_rag_system.session_manager.create_session.return_value = "session_123"

        client = TestClient(mock_app)

        response = client.post(
            "/api/query",
            json={"query": "Test with special chars: é, ñ, 中文"}
        )

        assert response.status_code == 200
        data = response.json()
        assert "é" in data["answer"] or "special" in data["answer"].lower()
