import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models import Course, Lesson, CourseChunk


@pytest.fixture
def sample_course():
    """Create a sample Course object for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(lesson_number=0, title="Welcome", lesson_link="https://example.com/ml-course/lesson-0"),
            Lesson(lesson_number=1, title="Supervised Learning", lesson_link="https://example.com/ml-course/lesson-1"),
            Lesson(lesson_number=2, title="Neural Networks", lesson_link="https://example.com/ml-course/lesson-2")
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample CourseChunk objects for testing"""
    return [
        CourseChunk(
            content="Course Introduction to Machine Learning Lesson 0 content: This is the introduction to machine learning.",
            course_title="Introduction to Machine Learning",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning involves training models on labeled data.",
            course_title="Introduction to Machine Learning",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Neural networks are inspired by biological neurons.",
            course_title="Introduction to Machine Learning",
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def sample_course_document():
    """Sample course document content in expected format"""
    return """Course Title: Introduction to Machine Learning
Course Link: https://example.com/ml-course
Course Instructor: Dr. Jane Smith

Lesson 0: Welcome
Lesson Link: https://example.com/ml-course/lesson-0
This is the introduction to machine learning. It covers basic concepts and terminology.

Lesson 1: Supervised Learning
Lesson Link: https://example.com/ml-course/lesson-1
Supervised learning involves training models on labeled data. The model learns patterns from examples.

Lesson 2: Neural Networks
Lesson Link: https://example.com/ml-course/lesson-2
Neural networks are inspired by biological neurons. They consist of layers of interconnected nodes.
"""


@pytest.fixture
def malformed_course_document():
    """Course document with missing metadata"""
    return """Introduction to Python

Some lesson content without proper structure.
More content here.
"""


@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection"""
    collection = Mock()
    collection.query = Mock(return_value={
        'documents': [['Sample document 1', 'Sample document 2']],
        'metadatas': [[
            {'course_title': 'Test Course', 'lesson_number': 1, 'chunk_index': 0},
            {'course_title': 'Test Course', 'lesson_number': 2, 'chunk_index': 1}
        ]],
        'distances': [[0.5, 0.7]],
        'ids': [['id1', 'id2']]
    })
    collection.add = Mock()
    collection.get = Mock(return_value={
        'ids': ['Course 1', 'Course 2'],
        'metadatas': [
            {'title': 'Course 1', 'instructor': 'Instructor 1', 'course_link': 'http://example.com/1'},
            {'title': 'Course 2', 'instructor': 'Instructor 2', 'course_link': 'http://example.com/2'}
        ]
    })
    return collection


@pytest.fixture
def mock_chroma_client(mock_chroma_collection):
    """Mock ChromaDB PersistentClient"""
    client = Mock()
    client.get_or_create_collection = Mock(return_value=mock_chroma_collection)
    client.delete_collection = Mock()
    return client


@pytest.fixture
def mock_anthropic_response_direct():
    """Mock Anthropic API response without tool use"""
    response = Mock()
    response.content = [Mock(text="This is the AI response", type="text")]
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def mock_anthropic_response_tool_use():
    """Mock Anthropic API response with tool use"""
    # Initial response with tool use
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "tool_123"
    tool_block.input = {"query": "machine learning"}

    response = Mock()
    response.content = [tool_block]
    response.stop_reason = "tool_use"
    return response


@pytest.fixture
def mock_anthropic_response_final():
    """Mock final Anthropic API response after tool execution"""
    response = Mock()
    response.content = [Mock(text="Based on the search results, here is the answer.", type="text")]
    response.stop_reason = "end_turn"
    return response


@pytest.fixture
def mock_anthropic_client(mock_anthropic_response_direct):
    """Mock Anthropic client"""
    client = Mock()
    client.messages = Mock()
    client.messages.create = Mock(return_value=mock_anthropic_response_direct)
    return client


@pytest.fixture
def mock_config():
    """Mock configuration object"""
    config = Mock()
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.ANTHROPIC_API_KEY = "test-api-key"
    config.ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
    return config
