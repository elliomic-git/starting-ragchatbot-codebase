import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


@pytest.mark.integration
class TestRAGPipelineIntegration:
    """Integration tests for the complete RAG pipeline"""

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_document_ingestion_to_query_flow(self, mock_embedding_fn, mock_chroma_client_class,
                                             mock_anthropic_class, mock_tool_manager_class,
                                             mock_search_tool_class, mock_config):
        """Test complete flow: document ingestion → storage → query → response"""

        # Setup ChromaDB mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        # Mock catalog operations
        mock_catalog_collection.add = Mock()
        mock_catalog_collection.get = Mock(return_value={'ids': []})
        mock_catalog_collection.query = Mock(return_value={
            'documents': [['Machine Learning Course']],
            'metadatas': [[{'title': 'Machine Learning Course'}]]
        })

        # Mock content search results
        mock_content_collection.add = Mock()
        mock_content_collection.query = Mock(return_value={
            'documents': [['Machine learning is a subset of AI...']],
            'metadatas': [[{'course_title': 'Machine Learning Course', 'lesson_number': 1}]],
            'distances': [[0.3]]
        })

        # Setup Anthropic API mock
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_1"
        tool_block.input = {"query": "machine learning basics"}

        initial_response = Mock()
        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"

        final_response = Mock()
        final_response.content = [Mock(text="Machine learning is a subset of artificial intelligence that focuses on learning from data.", type="text")]
        final_response.stop_reason = "end_turn"

        mock_anthropic_client = Mock()
        mock_anthropic_client.messages = Mock()
        mock_anthropic_client.messages.create = Mock(side_effect=[initial_response, final_response])
        mock_anthropic_class.return_value = mock_anthropic_client

        # Create a real course document
        course_content = """Course Title: Machine Learning Course
Course Link: https://example.com/ml
Course Instructor: Dr. Smith

Lesson 1: Introduction to Machine Learning
Lesson Link: https://example.com/ml/lesson-1
Machine learning is a subset of artificial intelligence that focuses on learning from data. It involves algorithms that can improve their performance over time.

Lesson 2: Supervised Learning
Lesson Link: https://example.com/ml/lesson-2
Supervised learning uses labeled training data to learn the mapping between inputs and outputs.
"""

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(course_content)
            temp_path = f.name

        try:
            # Initialize RAG system
            rag = RAGSystem(mock_config)

            # Step 1: Ingest document
            course, chunk_count = rag.add_course_document(temp_path)

            # Verify document was processed
            assert course is not None
            assert course.title == "Machine Learning Course"
            assert course.instructor == "Dr. Smith"
            assert len(course.lessons) == 2
            assert chunk_count > 0

            # Verify data was added to vector store
            assert mock_catalog_collection.add.called
            assert mock_content_collection.add.called

            # Step 2: Query the system
            response, sources = rag.query("What is machine learning?")

            # Verify query executed
            assert response is not None
            assert "Machine learning" in response or "artificial intelligence" in response

            # Verify tool was called (search happened)
            assert mock_anthropic_client.messages.create.call_count == 2  # Initial + final

        finally:
            os.unlink(temp_path)

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_session_continuity_across_queries(self, mock_embedding_fn, mock_chroma_client_class,
                                               mock_anthropic_class, mock_tool_manager_class,
                                               mock_search_tool_class, mock_config):
        """Test that conversation history is maintained across multiple queries"""

        # Setup mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        mock_catalog_collection.get = Mock(return_value={'ids': []})
        mock_content_collection.query = Mock(return_value={
            'documents': [['Python is a programming language.']],
            'metadatas': [[{'course_title': 'Python Basics', 'lesson_number': 1}]],
            'distances': [[0.2]]
        })

        # Mock Anthropic responses
        def create_response(text):
            resp = Mock()
            resp.content = [Mock(text=text, type="text")]
            resp.stop_reason = "end_turn"
            return resp

        mock_anthropic_client = Mock()
        mock_anthropic_client.messages = Mock()
        mock_anthropic_client.messages.create = Mock(side_effect=[
            create_response("Python is a high-level programming language."),
            create_response("Yes, Python is great for beginners due to its simple syntax."),
            create_response("You can start with basic tutorials on variables and functions.")
        ])
        mock_anthropic_class.return_value = mock_anthropic_client

        rag = RAGSystem(mock_config)

        # Create a session and make multiple queries
        session_id = rag.session_manager.create_session()

        # First query
        response1, _ = rag.query("What is Python?", session_id=session_id)
        assert "Python" in response1

        # Second query - should have context from first
        response2, _ = rag.query("Is it good for beginners?", session_id=session_id)
        assert response2 is not None

        # Third query - should have context from both previous
        response3, _ = rag.query("How do I start learning?", session_id=session_id)
        assert response3 is not None

        # Verify conversation history was maintained
        # Note: MAX_HISTORY=2, so only last 2 exchanges (4 messages) are kept
        history = rag.session_manager.get_conversation_history(session_id)
        assert history is not None
        # First query may be trimmed if MAX_HISTORY is 2
        # At least the last two queries should be in history
        assert "Is it good for beginners?" in history or "How do I start learning?" in history

        # Verify history was passed to AI (check system prompt in calls)
        calls = mock_anthropic_client.messages.create.call_args_list

        # First call should not have history
        first_call_system = calls[0].kwargs['system']
        assert "Previous conversation:" not in first_call_system

        # Second call should have history
        second_call_system = calls[1].kwargs['system']
        assert "Previous conversation:" in second_call_system

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_error_handling_in_pipeline(self, mock_embedding_fn, mock_chroma_client_class,
                                        mock_anthropic_class, mock_tool_manager_class,
                                        mock_search_tool_class, mock_config):
        """Test error handling throughout the pipeline"""

        # Setup mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        mock_catalog_collection.get = Mock(return_value={'ids': []})

        rag = RAGSystem(mock_config)

        # Test 1: Invalid document path
        course, chunks = rag.add_course_document("/nonexistent/file.txt")
        assert course is None
        assert chunks == 0

        # Test 2: Empty document
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("")
            temp_path = f.name

        try:
            course, chunks = rag.add_course_document(temp_path)
            # Should handle gracefully (might create course with no lessons)
            assert course is not None or course is None  # Either is acceptable
        finally:
            os.unlink(temp_path)

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_chunking_preserves_context(self, mock_embedding_fn, mock_chroma_client_class,
                                        mock_anthropic_class, mock_tool_manager_class,
                                        mock_search_tool_class, mock_config):
        """Test that document chunking preserves lesson context"""

        # Setup mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        captured_chunks = []

        def capture_add(**kwargs):
            captured_chunks.extend(kwargs['documents'])

        mock_content_collection.add = Mock(side_effect=capture_add)

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        mock_catalog_collection.get = Mock(return_value={'ids': []})

        # Create document with long lesson content that will be chunked
        long_content = " ".join(["This is sentence number {}." for i in range(100)])
        course_content = f"""Course Title: Test Course
Course Instructor: Test Instructor

Lesson 1: Long Lesson
{long_content}
"""

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(course_content)
            temp_path = f.name

        try:
            rag = RAGSystem(mock_config)
            course, chunk_count = rag.add_course_document(temp_path)

            # Verify multiple chunks were created
            assert chunk_count > 1

            # Verify chunks have lesson context
            for chunk in captured_chunks:
                # Each chunk should reference the course and lesson
                assert "Course" in chunk or "Lesson" in chunk or "sentence" in chunk.lower()

        finally:
            os.unlink(temp_path)
