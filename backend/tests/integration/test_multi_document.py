import pytest
import tempfile
import os
from unittest.mock import Mock, patch
from rag_system import RAGSystem
from models import Course


@pytest.mark.integration
class TestMultiDocumentScenarios:
    """Integration tests for multi-document scenarios"""

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_multiple_courses_ingestion(self, mock_embedding_fn, mock_chroma_client_class,
                                        mock_anthropic_class, mock_tool_manager_class,
                                        mock_search_tool_class, mock_config):
        """Test ingesting multiple course documents"""

        # Setup ChromaDB mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        stored_courses = []

        def capture_course_add(**kwargs):
            stored_courses.extend(kwargs['ids'])

        mock_catalog_collection.add = Mock(side_effect=capture_course_add)
        mock_catalog_collection.get = Mock(return_value={'ids': []})
        mock_content_collection.add = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        # Create multiple course documents
        course1_content = """Course Title: Python Basics
Course Instructor: Alice

Lesson 1: Introduction
Python is a programming language.
"""

        course2_content = """Course Title: Machine Learning
Course Instructor: Bob

Lesson 1: What is ML
Machine learning is a subset of AI.
"""

        course3_content = """Course Title: Web Development
Course Instructor: Carol

Lesson 1: HTML Basics
HTML is the structure of web pages.
"""

        temp_files = []
        try:
            # Create temp files for each course
            for content in [course1_content, course2_content, course3_content]:
                f = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt')
                f.write(content)
                f.close()
                temp_files.append(f.name)

            rag = RAGSystem(mock_config)

            # Ingest all courses
            for temp_file in temp_files:
                course, chunks = rag.add_course_document(temp_file)
                assert course is not None
                assert chunks > 0

            # Verify all three courses were stored
            assert len(stored_courses) == 3
            assert "Python Basics" in stored_courses
            assert "Machine Learning" in stored_courses
            assert "Web Development" in stored_courses

        finally:
            for temp_file in temp_files:
                os.unlink(temp_file)

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('os.path.exists')
    def test_duplicate_course_prevention(self, mock_exists, mock_listdir, mock_isfile,
                                         mock_embedding_fn, mock_chroma_client_class,
                                         mock_anthropic_class, mock_tool_manager_class,
                                         mock_search_tool_class, mock_config):
        """Test that duplicate courses are not re-ingested"""

        # Setup filesystem mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ['course1.txt', 'course2.txt', 'course1_copy.txt']
        mock_isfile.return_value = True

        # Setup ChromaDB mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        added_courses = []

        def capture_add(**kwargs):
            added_courses.extend(kwargs['ids'])

        mock_catalog_collection.add = Mock(side_effect=capture_add)
        mock_catalog_collection.get = Mock(return_value={'ids': []})
        mock_content_collection.add = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        # Create temp directory with course files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create three files, two with same course title
            course1_content = """Course Title: Python Basics
Lesson 1: Intro
Content here."""

            course2_content = """Course Title: Web Dev
Lesson 1: HTML
Content here."""

            # course1_copy has same title as course1
            course1_copy_content = """Course Title: Python Basics
Lesson 1: Different Intro
Different content."""

            files = [
                (os.path.join(temp_dir, 'course1.txt'), course1_content),
                (os.path.join(temp_dir, 'course2.txt'), course2_content),
                (os.path.join(temp_dir, 'course1_copy.txt'), course1_copy_content)
            ]

            for filepath, content in files:
                with open(filepath, 'w') as f:
                    f.write(content)

            # Mock file reading
            def read_file_side_effect(path):
                for fp, content in files:
                    if fp in path or os.path.basename(fp) in path:
                        return content
                return ""

            with patch('document_processor.DocumentProcessor.read_file', side_effect=read_file_side_effect):
                rag = RAGSystem(mock_config)

                # Add folder
                total_courses, total_chunks = rag.add_course_folder(temp_dir)

                # Should only add 2 unique courses (Python Basics once, Web Dev once)
                assert total_courses == 2

                # Verify only unique course titles were added
                unique_courses = set(added_courses)
                assert len(unique_courses) == 2
                assert "Python Basics" in unique_courses
                assert "Web Dev" in unique_courses

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_cross_course_search(self, mock_embedding_fn, mock_chroma_client_class,
                                 mock_anthropic_class, mock_tool_manager_class,
                                 mock_search_tool_class, mock_config):
        """Test searching across multiple courses"""

        # Setup ChromaDB mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        # Mock search to return results from multiple courses
        mock_content_collection.query = Mock(return_value={
            'documents': [
                ['Python has simple syntax for beginners.'],
                ['JavaScript is widely used for web development.'],
                ['Both Python and JavaScript are popular.']
            ],
            'metadatas': [
                [{'course_title': 'Python Basics', 'lesson_number': 1}],
                [{'course_title': 'JavaScript Fundamentals', 'lesson_number': 1}],
                [{'course_title': 'Programming Comparison', 'lesson_number': 2}]
            ],
            'distances': [[0.2], [0.3], [0.4]]
        })

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        mock_catalog_collection.get = Mock(return_value={'ids': []})

        # Mock Anthropic response
        final_response = Mock()
        final_response.content = [Mock(text="Both Python and JavaScript are beginner-friendly.", type="text")]
        final_response.stop_reason = "end_turn"

        mock_anthropic_client = Mock()
        mock_anthropic_client.messages = Mock()
        mock_anthropic_client.messages.create = Mock(return_value=final_response)
        mock_anthropic_class.return_value = mock_anthropic_client

        rag = RAGSystem(mock_config)

        # Query without course filter - should search across all courses
        response, sources = rag.query("Which programming language is better for beginners?")

        # Verify response was generated
        assert response is not None
        assert "Python" in response or "JavaScript" in response or "beginner" in response.lower()

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_course_specific_search(self, mock_embedding_fn, mock_chroma_client_class,
                                    mock_anthropic_class, mock_tool_manager_class,
                                    mock_search_tool_class, mock_config):
        """Test filtering search to specific course"""

        # Setup ChromaDB mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        # Mock course resolution
        mock_catalog_collection.query = Mock(return_value={
            'documents': [['Python Basics']],
            'metadatas': [[{'title': 'Python Basics'}]]
        })

        # Mock filtered search results
        mock_content_collection.query = Mock(return_value={
            'documents': [['Python uses indentation for code blocks.']],
            'metadatas': [[{'course_title': 'Python Basics', 'lesson_number': 2}]],
            'distances': [[0.1]]
        })

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        mock_catalog_collection.get = Mock(return_value={'ids': ['Python Basics', 'Java Basics']})

        # Mock tool use flow
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_1"
        tool_block.input = {"query": "syntax", "course_name": "Python"}

        initial_response = Mock()
        initial_response.content = [tool_block]
        initial_response.stop_reason = "tool_use"

        final_response = Mock()
        final_response.content = [Mock(text="Python uses indentation.", type="text")]
        final_response.stop_reason = "end_turn"

        mock_anthropic_client = Mock()
        mock_anthropic_client.messages = Mock()
        mock_anthropic_client.messages.create = Mock(side_effect=[initial_response, final_response])
        mock_anthropic_class.return_value = mock_anthropic_client

        rag = RAGSystem(mock_config)

        # Query with implicit course filter in question
        response, sources = rag.query("What is Python's syntax like?")

        # Verify response generated
        assert response is not None

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_analytics_with_multiple_courses(self, mock_embedding_fn, mock_chroma_client_class,
                                             mock_anthropic_class, mock_tool_manager_class,
                                             mock_search_tool_class, mock_config):
        """Test getting analytics for multiple courses"""

        # Setup ChromaDB mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        mock_catalog_collection.get = Mock(return_value={
            'ids': ['Python Basics', 'Web Development', 'Machine Learning', 'Data Science'],
            'metadatas': [
                {'title': 'Python Basics'},
                {'title': 'Web Development'},
                {'title': 'Machine Learning'},
                {'title': 'Data Science'}
            ]
        })

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        rag = RAGSystem(mock_config)

        # Get analytics
        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 4
        assert len(analytics["course_titles"]) == 4
        assert "Python Basics" in analytics["course_titles"]
        assert "Machine Learning" in analytics["course_titles"]

    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('ai_generator.anthropic.Anthropic')
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('os.path.exists')
    def test_mixed_file_types_in_folder(self, mock_exists, mock_listdir, mock_isfile,
                                        mock_embedding_fn, mock_chroma_client_class,
                                        mock_anthropic_class, mock_tool_manager_class,
                                        mock_search_tool_class, mock_config):
        """Test folder ingestion with mixed file types"""

        mock_exists.return_value = True
        mock_listdir.return_value = [
            'course1.txt',      # Valid
            'course2.pdf',      # Valid
            'course3.docx',     # Valid
            'readme.md',        # Invalid - should be skipped
            'image.png',        # Invalid - should be skipped
            'data.csv',         # Invalid - should be skipped
            '.hidden.txt'       # Invalid - hidden file
        ]
        mock_isfile.return_value = True

        # Setup ChromaDB mocks
        mock_catalog_collection = Mock()
        mock_content_collection = Mock()

        processed_files = []

        def track_processing(**kwargs):
            # Track which courses were processed
            processed_files.extend(kwargs['ids'])

        mock_catalog_collection.add = Mock(side_effect=track_processing)
        mock_catalog_collection.get = Mock(return_value={'ids': []})
        mock_content_collection.add = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog_collection
            return mock_content_collection

        mock_chroma_client = Mock()
        mock_chroma_client.get_or_create_collection = Mock(side_effect=get_collection)
        mock_chroma_client_class.return_value = mock_chroma_client

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create valid course files
            for filename in ['course1.txt', 'course2.pdf', 'course3.docx']:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write(f"Course Title: {filename}\nLesson 1: Test\nContent.")

            # Create invalid files (should be ignored)
            for filename in ['readme.md', 'image.png', 'data.csv']:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'w') as f:
                    f.write("Invalid content")

            rag = RAGSystem(mock_config)

            total_courses, total_chunks = rag.add_course_folder(temp_dir)

            # Should only process .txt, .pdf, .docx files
            assert total_courses == 3
            assert len(processed_files) == 3
