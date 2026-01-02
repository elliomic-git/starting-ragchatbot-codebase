import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from rag_system import RAGSystem
from models import Course, Lesson, CourseChunk


@pytest.mark.unit
class TestRAGSystem:
    """Unit tests for RAGSystem"""

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_initialization(self, mock_search_tool_class, mock_tool_manager_class,
                           mock_session_class, mock_ai_class, mock_vector_class,
                           mock_doc_class, mock_config):
        """Test RAGSystem initializes all components correctly"""
        rag = RAGSystem(mock_config)

        # Verify all components were initialized with correct config
        mock_doc_class.assert_called_once_with(
            mock_config.CHUNK_SIZE,
            mock_config.CHUNK_OVERLAP
        )
        mock_vector_class.assert_called_once_with(
            mock_config.CHROMA_PATH,
            mock_config.EMBEDDING_MODEL,
            mock_config.MAX_RESULTS
        )
        mock_ai_class.assert_called_once_with(
            mock_config.ANTHROPIC_API_KEY,
            mock_config.ANTHROPIC_MODEL
        )
        mock_session_class.assert_called_once_with(mock_config.MAX_HISTORY)

        # Verify tool setup
        assert rag.tool_manager is not None
        assert rag.search_tool is not None

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_add_course_document_success(self, mock_search_tool_class, mock_tool_manager_class,
                                        mock_session_class, mock_ai_class, mock_vector_class,
                                        mock_doc_class, mock_config, sample_course,
                                        sample_course_chunks):
        """Test successfully adding a course document"""
        # Setup mocks
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document = Mock(
            return_value=(sample_course, sample_course_chunks)
        )
        mock_doc_class.return_value = mock_doc_processor

        mock_vector_store = Mock()
        mock_vector_class.return_value = mock_vector_store

        rag = RAGSystem(mock_config)

        # Add course document
        course, chunk_count = rag.add_course_document("test.txt")

        # Verify processing was called
        mock_doc_processor.process_course_document.assert_called_once_with("test.txt")

        # Verify data was added to vector store
        mock_vector_store.add_course_metadata.assert_called_once_with(sample_course)
        mock_vector_store.add_course_content.assert_called_once_with(sample_course_chunks)

        # Verify return values
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_add_course_document_error_handling(self, mock_search_tool_class, mock_tool_manager_class,
                                                mock_session_class, mock_ai_class, mock_vector_class,
                                                mock_doc_class, mock_config):
        """Test error handling when processing document fails"""
        # Setup mock to raise exception
        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document = Mock(
            side_effect=Exception("Processing failed")
        )
        mock_doc_class.return_value = mock_doc_processor

        rag = RAGSystem(mock_config)

        # Should return None and 0 on error
        course, chunk_count = rag.add_course_document("bad_file.txt")

        assert course is None
        assert chunk_count == 0

    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_processes_files(self, mock_doc_class, mock_vector_class,
                                               mock_ai_class, mock_session_class,
                                               mock_tool_manager_class, mock_search_tool_class,
                                               mock_exists, mock_listdir, mock_isfile,
                                               mock_config, sample_course_chunks):
        """Test adding all course documents from a folder"""
        # Setup mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ['course1.txt', 'course2.pdf', 'course3.docx', 'ignore.jpg']
        mock_isfile.return_value = True  # All files exist

        # Create different courses for each file
        def process_side_effect(path):
            if 'course1' in path:
                course = Course(title="Course 1", course_link=None, instructor=None)
            elif 'course2' in path:
                course = Course(title="Course 2", course_link=None, instructor=None)
            else:
                course = Course(title="Course 3", course_link=None, instructor=None)
            return (course, sample_course_chunks)

        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document = Mock(side_effect=process_side_effect)
        mock_doc_class.return_value = mock_doc_processor

        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles = Mock(return_value=[])
        mock_vector_class.return_value = mock_vector_store

        rag = RAGSystem(mock_config)

        # Add folder
        total_courses, total_chunks = rag.add_course_folder("/test/folder")

        # Should process only .txt, .pdf, .docx files
        assert mock_doc_processor.process_course_document.call_count == 3

        # Should add each course
        assert total_courses == 3
        assert total_chunks == len(sample_course_chunks) * 3

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.path.exists')
    def test_add_course_folder_nonexistent_folder(self, mock_exists, mock_search_tool_class,
                                                  mock_tool_manager_class, mock_session_class,
                                                  mock_ai_class, mock_vector_class, mock_doc_class,
                                                  mock_config):
        """Test handling of non-existent folder"""
        mock_exists.return_value = False

        rag = RAGSystem(mock_config)

        total_courses, total_chunks = rag.add_course_folder("/nonexistent")

        assert total_courses == 0
        assert total_chunks == 0

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    @patch('os.path.exists')
    @patch('os.listdir')
    def test_add_course_folder_with_clear_existing(self, mock_listdir, mock_exists,
                                                   mock_search_tool_class, mock_tool_manager_class,
                                                   mock_session_class, mock_ai_class,
                                                   mock_vector_class, mock_doc_class, mock_config):
        """Test clearing existing data before adding folder"""
        mock_exists.return_value = True
        mock_listdir.return_value = []

        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles = Mock(return_value=[])
        mock_vector_class.return_value = mock_vector_store

        rag = RAGSystem(mock_config)

        rag.add_course_folder("/test/folder", clear_existing=True)

        # Should call clear_all_data
        mock_vector_store.clear_all_data.assert_called_once()

    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_duplicate_detection(self, mock_doc_class, mock_vector_class,
                                                   mock_ai_class, mock_session_class,
                                                   mock_tool_manager_class, mock_search_tool_class,
                                                   mock_exists, mock_listdir, mock_isfile,
                                                   mock_config, sample_course, sample_course_chunks):
        """Test that duplicate courses are skipped"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['course1.txt', 'course2.txt']
        mock_isfile.return_value = True

        # First course is new, second already exists
        existing_course = Course(title="Existing Course", course_link=None, instructor=None)

        def process_side_effect(path):
            if 'course1' in path:
                return (sample_course, sample_course_chunks)
            else:
                return (existing_course, sample_course_chunks)

        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document = Mock(side_effect=process_side_effect)
        mock_doc_class.return_value = mock_doc_processor

        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles = Mock(return_value=["Existing Course"])
        mock_vector_class.return_value = mock_vector_store

        rag = RAGSystem(mock_config)

        total_courses, total_chunks = rag.add_course_folder("/test/folder")

        # Only one course should be added (the new one)
        assert total_courses == 1
        assert total_chunks == len(sample_course_chunks)

        # Verify only one course was added to vector store
        assert mock_vector_store.add_course_metadata.call_count == 1
        mock_vector_store.add_course_metadata.assert_called_with(sample_course)

    @patch('os.path.isfile')
    @patch('os.listdir')
    @patch('os.path.exists')
    @patch('rag_system.CourseSearchTool')
    @patch('rag_system.ToolManager')
    @patch('rag_system.SessionManager')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.VectorStore')
    @patch('rag_system.DocumentProcessor')
    def test_add_course_folder_handles_individual_file_errors(self, mock_doc_class, mock_vector_class,
                                                              mock_ai_class, mock_session_class,
                                                              mock_tool_manager_class,
                                                              mock_search_tool_class,
                                                              mock_exists, mock_listdir, mock_isfile,
                                                              mock_config, sample_course,
                                                              sample_course_chunks):
        """Test that errors in individual files don't stop processing"""
        mock_exists.return_value = True
        mock_listdir.return_value = ['good.txt', 'bad.txt']
        mock_isfile.return_value = True

        def process_side_effect(path):
            if 'bad' in path:
                raise Exception("Bad file")
            return (sample_course, sample_course_chunks)

        mock_doc_processor = Mock()
        mock_doc_processor.process_course_document = Mock(side_effect=process_side_effect)
        mock_doc_class.return_value = mock_doc_processor

        mock_vector_store = Mock()
        mock_vector_store.get_existing_course_titles = Mock(return_value=[])
        mock_vector_class.return_value = mock_vector_store

        rag = RAGSystem(mock_config)

        total_courses, total_chunks = rag.add_course_folder("/test/folder")

        # Should process good file despite error in bad file
        assert total_courses == 1

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_without_session(self, mock_search_tool_class, mock_tool_manager_class,
                                   mock_session_class, mock_ai_class, mock_vector_class,
                                   mock_doc_class, mock_config):
        """Test query without session ID"""
        mock_ai = Mock()
        mock_ai.generate_response = Mock(return_value="AI response")
        mock_ai_class.return_value = mock_ai

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions = Mock(return_value=[])
        mock_tool_manager.get_last_sources = Mock(return_value=[])
        mock_tool_manager_class.return_value = mock_tool_manager

        mock_session = Mock()
        mock_session.get_conversation_history = Mock(return_value=None)
        mock_session_class.return_value = mock_session

        rag = RAGSystem(mock_config)

        response, sources = rag.query("What is Python?")

        # Should generate response
        assert response == "AI response"
        assert sources == []

        # Should not try to get history for None session
        mock_session.get_conversation_history.assert_not_called()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_with_session(self, mock_search_tool_class, mock_tool_manager_class,
                               mock_session_class, mock_ai_class, mock_vector_class,
                               mock_doc_class, mock_config):
        """Test query with session ID uses conversation history"""
        mock_ai = Mock()
        mock_ai.generate_response = Mock(return_value="AI response")
        mock_ai_class.return_value = mock_ai

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions = Mock(return_value=[])
        mock_tool_manager.get_last_sources = Mock(return_value=[])
        mock_tool_manager_class.return_value = mock_tool_manager

        mock_session = Mock()
        mock_session.get_conversation_history = Mock(return_value="User: Hi\nAssistant: Hello")
        mock_session_class.return_value = mock_session

        rag = RAGSystem(mock_config)

        response, sources = rag.query("Follow up", session_id="session_1")

        # Should retrieve history
        mock_session.get_conversation_history.assert_called_once_with("session_1")

        # Should pass history to AI
        call_args = mock_ai.generate_response.call_args
        assert call_args.kwargs['conversation_history'] == "User: Hi\nAssistant: Hello"

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_updates_conversation_history(self, mock_search_tool_class, mock_tool_manager_class,
                                                mock_session_class, mock_ai_class, mock_vector_class,
                                                mock_doc_class, mock_config):
        """Test that query updates conversation history"""
        mock_ai = Mock()
        mock_ai.generate_response = Mock(return_value="AI response")
        mock_ai_class.return_value = mock_ai

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions = Mock(return_value=[])
        mock_tool_manager.get_last_sources = Mock(return_value=[])
        mock_tool_manager_class.return_value = mock_tool_manager

        mock_session = Mock()
        mock_session.get_conversation_history = Mock(return_value=None)
        mock_session_class.return_value = mock_session

        rag = RAGSystem(mock_config)

        rag.query("What is ML?", session_id="session_1")

        # Should add exchange to session
        mock_session.add_exchange.assert_called_once_with(
            "session_1",
            "What is ML?",
            "AI response"
        )

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_passes_tools_to_ai(self, mock_search_tool_class, mock_tool_manager_class,
                                     mock_session_class, mock_ai_class, mock_vector_class,
                                     mock_doc_class, mock_config):
        """Test that query passes tool definitions to AI generator"""
        mock_ai = Mock()
        mock_ai.generate_response = Mock(return_value="Response")
        mock_ai_class.return_value = mock_ai

        tool_defs = [{"name": "search_course_content"}]
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions = Mock(return_value=tool_defs)
        mock_tool_manager.get_last_sources = Mock(return_value=[])
        mock_tool_manager_class.return_value = mock_tool_manager

        rag = RAGSystem(mock_config)

        rag.query("Question")

        # Verify tools and tool_manager passed to AI
        call_args = mock_ai.generate_response.call_args
        assert call_args.kwargs['tools'] == tool_defs
        assert call_args.kwargs['tool_manager'] == mock_tool_manager

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_returns_sources_and_resets(self, mock_search_tool_class, mock_tool_manager_class,
                                             mock_session_class, mock_ai_class, mock_vector_class,
                                             mock_doc_class, mock_config):
        """Test that query returns sources and resets them"""
        mock_ai = Mock()
        mock_ai.generate_response = Mock(return_value="Response")
        mock_ai_class.return_value = mock_ai

        sources = [{"text": "Source 1", "url": "http://example.com"}]
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions = Mock(return_value=[])
        mock_tool_manager.get_last_sources = Mock(return_value=sources)
        mock_tool_manager_class.return_value = mock_tool_manager

        rag = RAGSystem(mock_config)

        response, returned_sources = rag.query("Question")

        # Should return sources
        assert returned_sources == sources

        # Should reset sources after getting them
        mock_tool_manager.reset_sources.assert_called_once()

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_get_course_analytics(self, mock_search_tool_class, mock_tool_manager_class,
                                  mock_session_class, mock_ai_class, mock_vector_class,
                                  mock_doc_class, mock_config):
        """Test getting course analytics"""
        mock_vector_store = Mock()
        mock_vector_store.get_course_count = Mock(return_value=5)
        mock_vector_store.get_existing_course_titles = Mock(
            return_value=["Course 1", "Course 2", "Course 3", "Course 4", "Course 5"]
        )
        mock_vector_class.return_value = mock_vector_store

        rag = RAGSystem(mock_config)

        analytics = rag.get_course_analytics()

        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 5
        assert "Course 1" in analytics["course_titles"]

    @patch('rag_system.DocumentProcessor')
    @patch('rag_system.VectorStore')
    @patch('rag_system.AIGenerator')
    @patch('rag_system.SessionManager')
    @patch('rag_system.ToolManager')
    @patch('rag_system.CourseSearchTool')
    def test_query_formats_prompt_correctly(self, mock_search_tool_class, mock_tool_manager_class,
                                           mock_session_class, mock_ai_class, mock_vector_class,
                                           mock_doc_class, mock_config):
        """Test that query formats the prompt correctly"""
        mock_ai = Mock()
        mock_ai.generate_response = Mock(return_value="Response")
        mock_ai_class.return_value = mock_ai

        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions = Mock(return_value=[])
        mock_tool_manager.get_last_sources = Mock(return_value=[])
        mock_tool_manager_class.return_value = mock_tool_manager

        rag = RAGSystem(mock_config)

        rag.query("What is machine learning?")

        # Check prompt formatting
        call_args = mock_ai.generate_response.call_args
        prompt = call_args.kwargs['query']
        assert "Answer this question about course materials:" in prompt
        assert "What is machine learning?" in prompt
