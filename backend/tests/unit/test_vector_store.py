import pytest
from unittest.mock import Mock, patch, MagicMock
import json
from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


@pytest.mark.unit
class TestSearchResults:
    """Unit tests for SearchResults dataclass"""

    def test_from_chroma_with_results(self):
        """Test creating SearchResults from ChromaDB query results"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.5, 0.7]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'value1'}, {'key': 'value2'}]
        assert results.distances == [0.5, 0.7]
        assert results.error is None

    def test_from_chroma_empty_results(self):
        """Test creating SearchResults from empty ChromaDB results"""
        chroma_results = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }

        results = SearchResults.from_chroma(chroma_results)

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []

    def test_empty_with_error(self):
        """Test creating empty SearchResults with error message"""
        results = SearchResults.empty("No results found")

        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "No results found"

    def test_is_empty_true(self):
        """Test is_empty returns True for empty results"""
        results = SearchResults(documents=[], metadata=[], distances=[])

        assert results.is_empty() is True

    def test_is_empty_false(self):
        """Test is_empty returns False for non-empty results"""
        results = SearchResults(
            documents=['doc1'],
            metadata=[{'key': 'value'}],
            distances=[0.5]
        )

        assert results.is_empty() is False


@pytest.mark.unit
class TestVectorStore:
    """Unit tests for VectorStore"""

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_initialization(self, mock_embedding_fn, mock_client_class, mock_chroma_collection):
        """Test VectorStore initializes correctly"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.get_or_create_collection = Mock(return_value=mock_chroma_collection)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)

        assert store.max_results == 5
        mock_client_class.assert_called_once()
        assert mock_client.get_or_create_collection.call_count == 2  # catalog + content

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_metadata(self, mock_embedding_fn, mock_client_class, sample_course):
        """Test adding course metadata to catalog"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_collection = Mock()
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        store.add_course_metadata(sample_course)

        # Verify add was called
        mock_collection.add.assert_called_once()
        call_args = mock_collection.add.call_args

        # Check documents
        assert call_args.kwargs['documents'] == [sample_course.title]

        # Check IDs
        assert call_args.kwargs['ids'] == [sample_course.title]

        # Check metadata
        metadata = call_args.kwargs['metadatas'][0]
        assert metadata['title'] == sample_course.title
        assert metadata['instructor'] == sample_course.instructor
        assert metadata['course_link'] == sample_course.course_link
        assert metadata['lesson_count'] == 3

        # Verify lessons JSON
        lessons = json.loads(metadata['lessons_json'])
        assert len(lessons) == 3
        assert lessons[0]['lesson_number'] == 0
        assert lessons[0]['lesson_title'] == "Welcome"

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content(self, mock_embedding_fn, mock_client_class, sample_course_chunks):
        """Test adding course content chunks"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_collection = Mock()
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        # Get the content collection (second call)
        content_collection = mock_client.get_or_create_collection.return_value

        store.add_course_content(sample_course_chunks)

        # Verify add was called with correct data
        content_collection.add.assert_called_once()
        call_args = content_collection.add.call_args

        assert len(call_args.kwargs['documents']) == 3
        assert len(call_args.kwargs['metadatas']) == 3
        assert len(call_args.kwargs['ids']) == 3

        # Check first chunk
        assert call_args.kwargs['documents'][0] == sample_course_chunks[0].content
        assert call_args.kwargs['metadatas'][0]['course_title'] == "Introduction to Machine Learning"
        assert call_args.kwargs['metadatas'][0]['lesson_number'] == 0

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content_empty_list(self, mock_embedding_fn, mock_client_class):
        """Test adding empty chunk list does nothing"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_collection = Mock()
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")

        store.add_course_content([])

        # Should not call add
        mock_collection.add.assert_not_called()

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_without_filters(self, mock_embedding_fn, mock_client_class):
        """Test basic search without course or lesson filters"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock collections
        mock_catalog = Mock()
        mock_content = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.get_or_create_collection = Mock(side_effect=get_collection)

        # Mock content search results
        mock_content.query = Mock(return_value={
            'documents': [['Result 1', 'Result 2']],
            'metadatas': [[{'course_title': 'Course A'}, {'course_title': 'Course B'}]],
            'distances': [[0.3, 0.5]]
        })

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)
        results = store.search(query="machine learning")

        # Verify search was called correctly
        mock_content.query.assert_called_once()
        call_args = mock_content.query.call_args
        assert call_args.kwargs['query_texts'] == ["machine learning"]
        assert call_args.kwargs['n_results'] == 5
        assert call_args.kwargs['where'] is None

        # Check results
        assert len(results.documents) == 2
        assert results.documents[0] == 'Result 1'

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_filter(self, mock_embedding_fn, mock_client_class):
        """Test search with course name filter"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_content = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.get_or_create_collection = Mock(side_effect=get_collection)

        # Mock course resolution
        mock_catalog.query = Mock(return_value={
            'documents': [['ML Course']],
            'metadatas': [[{'title': 'Machine Learning 101'}]]
        })

        # Mock content search
        mock_content.query = Mock(return_value={
            'documents': [['Result 1']],
            'metadatas': [[{'course_title': 'Machine Learning 101'}]],
            'distances': [[0.3]]
        })

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        results = store.search(query="neural networks", course_name="ML")

        # Verify course resolution was called
        mock_catalog.query.assert_called_once()
        assert mock_catalog.query.call_args.kwargs['query_texts'] == ["ML"]

        # Verify content search with filter
        call_args = mock_content.query.call_args
        assert call_args.kwargs['where'] == {'course_title': 'Machine Learning 101'}

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_lesson_filter(self, mock_embedding_fn, mock_client_class):
        """Test search with lesson number filter"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_content = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.get_or_create_collection = Mock(side_effect=get_collection)

        mock_content.query = Mock(return_value={
            'documents': [['Result 1']],
            'metadatas': [[{'course_title': 'Course', 'lesson_number': 2}]],
            'distances': [[0.3]]
        })

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        results = store.search(query="test", lesson_number=2)

        # Verify filter applied
        call_args = mock_content.query.call_args
        assert call_args.kwargs['where'] == {'lesson_number': 2}

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_both_filters(self, mock_embedding_fn, mock_client_class):
        """Test search with both course and lesson filters"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_content = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.get_or_create_collection = Mock(side_effect=get_collection)

        # Mock course resolution
        mock_catalog.query = Mock(return_value={
            'documents': [['Course']],
            'metadatas': [[{'title': 'Python Course'}]]
        })

        mock_content.query = Mock(return_value={
            'documents': [['Result']],
            'metadatas': [[{'course_title': 'Python Course', 'lesson_number': 1}]],
            'distances': [[0.2]]
        })

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        results = store.search(query="test", course_name="Python", lesson_number=1)

        # Verify AND filter
        call_args = mock_content.query.call_args
        expected_filter = {
            "$and": [
                {"course_title": "Python Course"},
                {"lesson_number": 1}
            ]
        }
        assert call_args.kwargs['where'] == expected_filter

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_course_not_found(self, mock_embedding_fn, mock_client_class):
        """Test search returns error when course not found"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_content = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.get_or_create_collection = Mock(side_effect=get_collection)

        # Mock empty course resolution
        mock_catalog.query = Mock(return_value={
            'documents': [[]],
            'metadatas': [[]]
        })

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        results = store.search(query="test", course_name="Nonexistent")

        # Should return error
        assert results.error == "No course found matching 'Nonexistent'"
        assert results.is_empty()

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_custom_limit(self, mock_embedding_fn, mock_client_class):
        """Test search with custom result limit"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_content = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.get_or_create_collection = Mock(side_effect=get_collection)

        mock_content.query = Mock(return_value={
            'documents': [['R1', 'R2', 'R3']],
            'metadatas': [[{}, {}, {}]],
            'distances': [[0.1, 0.2, 0.3]]
        })

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model", max_results=5)
        results = store.search(query="test", limit=3)

        # Verify custom limit used
        call_args = mock_content.query.call_args
        assert call_args.kwargs['n_results'] == 3

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_error_handling(self, mock_embedding_fn, mock_client_class):
        """Test search handles exceptions gracefully"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_content = Mock()

        def get_collection(name, **kwargs):
            if name == "course_catalog":
                return mock_catalog
            return mock_content

        mock_client.get_or_create_collection = Mock(side_effect=get_collection)

        # Mock search error
        mock_content.query = Mock(side_effect=Exception("Database error"))

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        results = store.search(query="test")

        assert results.error is not None
        assert "Search error" in results.error

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_existing_course_titles(self, mock_embedding_fn, mock_client_class):
        """Test getting all course titles"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_catalog.get = Mock(return_value={
            'ids': ['Course 1', 'Course 2', 'Course 3']
        })

        mock_client.get_or_create_collection = Mock(return_value=mock_catalog)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        titles = store.get_existing_course_titles()

        assert titles == ['Course 1', 'Course 2', 'Course 3']

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_count(self, mock_embedding_fn, mock_client_class):
        """Test getting course count"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_catalog.get = Mock(return_value={
            'ids': ['C1', 'C2', 'C3', 'C4']
        })

        mock_client.get_or_create_collection = Mock(return_value=mock_catalog)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        count = store.get_course_count()

        assert count == 4

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_clear_all_data(self, mock_embedding_fn, mock_client_class):
        """Test clearing all data from collections"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_collection = Mock()
        mock_client.get_or_create_collection = Mock(return_value=mock_collection)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        store.clear_all_data()

        # Should delete both collections
        assert mock_client.delete_collection.call_count == 2

        # Should recreate collections
        assert mock_client.get_or_create_collection.call_count >= 4  # 2 initial + 2 recreate

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_link(self, mock_embedding_fn, mock_client_class):
        """Test getting course link by title"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        mock_catalog.get = Mock(return_value={
            'metadatas': [{'course_link': 'https://example.com/course'}]
        })

        mock_client.get_or_create_collection = Mock(return_value=mock_catalog)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        link = store.get_course_link("Test Course")

        assert link == 'https://example.com/course'
        mock_catalog.get.assert_called_with(ids=["Test Course"])

    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_lesson_link(self, mock_embedding_fn, mock_client_class):
        """Test getting lesson link by course and lesson number"""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        mock_catalog = Mock()
        lessons_json = json.dumps([
            {'lesson_number': 0, 'lesson_link': 'https://example.com/lesson-0'},
            {'lesson_number': 1, 'lesson_link': 'https://example.com/lesson-1'}
        ])
        mock_catalog.get = Mock(return_value={
            'metadatas': [{'lessons_json': lessons_json}]
        })

        mock_client.get_or_create_collection = Mock(return_value=mock_catalog)

        store = VectorStore(chroma_path="./test_db", embedding_model="test-model")
        link = store.get_lesson_link("Test Course", 1)

        assert link == 'https://example.com/lesson-1'
