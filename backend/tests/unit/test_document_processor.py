import pytest
import tempfile
import os
from unittest.mock import Mock, patch, mock_open
from document_processor import DocumentProcessor
from models import Course, Lesson, CourseChunk


@pytest.mark.unit
class TestDocumentProcessor:
    """Unit tests for DocumentProcessor"""

    def test_initialization(self):
        """Test DocumentProcessor initializes with correct config"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        assert processor.chunk_size == 800
        assert processor.chunk_overlap == 100

    def test_read_file_utf8(self):
        """Test reading a UTF-8 encoded file"""
        processor = DocumentProcessor(800, 100)
        content = "Test content with UTF-8: café"

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(content)
            temp_path = f.name

        try:
            result = processor.read_file(temp_path)
            assert result == content
        finally:
            os.unlink(temp_path)

    def test_read_file_with_encoding_errors(self):
        """Test reading file with encoding errors uses fallback"""
        processor = DocumentProcessor(800, 100)

        with patch('builtins.open', mock_open(read_data="test")) as mock_file:
            # First call raises UnicodeDecodeError, second succeeds
            mock_file.return_value.read.side_effect = [
                UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'),
                "fallback content"
            ]

            result = processor.read_file("test.txt")
            assert result == "fallback content"

    def test_chunk_text_basic(self):
        """Test basic text chunking respects size limits"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=10)
        text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."

        chunks = processor.chunk_text(text)

        assert len(chunks) > 0
        for chunk in chunks:
            assert len(chunk) <= 50 or chunk.count('.') == 1  # Allow single long sentence

    def test_chunk_text_with_overlap(self):
        """Test that chunks have proper overlap"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)
        text = "First sentence here. Second sentence here. Third sentence here. Fourth sentence here. Fifth sentence here."

        chunks = processor.chunk_text(text)

        assert len(chunks) >= 2
        # Check for content overlap between consecutive chunks
        if len(chunks) >= 2:
            # Some content from first chunk should appear in second chunk
            first_chunk_end = chunks[0].split()[-3:]  # Last few words
            assert any(word in chunks[1] for word in first_chunk_end)

    def test_chunk_text_sentence_aware(self):
        """Test that chunking doesn't break mid-sentence"""
        processor = DocumentProcessor(chunk_size=60, chunk_overlap=10)
        text = "This is a complete sentence. This is another complete sentence with more words."

        chunks = processor.chunk_text(text)

        for chunk in chunks:
            # Each chunk should end with proper punctuation or be the last chunk
            assert chunk.strip().endswith('.') or chunk.strip().endswith('?') or chunk.strip().endswith('!')

    def test_chunk_text_normalizes_whitespace(self):
        """Test that chunking normalizes excessive whitespace"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        text = "This   has    extra     spaces.\n\n\nAnd   newlines."

        chunks = processor.chunk_text(text)

        for chunk in chunks:
            # Should not have multiple consecutive spaces
            assert '  ' not in chunk

    def test_chunk_text_empty_string(self):
        """Test chunking empty string returns empty list"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)

        chunks = processor.chunk_text("")

        assert chunks == []

    def test_chunk_text_single_sentence(self):
        """Test chunking single sentence"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        text = "This is a single sentence."

        chunks = processor.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == text

    def test_chunk_text_very_long_sentence(self):
        """Test chunking with sentence longer than chunk size"""
        processor = DocumentProcessor(chunk_size=30, chunk_overlap=5)
        text = "This is a very long sentence that exceeds the chunk size limit significantly."

        chunks = processor.chunk_text(text)

        # Should still create at least one chunk
        assert len(chunks) >= 1

    def test_chunk_text_handles_abbreviations(self):
        """Test that chunking handles common abbreviations like Dr., Mr."""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=10)
        text = "Dr. Smith works at the university. Mr. Jones is a professor. They collaborate on research."

        chunks = processor.chunk_text(text)

        # Should not split on abbreviations
        for chunk in chunks:
            if "Dr." in chunk:
                assert "Dr. Smith" in chunk or chunk.startswith("Dr.")

    def test_process_course_document_complete(self, sample_course_document):
        """Test processing a complete well-formed course document"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(sample_course_document)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            # Verify course metadata
            assert course.title == "Introduction to Machine Learning"
            assert course.course_link == "https://example.com/ml-course"
            assert course.instructor == "Dr. Jane Smith"
            assert len(course.lessons) == 3

            # Verify lessons
            assert course.lessons[0].lesson_number == 0
            assert course.lessons[0].title == "Welcome"
            assert course.lessons[0].lesson_link == "https://example.com/ml-course/lesson-0"

            # Verify chunks created
            assert len(chunks) > 0
            assert all(isinstance(chunk, CourseChunk) for chunk in chunks)
            assert all(chunk.course_title == "Introduction to Machine Learning" for chunk in chunks)

        finally:
            os.unlink(temp_path)

    def test_process_course_document_missing_metadata(self):
        """Test processing document with missing metadata fields"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        content = """Course Title: Python Basics

Lesson 0: Introduction
This is the lesson content.
"""

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(content)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            assert course.title == "Python Basics"
            assert course.course_link is None
            assert course.instructor is None

        finally:
            os.unlink(temp_path)

    def test_process_course_document_no_lessons(self):
        """Test processing document without lesson structure"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        content = """Course Title: Random Content
Course Link: http://example.com
Course Instructor: John Doe

This is just random content without lesson markers.
More content here.
"""

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(content)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            assert course.title == "Random Content"
            assert len(course.lessons) == 0
            # Should still create chunks from content
            assert len(chunks) > 0

        finally:
            os.unlink(temp_path)

    def test_process_course_document_chunk_counter_increments(self, sample_course_document):
        """Test that chunk indices increment correctly"""
        processor = DocumentProcessor(chunk_size=100, chunk_overlap=20)

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(sample_course_document)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            # Verify chunk indices are sequential
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_index == i

        finally:
            os.unlink(temp_path)

    def test_process_course_document_lesson_context_added(self, sample_course_document):
        """Test that lesson context is added to chunks"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(sample_course_document)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            # Last lesson chunks should have course title and lesson context
            last_lesson_chunks = [c for c in chunks if c.lesson_number == 2]
            for chunk in last_lesson_chunks:
                assert "Course Introduction to Machine Learning" in chunk.content
                assert "Lesson 2" in chunk.content

        finally:
            os.unlink(temp_path)

    def test_process_course_document_lesson_numbers_parsed(self, sample_course_document):
        """Test that lesson numbers are correctly parsed"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(sample_course_document)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            lesson_numbers = [lesson.lesson_number for lesson in course.lessons]
            assert lesson_numbers == [0, 1, 2]

        finally:
            os.unlink(temp_path)

    def test_process_course_document_lesson_links_parsed(self, sample_course_document):
        """Test that lesson links are correctly extracted"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(sample_course_document)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            assert course.lessons[0].lesson_link == "https://example.com/ml-course/lesson-0"
            assert course.lessons[1].lesson_link == "https://example.com/ml-course/lesson-1"
            assert course.lessons[2].lesson_link == "https://example.com/ml-course/lesson-2"

        finally:
            os.unlink(temp_path)

    def test_process_course_document_malformed(self, malformed_course_document):
        """Test processing malformed document falls back gracefully"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write(malformed_course_document)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            # Should use first line as title
            assert course.title == "Introduction to Python"
            # Should still create chunks
            assert len(chunks) > 0

        finally:
            os.unlink(temp_path)

    def test_process_course_document_uses_first_line_as_title(self):
        """Test that first line is used as title if no Course Title: marker found"""
        processor = DocumentProcessor(chunk_size=800, chunk_overlap=100)
        content = "Just content without a title.\nMore content here."

        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt',
                                        prefix='my_course_') as f:
            f.write(content)
            temp_path = f.name

        try:
            course, chunks = processor.process_course_document(temp_path)

            # Should use first line as title
            assert course.title == "Just content without a title."

        finally:
            os.unlink(temp_path)

    def test_chunk_text_no_overlap_when_zero(self):
        """Test chunking without overlap when overlap is 0"""
        processor = DocumentProcessor(chunk_size=50, chunk_overlap=0)
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."

        chunks = processor.chunk_text(text)

        # With no overlap, chunks should be completely independent
        assert len(chunks) >= 2
        # No content should repeat between chunks
        for i in range(len(chunks) - 1):
            # This is approximate - just check they're different
            assert chunks[i] != chunks[i + 1]
