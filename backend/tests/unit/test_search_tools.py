import pytest
from unittest.mock import Mock, MagicMock
from search_tools import CourseSearchTool, ToolManager, Tool
from vector_store import SearchResults


@pytest.mark.unit
class TestCourseSearchTool:
    """Unit tests for CourseSearchTool"""

    def test_get_tool_definition(self):
        """Test tool definition has correct schema"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)

        definition = tool.get_tool_definition()

        assert definition['name'] == "search_course_content"
        assert 'description' in definition
        assert 'input_schema' in definition
        assert definition['input_schema']['type'] == 'object'
        assert 'properties' in definition['input_schema']
        assert 'query' in definition['input_schema']['properties']
        assert definition['input_schema']['required'] == ['query']

    def test_tool_definition_has_optional_parameters(self):
        """Test tool definition includes optional course_name and lesson_number"""
        mock_store = Mock()
        tool = CourseSearchTool(mock_store)

        definition = tool.get_tool_definition()
        properties = definition['input_schema']['properties']

        assert 'course_name' in properties
        assert 'lesson_number' in properties
        assert properties['lesson_number']['type'] == 'integer'

    def test_execute_query_only(self):
        """Test executing search with query only"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc 1', 'Doc 2'],
            metadata=[
                {'course_title': 'Course A', 'lesson_number': 1},
                {'course_title': 'Course B', 'lesson_number': 2}
            ],
            distances=[0.3, 0.5]
        ))

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="machine learning")

        mock_store.search.assert_called_once_with(
            query="machine learning",
            course_name=None,
            lesson_number=None
        )
        assert "[Course A - Lesson 1]" in result
        assert "Doc 1" in result

    def test_execute_with_course_filter(self):
        """Test executing search with course name filter"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc 1'],
            metadata=[{'course_title': 'ML Course', 'lesson_number': 1}],
            distances=[0.3]
        ))

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="neural networks", course_name="ML Course")

        mock_store.search.assert_called_once_with(
            query="neural networks",
            course_name="ML Course",
            lesson_number=None
        )

    def test_execute_with_lesson_filter(self):
        """Test executing search with lesson number filter"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc 1'],
            metadata=[{'course_title': 'Course', 'lesson_number': 3}],
            distances=[0.2]
        ))

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test", lesson_number=3)

        mock_store.search.assert_called_once_with(
            query="test",
            course_name=None,
            lesson_number=3
        )

    def test_execute_with_all_parameters(self):
        """Test executing search with all parameters"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc'],
            metadata=[{'course_title': 'Python', 'lesson_number': 2}],
            distances=[0.1]
        ))

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="functions", course_name="Python", lesson_number=2)

        mock_store.search.assert_called_once_with(
            query="functions",
            course_name="Python",
            lesson_number=2
        )

    def test_execute_returns_error_on_search_error(self):
        """Test that search errors are returned"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search failed"
        ))

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert result == "Search failed"

    def test_execute_empty_results_message(self):
        """Test empty results return appropriate message"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        ))

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters_message(self):
        """Test empty results with filters show filter info"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=[],
            metadata=[],
            distances=[]
        ))

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test", course_name="Python", lesson_number=1)

        assert "in course 'Python'" in result
        assert "in lesson 1" in result

    def test_format_results_includes_context(self):
        """Test that formatted results include course and lesson context"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Content here'],
            metadata=[{'course_title': 'Test Course', 'lesson_number': 5}],
            distances=[0.4]
        ))
        mock_store.get_lesson_link = Mock(return_value="https://example.com/lesson-5")

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert "[Test Course - Lesson 5]" in result
        assert "Content here" in result

    def test_format_results_without_lesson_number(self):
        """Test formatting results when lesson number is None"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Content'],
            metadata=[{'course_title': 'Course', 'lesson_number': None}],
            distances=[0.3]
        ))
        mock_store.get_course_link = Mock(return_value="https://example.com/course")

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert "[Course]" in result
        assert "Lesson" not in result.split('\n')[0]  # First line shouldn't have lesson

    def test_sources_tracking(self):
        """Test that sources are tracked correctly"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc 1', 'Doc 2'],
            metadata=[
                {'course_title': 'Course A', 'lesson_number': 1},
                {'course_title': 'Course B', 'lesson_number': 2}
            ],
            distances=[0.3, 0.5]
        ))
        mock_store.get_lesson_link = Mock(side_effect=[
            "https://example.com/a/1",
            "https://example.com/b/2"
        ])

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]['text'] == "Course A - Lesson 1"
        assert tool.last_sources[0]['url'] == "https://example.com/a/1"
        assert tool.last_sources[1]['text'] == "Course B - Lesson 2"
        assert tool.last_sources[1]['url'] == "https://example.com/b/2"

    def test_source_deduplication(self):
        """Test that duplicate sources are deduplicated"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc 1', 'Doc 2', 'Doc 3'],
            metadata=[
                {'course_title': 'Course A', 'lesson_number': 1},
                {'course_title': 'Course A', 'lesson_number': 1},  # Duplicate
                {'course_title': 'Course B', 'lesson_number': 2}
            ],
            distances=[0.1, 0.2, 0.3]
        ))
        mock_store.get_lesson_link = Mock(return_value="https://example.com/link")

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        # Should only have 2 unique sources
        assert len(tool.last_sources) == 2

    def test_url_fallback_to_course_link(self):
        """Test URL falls back to course link when lesson link not available"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc'],
            metadata=[{'course_title': 'Course', 'lesson_number': 1}],
            distances=[0.3]
        ))
        mock_store.get_lesson_link = Mock(return_value=None)
        mock_store.get_course_link = Mock(return_value="https://example.com/course")

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert tool.last_sources[0]['url'] == "https://example.com/course"
        mock_store.get_lesson_link.assert_called_once()
        mock_store.get_course_link.assert_called_once()

    def test_url_none_when_no_links_available(self):
        """Test URL is None when neither lesson nor course link available"""
        mock_store = Mock()
        mock_store.search = Mock(return_value=SearchResults(
            documents=['Doc'],
            metadata=[{'course_title': 'Course', 'lesson_number': None}],
            distances=[0.3]
        ))
        mock_store.get_course_link = Mock(return_value=None)

        tool = CourseSearchTool(mock_store)
        result = tool.execute(query="test")

        assert tool.last_sources[0]['url'] is None


@pytest.mark.unit
class TestToolManager:
    """Unit tests for ToolManager"""

    def test_register_tool(self):
        """Test registering a tool"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={
            "name": "test_tool",
            "description": "Test"
        })

        manager.register_tool(mock_tool)

        assert "test_tool" in manager.tools
        assert manager.tools["test_tool"] == mock_tool

    def test_register_tool_without_name_raises_error(self):
        """Test that registering tool without name raises error"""
        manager = ToolManager()
        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={"description": "Test"})

        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)

    def test_get_tool_definitions(self):
        """Test getting all tool definitions"""
        manager = ToolManager()

        tool1 = Mock()
        tool1.get_tool_definition = Mock(return_value={"name": "tool1"})

        tool2 = Mock()
        tool2.get_tool_definition = Mock(return_value={"name": "tool2"})

        manager.register_tool(tool1)
        manager.register_tool(tool2)

        definitions = manager.get_tool_definitions()

        assert len(definitions) == 2
        assert {"name": "tool1"} in definitions
        assert {"name": "tool2"} in definitions

    def test_execute_tool_by_name(self):
        """Test executing a tool by name"""
        manager = ToolManager()

        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={"name": "search"})
        mock_tool.execute = Mock(return_value="Search results")

        manager.register_tool(mock_tool)

        result = manager.execute_tool("search", query="test", course_name="Python")

        mock_tool.execute.assert_called_once_with(query="test", course_name="Python")
        assert result == "Search results"

    def test_execute_tool_not_found(self):
        """Test executing non-existent tool returns error message"""
        manager = ToolManager()

        result = manager.execute_tool("nonexistent", param="value")

        assert "Tool 'nonexistent' not found" in result

    def test_get_last_sources_from_search_tool(self):
        """Test getting sources from search tool"""
        manager = ToolManager()

        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={"name": "search"})
        mock_tool.last_sources = [
            {"text": "Source 1", "url": "http://example.com/1"},
            {"text": "Source 2", "url": "http://example.com/2"}
        ]

        manager.register_tool(mock_tool)

        sources = manager.get_last_sources()

        assert len(sources) == 2
        assert sources[0]['text'] == "Source 1"

    def test_get_last_sources_empty_when_no_sources(self):
        """Test get_last_sources returns empty list when no sources"""
        manager = ToolManager()

        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={"name": "tool"})
        # Tool doesn't have last_sources attribute at all
        del mock_tool.last_sources

        manager.register_tool(mock_tool)

        sources = manager.get_last_sources()

        assert sources == []

    def test_get_last_sources_from_multiple_tools(self):
        """Test getting sources when multiple tools exist"""
        manager = ToolManager()

        tool1 = Mock()
        tool1.get_tool_definition = Mock(return_value={"name": "tool1"})
        tool1.last_sources = []

        tool2 = Mock()
        tool2.get_tool_definition = Mock(return_value={"name": "tool2"})
        tool2.last_sources = [{"text": "Source", "url": "http://example.com"}]

        manager.register_tool(tool1)
        manager.register_tool(tool2)

        sources = manager.get_last_sources()

        assert len(sources) == 1
        assert sources[0]['text'] == "Source"

    def test_reset_sources(self):
        """Test resetting sources from all tools"""
        manager = ToolManager()

        tool1 = Mock()
        tool1.get_tool_definition = Mock(return_value={"name": "tool1"})
        tool1.last_sources = [{"text": "Source 1"}]

        tool2 = Mock()
        tool2.get_tool_definition = Mock(return_value={"name": "tool2"})
        tool2.last_sources = [{"text": "Source 2"}]

        manager.register_tool(tool1)
        manager.register_tool(tool2)

        manager.reset_sources()

        assert tool1.last_sources == []
        assert tool2.last_sources == []

    def test_reset_sources_handles_tools_without_sources(self):
        """Test reset_sources doesn't error on tools without last_sources attribute"""
        manager = ToolManager()

        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={"name": "tool"})
        # No last_sources attribute

        manager.register_tool(mock_tool)

        # Should not raise an error
        manager.reset_sources()

    def test_multiple_tool_registration(self):
        """Test registering multiple different tools"""
        manager = ToolManager()
        mock_store = Mock()

        search_tool = CourseSearchTool(mock_store)
        manager.register_tool(search_tool)

        assert len(manager.tools) == 1
        assert "search_course_content" in manager.tools

    def test_tool_execute_with_no_kwargs(self):
        """Test executing tool with no additional kwargs"""
        manager = ToolManager()

        mock_tool = Mock()
        mock_tool.get_tool_definition = Mock(return_value={"name": "simple_tool"})
        mock_tool.execute = Mock(return_value="Done")

        manager.register_tool(mock_tool)

        result = manager.execute_tool("simple_tool")

        mock_tool.execute.assert_called_once_with()
        assert result == "Done"
