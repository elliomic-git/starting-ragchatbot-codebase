import pytest
from session_manager import SessionManager, Message


@pytest.mark.unit
class TestSessionManager:
    """Unit tests for SessionManager"""

    def test_create_session_returns_unique_id(self):
        """Test that create_session returns a unique session ID"""
        manager = SessionManager(max_history=5)
        session1 = manager.create_session()
        session2 = manager.create_session()

        assert session1 != session2
        assert session1 == "session_1"
        assert session2 == "session_2"

    def test_create_session_increments_counter(self):
        """Test that session counter increments correctly"""
        manager = SessionManager(max_history=5)

        for i in range(1, 6):
            session_id = manager.create_session()
            assert session_id == f"session_{i}"

    def test_create_session_initializes_empty_list(self):
        """Test that new session has empty message list"""
        manager = SessionManager(max_history=5)
        session_id = manager.create_session()

        assert session_id in manager.sessions
        assert manager.sessions[session_id] == []

    def test_multiple_sessions_are_independent(self):
        """Test that multiple sessions maintain separate message histories"""
        manager = SessionManager(max_history=5)
        session1 = manager.create_session()
        session2 = manager.create_session()

        manager.add_message(session1, "user", "Hello from session 1")
        manager.add_message(session2, "user", "Hello from session 2")

        assert len(manager.sessions[session1]) == 1
        assert len(manager.sessions[session2]) == 1
        assert manager.sessions[session1][0].content == "Hello from session 1"
        assert manager.sessions[session2][0].content == "Hello from session 2"

    def test_add_message_stores_role_and_content(self):
        """Test that add_message correctly stores message with role and content"""
        manager = SessionManager(max_history=5)
        session_id = manager.create_session()

        manager.add_message(session_id, "user", "Test message")

        assert len(manager.sessions[session_id]) == 1
        message = manager.sessions[session_id][0]
        assert isinstance(message, Message)
        assert message.role == "user"
        assert message.content == "Test message"

    def test_add_message_creates_session_if_not_exists(self):
        """Test that add_message creates session if it doesn't exist"""
        manager = SessionManager(max_history=5)

        manager.add_message("new_session", "user", "Test")

        assert "new_session" in manager.sessions
        assert len(manager.sessions["new_session"]) == 1

    def test_add_exchange_adds_both_messages(self):
        """Test that add_exchange adds both user and assistant messages"""
        manager = SessionManager(max_history=5)
        session_id = manager.create_session()

        manager.add_exchange(session_id, "What is ML?", "Machine Learning is...")

        assert len(manager.sessions[session_id]) == 2
        assert manager.sessions[session_id][0].role == "user"
        assert manager.sessions[session_id][0].content == "What is ML?"
        assert manager.sessions[session_id][1].role == "assistant"
        assert manager.sessions[session_id][1].content == "Machine Learning is..."

    def test_history_limit_enforced(self):
        """Test that message history respects max_history limit"""
        manager = SessionManager(max_history=2)  # Only keep 2 exchanges = 4 messages
        session_id = manager.create_session()

        # Add 3 exchanges (6 messages total)
        manager.add_exchange(session_id, "Q1", "A1")
        manager.add_exchange(session_id, "Q2", "A2")
        manager.add_exchange(session_id, "Q3", "A3")

        # Should only keep last 2 exchanges (4 messages)
        assert len(manager.sessions[session_id]) == 4
        assert manager.sessions[session_id][0].content == "Q2"
        assert manager.sessions[session_id][1].content == "A2"
        assert manager.sessions[session_id][2].content == "Q3"
        assert manager.sessions[session_id][3].content == "A3"

    def test_history_trimming_with_odd_messages(self):
        """Test history trimming when adding individual messages"""
        manager = SessionManager(max_history=2)
        session_id = manager.create_session()

        # Add 5 individual messages
        for i in range(5):
            manager.add_message(session_id, "user", f"Message {i}")

        # Should keep only last 4 messages (max_history * 2)
        assert len(manager.sessions[session_id]) == 4
        assert manager.sessions[session_id][0].content == "Message 1"

    def test_get_conversation_history_formats_correctly(self):
        """Test that conversation history is formatted with role labels"""
        manager = SessionManager(max_history=5)
        session_id = manager.create_session()

        manager.add_exchange(session_id, "What is Python?", "Python is a programming language.")

        history = manager.get_conversation_history(session_id)

        assert history is not None
        assert "User: What is Python?" in history
        assert "Assistant: Python is a programming language." in history

    def test_get_conversation_history_multi_turn(self):
        """Test conversation history formatting with multiple exchanges"""
        manager = SessionManager(max_history=5)
        session_id = manager.create_session()

        manager.add_exchange(session_id, "Q1", "A1")
        manager.add_exchange(session_id, "Q2", "A2")

        history = manager.get_conversation_history(session_id)
        lines = history.split("\n")

        assert len(lines) == 4
        assert lines[0] == "User: Q1"
        assert lines[1] == "Assistant: A1"
        assert lines[2] == "User: Q2"
        assert lines[3] == "Assistant: A2"

    def test_get_conversation_history_empty_session_returns_none(self):
        """Test that empty session returns None for history"""
        manager = SessionManager(max_history=5)
        session_id = manager.create_session()

        history = manager.get_conversation_history(session_id)

        assert history is None

    def test_get_conversation_history_nonexistent_session_returns_none(self):
        """Test that nonexistent session returns None"""
        manager = SessionManager(max_history=5)

        history = manager.get_conversation_history("nonexistent")

        assert history is None

    def test_get_conversation_history_none_session_returns_none(self):
        """Test that None session_id returns None"""
        manager = SessionManager(max_history=5)

        history = manager.get_conversation_history(None)

        assert history is None

    def test_clear_session_removes_all_messages(self):
        """Test that clear_session removes all messages from a session"""
        manager = SessionManager(max_history=5)
        session_id = manager.create_session()

        manager.add_exchange(session_id, "Q1", "A1")
        manager.add_exchange(session_id, "Q2", "A2")
        assert len(manager.sessions[session_id]) == 4

        manager.clear_session(session_id)

        assert len(manager.sessions[session_id]) == 0

    def test_clear_session_nonexistent_session_no_error(self):
        """Test that clearing nonexistent session doesn't raise error"""
        manager = SessionManager(max_history=5)

        # Should not raise an exception
        manager.clear_session("nonexistent")

    def test_message_dataclass_creation(self):
        """Test Message dataclass can be created correctly"""
        message = Message(role="user", content="Test content")

        assert message.role == "user"
        assert message.content == "Test content"
