import os
import sys
import pytest
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage

# Add project root and src directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)
sys.path.insert(0, project_root)

from state import AgentState, ConversationState, UserInfo

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "intro: mark test as an intro node test")
    config.addinivalue_line("markers", "router: mark test as a router node test")
    config.addinivalue_line("markers", "reattempt_live_contact: mark test as a reattempt_live_contact node test")
    config.addinivalue_line("markers", "info_collector: mark test as an info_collector node test")
    config.addinivalue_line("markers", "tour_scheduler: mark test as a tour_scheduler node test")
    config.addinivalue_line("markers", "knowledge_base: mark test as a knowledge_base node test")

@pytest.fixture
def base_state():
    """Create a base state for testing."""
    return AgentState()

@pytest.fixture
def intro_state_with_history():
    """Create a state with conversation history for intro testing."""
    state = AgentState()
    state.messages.append([AIMessage(content="Hi, this is ACME Senior Living. My name is Sophie. How may I help you today?"),
                           HumanMessage(content="Hello"),
                           HumanMessage(content="What are your prices?")
                           ])
    return state

@pytest.fixture
def intro_state_with_parsing_failure():
    """Create a state with parsing failure for intro testing."""
    state = AgentState()
    state.messages.append(HumanMessage(content="I need help"))
    state.failed_parsing = True
    state.n_parsing_fails = 1
    return state

@pytest.fixture
def intro_state_with_max_failures():
    """Create a state with maximum parsing failures for intro testing."""
    state = AgentState()
    state.messages.append([HumanMessage(content="I need help")])
    state.failed_parsing = True
    state.n_parsing_fails = 3
    return state 