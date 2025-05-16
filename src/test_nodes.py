import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage

from state import AgentState, ConversationState, UserInfo, Inquiry
from nodes import (
    intro,
    router,
    reattempt_live_contact,
    info_collector,
    tour_scheduler,
    knowledge_base,
    determine_next_step
)

@pytest.fixture
def base_state():
    """Create a base state for testing."""
    return AgentState(
        messages=[],
        conversation_state=ConversationState(),
        user_info=UserInfo(),
        inquiry=Inquiry()
    )

def test_intro_node_phone_request(base_state):
    """Test intro node handling phone number request."""
    # Setup
    base_state.messages = [HumanMessage(content="What's your phone number?")]
    
    # Execute
    result = intro(base_state)
    
    # Verify
    assert result.next_node == "intro"
    assert len(result.messages) == 2  # User message + AI response
    assert "850-445-8362" in result.messages[1].content

def test_intro_node_employment_request(base_state):
    """Test intro node handling employment request."""
    # Setup
    base_state.messages = [HumanMessage(content="Are you hiring?")]
    
    # Execute
    result = intro(base_state)
    
    # Verify
    assert result.next_node == "intro"
    assert len(result.messages) == 2
    assert "careers" in result.messages[1].content.lower()

def test_intro_node_callback_request(base_state):
    """Test intro node handling callback request."""
    # Setup
    base_state.messages = [HumanMessage(content="Can someone call me back?")]
    
    # Execute
    result = intro(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.wants_callback == True

def test_router_first_message_disclosure(base_state):
    """Test router adds disclosure for first message."""
    # Setup
    base_state.messages = [HumanMessage(content="Hello")]
    base_state.conversation_state.is_first_message = True
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert len(result.messages) == 2
    assert "conversation is being recorded" in result.messages[1].content
    assert not result.conversation_state.is_first_message

def test_router_phone_request(base_state):
    """Test router handling phone number request."""
    # Setup
    base_state.messages = [HumanMessage(content="What's your phone number?")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert "850-445-8362" in result.messages[-1].content

def test_router_tour_request(base_state):
    """Test router handling tour request."""
    # Setup
    base_state.messages = [HumanMessage(content="I'd like to schedule a tour")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "tour_scheduler"

def test_router_floorplan_request(base_state):
    """Test router handling floorplan request."""
    # Setup
    base_state.messages = [HumanMessage(content="Can I see the floorplans?")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.wants_brochure == True

def test_router_frustration(base_state):
    """Test router handling frustration."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "reattempt_live_contact"

def test_reattempt_live_contact_first_attempt(base_state):
    """Test reattempt_live_contact on first attempt."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    
    # Execute
    result = reattempt_live_contact(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.wants_callback == True
    assert len(result.messages) == 2  # Original message + pause message
    assert "[10 second pause]" in result.messages[1].content

def test_reattempt_live_contact_quick_retry(base_state):
    """Test reattempt_live_contact when retrying too quickly."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    base_state.conversation_state.time_of_transfer_attempt = datetime.now() - timedelta(minutes=1)
    
    # Execute
    result = reattempt_live_contact(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert len(result.messages) == 1  # No pause message added

def test_info_collector_incomplete(base_state):
    """Test info_collector with incomplete information."""
    # Setup
    base_state.messages = [HumanMessage(content="My name is John")]
    
    # Execute
    result = info_collector(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.user_info.first_name == "John"
    assert not result.user_info.email
    assert not result.user_info.phone

def test_info_collector_complete(base_state):
    """Test info_collector with complete information."""
    # Setup
    base_state.messages = [
        HumanMessage(content="My name is John Smith"),
        HumanMessage(content="My email is john@example.com"),
        HumanMessage(content="My phone is 555-123-4567")
    ]
    
    # Execute
    result = info_collector(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert result.user_info.first_name == "John"
    assert result.user_info.last_name == "Smith"
    assert result.user_info.email == "john@example.com"
    assert result.user_info.phone == "555-123-4567"

def test_tour_scheduler_first_attempt(base_state):
    """Test tour_scheduler on first attempt."""
    # Setup
    base_state.messages = [HumanMessage(content="I'd like to schedule a tour")]
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "tour_scheduler"
    assert result.conversation_state.tour_scheduling_attempts == 1

def test_tour_scheduler_max_attempts(base_state):
    """Test tour_scheduler after max attempts."""
    # Setup
    base_state.messages = [HumanMessage(content="I'd like to schedule a tour")]
    base_state.conversation_state.tour_scheduling_attempts = 3
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.wants_callback == True

def test_tour_scheduler_successful(base_state):
    """Test tour_scheduler with successful scheduling."""
    # Setup
    base_state.messages = [
        HumanMessage(content="I'd like to schedule a tour"),
        HumanMessage(content="Tomorrow at 2pm works"),
        HumanMessage(content="My name is John Smith"),
        HumanMessage(content="john@example.com"),
        HumanMessage(content="555-123-4567")
    ]
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert result.inquiry.tour_scheduled == True
    assert result.user_info.first_name == "John"
    assert result.user_info.email == "john@example.com"

def test_knowledge_base_pricing(base_state):
    """Test knowledge_base handling pricing inquiry."""
    # Setup
    base_state.messages = [HumanMessage(content="How much does it cost?")]
    base_state.conversation_state.inquiry_type = "pricing"
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert len(result.messages) == 2
    assert "$" in result.messages[1].content

def test_knowledge_base_unknown_topic(base_state):
    """Test knowledge_base handling unknown topic."""
    # Setup
    base_state.messages = [HumanMessage(content="What's the weather like?")]
    base_state.conversation_state.inquiry_type = "uncategorized"
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "reattempt_live_contact"
    assert "only have information about" in result.messages[1].content.lower()

def test_determine_next_step_with_next_node(base_state):
    """Test determine_next_step when next_node is set."""
    # Setup
    base_state.next_node = "tour_scheduler"
    
    # Execute
    result = determine_next_step(base_state)
    
    # Verify
    assert result == "tour_scheduler"
    assert base_state.next_node is None

def test_determine_next_step_default(base_state):
    """Test determine_next_step default behavior."""
    # Setup
    base_state.next_node = None
    
    # Execute
    result = determine_next_step(base_state)
    
    # Verify
    assert result == "router" 