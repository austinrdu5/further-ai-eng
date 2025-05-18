import pytest
import re
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from state import AgentState, ConversationState, UserInfo
from nodes import (
    intro,
    router,
    reattempt_live_contact,
    info_collector,
    tour_scheduler,
    knowledge_base
)

# Intro Node Tests
@pytest.mark.intro
def test_intro_node_phone_request(base_state):
    """Test intro node handling phone number request."""
    # Verify initial state
    assert len(base_state.messages) > 0
    assert isinstance(base_state.messages[0], SystemMessage)
    assert "You are a helpful senior living agent named Sophie" in base_state.messages[0].content
    assert base_state.next_node == "intro"
    
    # Setup
    base_state.messages += [HumanMessage(content="What's the community phone number?")]
    
    # Execute
    result = intro(base_state)
    
    # Verify
    assert result.next_node == "intro"
    assert re.search(r"8\s*5\s*0\s*[-()]*\s*4\s*4\s*5\s*[-()]*\s*8\s*3\s*6\s*2", result.messages[-1].content)

@pytest.mark.intro
def test_intro_node_employment_request(base_state):
    """Test intro node handling employment request."""
    # Setup
    base_state.messages = [HumanMessage(content="Are you hiring?")]
    
    # Execute
    result = intro(base_state)
    
    # Verify
    assert result.next_node == "intro"
    assert "careers" in result.messages[-1].content.lower()

@pytest.mark.intro
def test_intro_node_callback_request(base_state):
    """Test intro node handling callback request."""
    # Setup
    base_state.messages = [HumanMessage(content="Can someone call me back?")]
    
    # Execute
    result = intro(base_state)
    
    # Verify
    assert result.next_node == "router"

@pytest.mark.intro
def test_intro_node_state_transition_to_router(base_state):
    """Test intro node state transition to router."""
    # Setup
    base_state.messages = [HumanMessage(content="I need information about pricing")]
    
    # Execute
    result = intro(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert "director" in result.messages[-1].content.lower()
    assert not result.failed_parsing
    assert result.n_parsing_fails == 0

# Router Node Tests
@pytest.mark.router
def test_router_first_message_disclosure(base_state):
    """Test router adds disclosure for first message."""
    # Setup
    base_state.messages = [HumanMessage(content="Hello")]
    base_state.conversation_state.disclosure_given = False
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert "conversation is being recorded" in result.messages[-1].content
    assert result.conversation_state.disclosure_given

@pytest.mark.router
def test_router_phone_request(base_state):
    """Test router handling phone number request."""
    # Setup
    base_state.messages = [HumanMessage(content="What's your phone number?")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "knowledge_base"
    assert "community_info" in result.conversation_state.inquiry_types

@pytest.mark.router
def test_router_tour_request(base_state):
    """Test router handling tour request."""
    # Setup
    base_state.messages = [HumanMessage(content="I'd like to schedule a tour")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "tour_scheduler"

@pytest.mark.router
def test_router_floorplan_request(base_state):
    """Test router handling floorplan request."""
    # Setup
    base_state.messages = [HumanMessage(content="Can I see the floorplans?")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.wants_brochure == True

@pytest.mark.router
def test_router_frustration(base_state):
    """Test router handling frustration."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "reattempt_live_contact"

@pytest.mark.router
def test_router_employment_request(base_state):
    """Test router handling employment request."""
    # Setup
    base_state.messages = [HumanMessage(content="Are you hiring?")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "knowledge_base"
    assert "employment" in result.conversation_state.inquiry_types

@pytest.mark.router
def test_router_callback_request(base_state):
    """Test router handling callback request."""
    # Setup
    base_state.messages = [HumanMessage(content="Can someone call me back?")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.wants_callback == True

@pytest.mark.router
def test_router_knowledge_base_routing(base_state):
    """Test router handling knowledge base routing."""
    # Setup
    base_state.messages = [HumanMessage(content="Do you have a pool?")]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "knowledge_base"
    assert "amenities" in result.conversation_state.inquiry_types

# Reattempt_live_contact Node Tests
@pytest.mark.reattempt_live_contact
def test_reattempt_live_contact_first_attempt(base_state):
    """Test reattempt_live_contact on first attempt."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    base_state.conversation_state.time_of_transfer_attempt = datetime.now() - timedelta(minutes=3)
    
    # Execute
    with patch('nodes.attempt_transfer', return_value=False):
        result = reattempt_live_contact(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.wants_callback == True
    assert "sales director is not currently available" in result.messages[-1].content

@pytest.mark.reattempt_live_contact
def test_reattempt_live_contact_quick_retry(base_state):
    """Test reattempt_live_contact when retrying too quickly."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    base_state.conversation_state.time_of_transfer_attempt = datetime.now() - timedelta(minutes=1)
    
    # Execute
    result = reattempt_live_contact(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert len(result.messages) == 1  # No additional messages added
    assert result.conversation_state.wants_callback == True

@pytest.mark.reattempt_live_contact
def test_reattempt_live_contact_successful_transfer(base_state):
    """Test reattempt_live_contact with successful transfer."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    base_state.conversation_state.time_of_transfer_attempt = datetime.now() - timedelta(minutes=3)
    
    # Execute
    with patch('nodes.attempt_transfer', return_value=True):
        result = reattempt_live_contact(base_state)
    
    # Verify
    assert result.next_node is None  # Conversation ends
    assert len(result.messages) == 1  # No additional messages added
    assert result.conversation_state.wants_callback == True

@pytest.mark.reattempt_live_contact
def test_reattempt_live_contact_no_previous_attempt(base_state):
    """Test reattempt_live_contact when no previous attempt exists."""
    # Setup
    base_state.messages = [HumanMessage(content="I want to talk to a real person!")]
    base_state.conversation_state.time_of_transfer_attempt = None
    
    # Execute
    with patch('nodes.attempt_transfer', return_value=False):
        result = reattempt_live_contact(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.conversation_state.time_of_transfer_attempt is not None
    assert result.conversation_state.wants_callback == True
    assert "sales director is not currently available" in result.messages[-1].content

# Info_collector Node Tests
@pytest.mark.info_collector
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

@pytest.mark.info_collector
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

@pytest.mark.info_collector
def test_info_collector_brochure_request(base_state):
    """Test info_collector handling brochure request."""
    # Setup
    base_state.messages = [HumanMessage(content="I'd like a brochure")]
    base_state.conversation_state.wants_brochure = True
    
    # Execute
    result = info_collector(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert "brochure" in result.messages[-1].content.lower()

# Tour_scheduler Node Tests
@pytest.mark.tour_scheduler
def test_tour_scheduler_first_attempt(base_state):
    """Test tour_scheduler on first attempt."""
    # Setup
    base_state.messages = [HumanMessage(content="I'd like to schedule a tour")]
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "tour_scheduler"
    assert result.conversation_state.tour_scheduling_attempts == 1

@pytest.mark.tour_scheduler
def test_tour_scheduler_max_attempts(base_state):
    """Test tour_scheduler after max attempts."""
    # Setup
    base_state.messages = [HumanMessage(content="I'd like to schedule a tour")]
    base_state.conversation_state.tour_scheduling_attempts = 3
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "reattempt_live_contact"

@pytest.mark.tour_scheduler
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
    assert result.conversation_state.tour_scheduled == True
    assert result.user_info.first_name == "John"
    assert result.user_info.email == "john@example.com"

@pytest.mark.tour_scheduler
def test_tour_scheduler_date_time_handling(base_state):
    """Test tour_scheduler handling specific date/time."""
    # Setup
    base_state.messages = [
        HumanMessage(content="I'd like to schedule a tour"),
        HumanMessage(content="Next Monday at 3pm")
    ]
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert result.conversation_state.tour_scheduling_attempts == 1
    assert result.conversation_state.tour_scheduled == True
    assert "Monday" in result.messages[-1].content
    assert "3pm" in result.messages[-1].content

# Knowledge_base Node Tests
@pytest.mark.knowledge_base
def test_knowledge_base_pricing(base_state):
    """Test knowledge_base handling pricing inquiry."""
    # Setup
    base_state.messages = [HumanMessage(content="How much does it cost?")]
    base_state.conversation_state.inquiry_types = ["pricing"]
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert "$" in result.messages[-1].content

@pytest.mark.knowledge_base
def test_knowledge_base_unknown_topic(base_state):
    """Test knowledge_base handling unknown topic."""
    # Setup
    base_state.messages = [HumanMessage(content="What's the weather like?")]
    base_state.conversation_state.inquiry_types = ["uncategorized"]
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert "only have information about" in result.messages[-1].content.lower()

@pytest.mark.knowledge_base
def test_knowledge_base_community_details(base_state):
    """Test knowledge_base handling community details inquiry."""
    # Setup
    base_state.messages = [HumanMessage(content="Tell me about the community")]
    base_state.conversation_state.inquiry_types = ["community_details"]
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert "community" in result.messages[-1].content.lower()

@pytest.mark.knowledge_base
def test_knowledge_base_financing(base_state):
    """Test knowledge_base handling financing inquiry."""
    # Setup
    base_state.messages = [HumanMessage(content="What financing options do you offer?")]
    base_state.conversation_state.inquiry_types = ["financing"]
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert "medicaid" in result.messages[-1].content.lower()

@pytest.mark.knowledge_base
def test_knowledge_base_continue_vs_redirect(base_state):
    """Test knowledge_base handling user choice to continue vs redirect."""
    # Setup
    base_state.messages = [
        HumanMessage(content="What's the weather like?"),
        HumanMessage(content="No, I'll continue with you")
    ]
    # base_state.conversation_state.inquiry_types = ["uncategorized"]
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert "continue" in result.messages[-1].content.lower()

def test_router_multiple_inquiries(base_state):
    """Test router handling multiple inquiry types in sequence."""
    # Setup
    base_state.messages = [
        HumanMessage(content="What are your prices?"),
        HumanMessage(content="And can I see the floorplans?"),
        HumanMessage(content="Actually, I'd like to schedule a tour")
    ]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "tour_scheduler"
    assert "pricing" in result.conversation_state.inquiry_types  # Should retain last inquiry type

def test_router_edge_case_classification(base_state):
    """Test router handling edge cases in inquiry classification."""
    # Setup
    base_state.messages = [
        HumanMessage(content=""),  # Empty message
        HumanMessage(content="..."),  # Just punctuation
        HumanMessage(content="I don't know what to ask")  # Ambiguous message
    ]
    
    # Execute
    result = router(base_state)
    
    # Verify
    assert result.next_node == "knowledge_base"
    assert "uncategorized" in result.conversation_state.inquiry_types

def test_info_collector_partial_info(base_state):
    """Test info_collector with partial information."""
    # Setup
    base_state.messages = [
        HumanMessage(content="My name is John Smith"),
        HumanMessage(content="My email is john@example.com")
    ]
    
    # Execute
    result = info_collector(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert result.user_info.first_name == "John"
    assert result.user_info.last_name == "Smith"
    assert result.user_info.email == "john@example.com"
    assert not result.user_info.phone
    assert "phone number" in result.messages[-1].content.lower()

def test_info_collector_invalid_format(base_state):
    """Test info_collector handling invalid information formats."""
    # Setup
    base_state.messages = [
        HumanMessage(content="My name is 123"),  # Invalid name
        HumanMessage(content="My email is not-an-email"),  # Invalid email
        HumanMessage(content="My phone is abc-def-ghij")  # Invalid phone
    ]
    
    # Execute
    result = info_collector(base_state)
    
    # Verify
    assert result.next_node == "info_collector"
    assert not result.user_info.first_name
    assert not result.user_info.email
    assert not result.user_info.phone
    assert "valid" in result.messages[-1].content.lower()

def test_tour_scheduler_invalid_datetime(base_state):
    """Test tour_scheduler handling invalid date/time formats."""
    # Setup
    base_state.messages = [
        HumanMessage(content="I'd like to schedule a tour"),
        HumanMessage(content="Next year at 25:00"),  # Invalid time
        HumanMessage(content="On the 32nd of January")  # Invalid date
    ]
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "tour_scheduler"
    assert result.conversation_state.tour_scheduling_attempts == 1
    assert "valid" in result.messages[-1].content.lower()

def test_tour_scheduler_unavailable_slot(base_state):
    """Test tour_scheduler handling unavailable time slots."""
    # Setup
    base_state.messages = [
        HumanMessage(content="I'd like to schedule a tour"),
        HumanMessage(content="Tomorrow at 3am"),  # Unavailable time
        HumanMessage(content="Next Sunday at 2pm")  # Unavailable day
    ]
    
    # Execute
    result = tour_scheduler(base_state)
    
    # Verify
    assert result.next_node == "tour_scheduler"
    assert result.conversation_state.tour_scheduling_attempts == 1
    assert "available" in result.messages[-1].content.lower()

def test_knowledge_base_sequential_questions(base_state):
    """Test knowledge_base handling multiple related questions in sequence."""
    # Setup
    base_state.messages = [
        HumanMessage(content="What are your prices?"),
        HumanMessage(content="And what financing options do you offer?"),
        HumanMessage(content="Can you tell me more about the community amenities?")
    ]
    base_state.conversation_state.inquiry_types = ["pricing"]
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert len(result.messages) == 4  # Original messages + response
    assert "pricing" in result.messages[-1].content.lower()
    assert "financing" in result.messages[-1].content.lower()
    assert "amenities" in result.messages[-1].content.lower()

def test_knowledge_base_edge_case_responses(base_state):
    """Test knowledge_base handling edge cases in responses."""
    # Setup
    base_state.messages = [
        HumanMessage(content="What's the exact price for a 2-bedroom unit?"),
        HumanMessage(content="Can you guarantee the price won't change?"),
        HumanMessage(content="What's the best unit you have?")
    ]
    base_state.conversation_state.inquiry_types = ["pricing"]
    
    # Execute
    result = knowledge_base(base_state)
    
    # Verify
    assert result.next_node == "router"
    assert len(result.messages) == 4
    assert "specific" in result.messages[-1].content.lower()
    assert "guarantee" in result.messages[-1].content.lower()
    assert "best" in result.messages[-1].content.lower()
