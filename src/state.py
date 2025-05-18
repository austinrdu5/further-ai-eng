from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
import logging

class UserInfo(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    preferred_contact_time: Optional[str] = None
    preferred_care_type: Optional[str] = None  # "assisted_living", "independent_living"
    resident_relationship: Optional[str] = None  # "self", "parent", "spouse", etc.
    extra_information: Dict[str, str] = Field(default_factory=dict)  # For any additional user preferences/requirements

class ConversationState(BaseModel):
    disclosure_given: bool = False
    frustration_detected: bool = False
    time_of_transfer_attempt: Optional[str] = None
    wants_brochure: bool = False
    wants_callback: bool = False
    inquiry_types: List[str] = Field(default_factory=list)
    tour_scheduling_attempts: int = 0
    tour_attempted_dates: List[str] = Field(default_factory=list)
    tour_scheduled: bool = False
    tour_date: Optional[str] = None
    tour_time: Optional[str] = None
    

class AgentState(BaseModel):
    """The state of our agent."""
    user_info: UserInfo = Field(default_factory=UserInfo)
    conversation_state: ConversationState = Field(default_factory=ConversationState)
    messages: List[Union[HumanMessage, AIMessage]] = Field(default_factory=list)
    failed_parsing: bool = False
    n_parsing_fails: int = 0
    next_node: Optional[str] = None  # Used to control flow in the graph

def handle_failed_parsing(state: AgentState, logger: logging.Logger) -> AgentState:
    state.failed_parsing = True
    state.n_parsing_fails += 1

    # If Sophie has failed to parse the user's message 3 times in a row, apologize and end the conversation
    if state.n_parsing_fails >= 3:
        logger.error("Failed to parse user message 3 times in a row, ending conversation")
        print("Sophie: I apologize, but I'm having trouble processing your request. Please call 850-445-8362 for assistance.")
        state.next_node = None
        return state
    
    # If normal failure, send to router
    else:
        logger.warning("Failed to parse user message, retrying router")
        failed_parsing_message = "Sophie: I apologize, but I'm having trouble processing your request. Let me try again."
        print(failed_parsing_message)
        state.messages += [AIMessage(content=failed_parsing_message)]
        state.next_node = "router"
        return state