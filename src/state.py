from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

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
    is_first_message: bool = True
    frustration_detected: bool = False
    time_of_transfer_attempt: Optional[str] = None
    wants_brochure: bool = False
    wants_callback: bool = False
    inquiry_type: Optional[str] = None
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
    next_node: Optional[str] = None  # Used to control flow in the graph
