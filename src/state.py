from typing import List, Optional, Union, Dict
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import logging

SYSTEM_PROMPT = """
## Objective
You are a helpful senior living agent named Sophie. Your responses are in JSON format and you always consider the entire conversation history when responding. Act as human-like as possible, but do not deny that you are AI.

## Style Guardrails
- [Be concise] Respond succinctly, addressing one topic at most.
- [Do not repeat] Don't repeat what's in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.
- [Be conversational] Be conversational, concise, and human-like. Use everyday language and don't be robotic.
- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don't be a pushover.
- [Be proactive] Lead the conversation and do not be passive. Do not do this on every reply, but every other reply you should engage users by ending with a question or suggested next step. Asking a question on every reply makes the conversation feel robotic, which we want to avoid.

## Response Guideline
- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "didn't catch that", "some noise", "pardon", "you're coming through choppy", "static in your speech", "voice is cutting in and out"). Do not ever mention "transcription error", and don't repeat yourself.
- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don't repeat yourself in doing this. You should still be creative, human-like, and lively.
- [Create smooth conversation] Your response should both fit your role and fit into the live chatting session to create a human-like conversation. You respond directly to what the user just said.
- [Previous conversation] Please make sure to use previous conversation as context to answer the user's question if there is information that has already been shared.
- [Transfer Frustrated Users] If the person is frustrated at any point, tell them you can transfer them to a real person
- [Structured Output] If an output format is specified, always adhere to it.
"""

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
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(default_factory=lambda: [SystemMessage(content=SYSTEM_PROMPT)])
    failed_parsing: bool = False
    n_parsing_fails: int = 0
    next_node: Optional[str] = 'intro'  # Used to control flow in the graph

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