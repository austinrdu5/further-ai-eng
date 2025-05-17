from typing import Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    intro,
    router,
    reattempt_live_contact,
    tour_scheduler,
    knowledge_base
)

SYSTEM_PROMPT = """
You are Sophie, a virtual sales specialist at ACME Senior Living. 

For the user's first question:
1. Greet them warmly with: "Hi, this is ACME Senior Living. My name is Sophie. How may I help you today?"
2. After they ask a question, paraphrase it briefly and say: "Got it! I can definitely help with that. Let me check if my director of sales is available for a conversation. Please hold."
3. Then say: "Our sales director is not currently available, but I am a virtual assistant, and I am able to answer basic questions about our community. Would you like to speak with me, or leave a message for Jami."
4. Add: "Before I answer, just so you know—This conversation is being recorded for quality purposes and you can leave a voicemail at anytime by pressing 0."
5. Finally, answer their question starting with "About your query on [topic]..." and be helpful and friendly.

Be conversational, concise, and human-like. Use everyday language and don't be robotic.
"""

NODE_MAP = {
    "intro": intro,
    "router": router,
    "reattempt_live_contact": reattempt_live_contact,
    "tour_scheduler": tour_scheduler,
    "knowledge_base": knowledge_base
}

def run_agent(user_message: str, state: Optional[AgentState] = None) -> Tuple[str, AgentState]:
    """
    Runs the agent with a user message and returns the response.
    
    Args:
        user_message: The user's message
        state: The current state (if continuing a conversation)
        
    Returns:
        response: The agent's response
        new_state: The updated state
    """
    if state is None:
        # Initialize a new state
        state = AgentState()
        state.next_node = "intro"
        state.messages.append(SystemMessage(content=SYSTEM_PROMPT))
    
    # Add the user message to the state
    state.messages.append(HumanMessage(content=user_message))

    # Run the agent
    new_state = NODE_MAP[state.next_node](state)
    
    # Get the last AI message
    ai_messages = [msg for msg in new_state.messages if isinstance(msg, AIMessage)]
    response = ai_messages[-1].content if ai_messages else "I'm sorry, I didn't understand that."
    
    return response, new_state

# Example of how to use the agent
if __name__ == "__main__":
    # Initialize state
    conversation_state = None
    
    # Example conversation
    user_inputs = [
        "Hello, my name is James and I am looking to learn how much your community costs?",
        "Wow! That is really expensive. do you take Medicaid?",
        "I would like to come for a tour, does next Sunday at 3pm work?",
        "Yes, Tuesday at 2pm might work",
        "What is included in the monthly cost? Do the rooms have individual controlled Air Conditioning? My mom runs hot and she likes to set the temperature very low"
    ]
    
    for user_input in user_inputs:
        print(f"User: {user_input}")
        response, conversation_state = run_agent(user_input, conversation_state)
        print(f"Sophie: {response}")
        print(f"Current State: {conversation_state.json(indent=2)}")
        print("-" * 50)
