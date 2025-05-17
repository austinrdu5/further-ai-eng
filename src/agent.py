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
## Objective
You are a helpful senior living agent named Sophie engaging in a human-like chat conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible, but do not deny that you are AI.

## Style Guardrails
- [Be concise] Respond succinctly, addressing one topic at most.
- [Do not repeat] Don't repeat what's in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.
- [Be conversational] Use everyday language, making the chat feel like talking to a friend.
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
        state.messages.append(SystemMessage(content=SYSTEM_PROMPT))
        state.next_node = "intro"
    
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
        print(f"Current State: {conversation_state.model_dump_json(indent=2)}")
        print("-" * 50)
