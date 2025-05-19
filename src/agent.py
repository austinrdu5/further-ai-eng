from typing import Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import json

from state import AgentState
from nodes import (
    configure_logging,
    intro,
    router,
    reattempt_live_contact,
    info_collector,
    tour_scheduler,
    knowledge_base
)

NODE_MAP = {
    "intro": intro,
    "router": router,
    "reattempt_live_contact": reattempt_live_contact,
    "info_collector": info_collector,
    "tour_scheduler": tour_scheduler,
    "knowledge_base": knowledge_base
}

def run_agent(state: AgentState, user_message: Optional[str] = None, verbose: bool = False) -> Tuple[str, AgentState]:
    """
    Runs the agent with a user message and returns the response.
    
    Args:
        state: The current state (if continuing a conversation)
        user_message: The user's message
        verbose: Whether to print detailed state information (default: False)
        
    Returns:
        response: The agent's response
        new_state: The updated state
    """
    # Add the user message to the state
    state.messages.append(HumanMessage(content=user_message))

    # Run the agent
    state.accepting_user_input = False
    while not state.accepting_user_input:
        new_state = NODE_MAP[state.next_node](state)
        state = new_state
    
    return state

# Example of how to use the agent
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Run the agent with optional verbose mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose mode with detailed state logging')
    parser.add_argument('--example', '-e', action='store_true', help='Run with example conversation inputs')
    args = parser.parse_args()
    
    # Configure logging once at the start
    configure_logging(args.verbose)
    
    # Initialize state
    conversation_state = AgentState()  # comes with the system prompt and next_node set to intro
    
    # Print greeting
    print("Sophie: Hi, this is ACME Senior Living. My name is Sophie. How may I help you today?")
    
    if args.example:
        example_inputs = [
            "Hello, my name is James and I am looking to learn how much your community costs?",
            "I can continue.",
            "Wow! That is really expensive. do you take Medicaid?",
            "I would like to come for a tour, does next Sunday at 3pm work?",
            "Yes, Tuesday at 2pm might work",
            "What is included in the monthly cost? Do the rooms have individual controlled Air Conditioning? My mom runs hot and she likes to set the temperature very low",
        ]

        for user_input in example_inputs:
            print(f"User: {user_input}")
            run_agent(conversation_state, user_input, verbose=args.verbose)

    else:
        # Interactive mode
        while True:
            try:
                user_input = input("User: ")
                if not user_input.strip():
                    continue

                if user_input.lower().strip() in ["exit", "quit", "bye"]:
                    print("\nGoodbye!")
                    break

                run_agent(conversation_state, user_input, verbose=args.verbose)

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
