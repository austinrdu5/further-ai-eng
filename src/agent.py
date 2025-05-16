from typing import Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from state import AgentState
from nodes import (
    greeting,
    router,
    tour_scheduler,
    pricing_handler,
    amenities_handler,
    payment_options_handler,
    contact_collector,
    frustration_handler,
    general_handler,
    validation_layer,
    determine_next_step
)

def create_agent_workflow():
    """Creates and configures the LangGraph workflow."""
    
    # Create a new graph
    workflow = StateGraph(AgentState)
    
    # Add all the nodes
    workflow.add_node("greeting", greeting)
    workflow.add_node("router", router)
    workflow.add_node("tour_scheduler", tour_scheduler)
    workflow.add_node("pricing_handler", pricing_handler)
    workflow.add_node("amenities_handler", amenities_handler)
    workflow.add_node("payment_options_handler", payment_options_handler)
    workflow.add_node("contact_collector", contact_collector)
    workflow.add_node("frustration_handler", frustration_handler)
    workflow.add_node("general_handler", general_handler)
    workflow.add_node("validation", validation_layer)
    
    # Set the entry point
    workflow.set_entry_point("greeting")
    
    # Connect nodes with conditional logic
    workflow.add_conditional_edges(
        "greeting",
        determine_next_step,
        {
            "router": "router",
            "tour_scheduler": "tour_scheduler",
            "pricing_handler": "pricing_handler",
            "amenities_handler": "amenities_handler",
            "payment_options_handler": "payment_options_handler",
            "general_handler": "general_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "router",
        determine_next_step,
        {
            "tour_scheduler": "tour_scheduler",
            "pricing_handler": "pricing_handler",
            "amenities_handler": "amenities_handler",
            "payment_options_handler": "payment_options_handler",
            "contact_collector": "contact_collector",
            "frustration_handler": "frustration_handler",
            "general_handler": "general_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "tour_scheduler",
        determine_next_step,
        {
            "tour_scheduler": "tour_scheduler",
            "general_handler": "general_handler",
            "router": "router"
        }
    )
    
    workflow.add_conditional_edges(
        "pricing_handler",
        determine_next_step,
        {
            "tour_scheduler": "tour_scheduler",
            "payment_options_handler": "payment_options_handler",
            "router": "router"
        }
    )
    
    workflow.add_conditional_edges(
        "amenities_handler",
        determine_next_step,
        {
            "contact_collector": "contact_collector",
            "tour_scheduler": "tour_scheduler",
            "router": "router"
        }
    )
    
    workflow.add_conditional_edges(
        "payment_options_handler",
        determine_next_step,
        {
            "contact_collector": "contact_collector",
            "router": "router"
        }
    )
    
    workflow.add_conditional_edges(
        "contact_collector",
        determine_next_step,
        {
            "contact_collector": "contact_collector",
            "general_handler": "general_handler"
        }
    )
    
    workflow.add_conditional_edges(
        "frustration_handler",
        determine_next_step,
        {
            "contact_collector": "contact_collector",
            "router": "router"
        }
    )
    
    workflow.add_conditional_edges(
        "general_handler",
        determine_next_step,
        {
            "contact_collector": "contact_collector",
            "router": "router"
        }
    )
    
    # All paths go through validation before returning to the user
    workflow.add_edge("tour_scheduler", "validation")
    workflow.add_edge("pricing_handler", "validation")
    workflow.add_edge("amenities_handler", "validation")
    workflow.add_edge("payment_options_handler", "validation")
    workflow.add_edge("contact_collector", "validation")
    workflow.add_edge("frustration_handler", "validation")
    workflow.add_edge("general_handler", "validation")
    
    # Add terminal states - after validation, we're done for this turn
    workflow.add_edge("validation", END)
    
    return workflow

# Create the compiled agent
agent_executor = create_agent_workflow().compile()

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
    
    # Add the user message to the state
    state.messages.append(HumanMessage(content=user_message))
    
    # Run the agent
    new_state = agent_executor.invoke(state)
    
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
