import json
import re
import logging
from typing import Optional, Literal, Callable, TypeVar, Any, List, Tuple
from functools import wraps
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.exceptions import OutputParserException

from state import AgentState
from knowledge import KNOWLEDGE_BASE, CURRENT_DATETIME

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# LLM setup
llm = ChatOpenAI(temperature=0.1, model="gpt-4")  # Lower temperature for more structured outputs

def structured_invoke(llm: ChatOpenAI, 
                     messages: List[Tuple[str, str]],
                     parser: StructuredOutputParser, 
                     max_retries: int = 3) -> dict:
    """
    Helper function to invoke LLM with retries for structured output.
    
    Args:
        llm: The LLM instance to use
        messages: List of (role, content) message tuples
        parser: The StructuredOutputParser to use
        max_retries: Maximum number of retry attempts
        
    Returns:
        Parsed response dictionary on success or apologetic response dictionary on failure
    """
    for attempt in range(max_retries):
        try:
            response = llm.invoke(messages)
            return parser.parse(response.content)
        except OutputParserException as e:
            logger.error(f"OutputParserException on attempt {attempt + 1}/{max_retries}: {e}")
            continue

    logger.error("All retry attempts failed")
    return {
        "response": "I apologize, but I'm having trouble processing your request. Let me connect you with someone who can help.",
        "next_node": "reattempt_live_contact"
    }

def intro(state: AgentState) -> AgentState:
    """First interaction with the user."""
    
    # If intro is already completed, just pass through
    if state.conversation_state.intro_completed:
        state.next_node = "router"
        return state
    
    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="response", description="The full response message to the user"),
        ResponseSchema(name="next_node", description="The next node to route to (must be either 'intro' or 'router')")
    ])
    
    # Define the system prompt for the initial greeting and routing
    intro_template = """
    Last user message: {user_message}
    
    Instructions:
    1. Rephrase the user's query to confirm understanding: Got it! I can definitely help with that.
    2. Analyze the user's question for the following cases:
        - If asking for community phone number or about existing vendor/resident:
            - Provide number: (850)-445-8362
            - Ask if they have other questions
            - Set next_node to "intro"
        - If asking about employment:
            - Direct to careers page: https://www.talkfurther.com/events-demo
            - Ask if they have other questions  
            - Set next_node to "intro"
        - For all other queries:
            - "Let me check if my director of sales is available for a conversation. Please hold."
            - Set next_node to "router"
    3. Be conversational, concise, and human-like. Use everyday language and don't be robotic.
    
    {format_instructions}
    """
    
    old_messages = state.messages[:-1].to_messages()
    new_message = intro_template.format(
        user_message=state.messages[-1].content,
        format_instructions=parser.get_format_instructions()
    )
    
    # Get structured response with retries
    parsed_response = structured_invoke(llm, old_messages + [('user', new_message)], parser)
    
    # Whether success or fail, parsed_response is a dictionary with keys "response" and "next_node"
    response = parsed_response["response"]
    next_node = parsed_response["next_node"]
    
    # Update state with the response
    state.messages += [AIMessage(content=response)]
    state.next_node = next_node
        
    # If next node is info_collector or reattempt_live_contact, set wants_callback flag
    if state.next_node in ["info_collector", "reattempt_live_contact"]:
        state.conversation_state.wants_callback = True
    
    return state

def router(state: AgentState) -> AgentState:
    """Routes the conversation to the appropriate handler based on the latest message."""
    
    # Get the latest message from the user
    if not state.messages or len(state.messages) < 2:
        # Default to router if there's not enough context
        state.next_node = "knowledge_base"
        return state
    
    # Get the last user message
    last_messages = [msg for msg in state.messages if isinstance(msg, HumanMessage)]
    if not last_messages:
        state.next_node = "knowledge_base"
        return state
    
    last_user_message = last_messages[-1].content
    
    # If this is the first message after intro, add disclosure
    if state.conversation_state.is_first_message:
        disclosure = "Before I answer, just so you know—This conversation is being recorded for quality purposes and you can leave a voicemail at anytime by pressing 0."
        state.messages.append(AIMessage(content=disclosure))
        state.conversation_state.is_first_message = False
    
    # Create a routing prompt to determine the topic
    routing_prompt = """
    Based on the user's message: "{user_message}"
    
    Determine which category this request falls into:
    
    1. Community phone number or existing vendor/resident inquiry
    2. Employment inquiry
    3. Callback request
    4. Tour request
    5. Floorplan request
    6. Frustration with AI
    7. Other inquiry (pricing, community details, financing, or uncategorized)
    
    Respond with ONLY one of: "phone", "employment", "callback", "tour", "floorplan", "frustration", or "other"
    """
    
    # Get the routing decision
    category = llm.invoke(
        routing_prompt.format(user_message=last_user_message)
    ).content.strip().lower()
    
    # Set the next node based on the category
    if category == "phone":
        state.next_node = "router"  # Stay in router to handle phone number request
    elif category == "employment":
        state.next_node = "router"  # Stay in router to handle employment request
    elif category == "callback":
        state.next_node = "info_collector"
        state.conversation_state.wants_callback = True
    elif category == "tour":
        state.next_node = "tour_scheduler"
    elif category == "floorplan":
        state.next_node = "info_collector"
        state.conversation_state.wants_brochure = True
    elif category == "frustration":
        state.next_node = "reattempt_live_contact"
    else:
        state.next_node = "knowledge_base"
        # Set inquiry type for knowledge base
        if "pricing" in last_user_message.lower():
            state.conversation_state.inquiry_type = "pricing"
        elif "community" in last_user_message.lower():
            state.conversation_state.inquiry_type = "community_details"
        elif "financing" in last_user_message.lower():
            state.conversation_state.inquiry_type = "financing"
        else:
            state.conversation_state.inquiry_type = "uncategorized"
    
    return state

def reattempt_live_contact(state: AgentState) -> AgentState:
    """Handles reattempting live contact for frustrated users."""
    
    # Set callback flag
    state.conversation_state.wants_callback = True
    
    # Check if enough time has passed since last attempt
    if not hasattr(state.conversation_state, 'time_of_transfer_attempt'):
        state.conversation_state.time_of_transfer_attempt = CURRENT_DATETIME
    
    time_since_last_attempt = CURRENT_DATETIME - state.conversation_state.time_of_transfer_attempt
    
    if time_since_last_attempt.total_seconds() > 120:  # 2 minutes
        # Simulate transfer attempt
        state.messages.append(AIMessage(content="[10 second pause]"))
        state.conversation_state.time_of_transfer_attempt = CURRENT_DATETIME
    
    # Proceed to info collector
    state.next_node = "info_collector"
    return state

def info_collector(state: AgentState) -> AgentState:
    """Collects contact information from the user."""
    
    info_collector_prompt = """
    You are Sophie, a virtual sales specialist at ACME Senior Living.
    
    Your task is to collect contact information from the user. Ask for the following information (one at a time):
    
    1. Their name (first and last)
    2. Their email address
    3. Their phone number
    4. Their address (if needed)
    
    Once you have all this information, thank them and ask if there's anything else you can help with.
    
    Be conversational and human-like. Respond directly to what they just said.
    """
    
    # Get the conversation history
    messages = state.messages
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", info_collector_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Get response from LLM
    response = llm.invoke(
        prompt.invoke({"messages": messages}).to_messages()
    )
    
    # Update state with the response
    state.messages.append(AIMessage(content=response.content))
    
    # Extract contact information from the conversation
    extraction_prompt = """
    Based on the conversation so far, extract the following information if available:
    
    1. First name
    2. Last name
    3. Email address
    4. Phone number
    5. Address
    
    Format your response as valid JSON. If a piece of information is not available, use null.
    
    Example:
    {
        "first_name": "John",
        "last_name": "Smith",
        "email": "john@example.com",
        "phone": "555-123-4567",
        "address": "123 Main St"
    }
    """
    
    extraction_result = llm.invoke(
        extraction_prompt + "\n\nConversation:\n" + 
        "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Sophie'}: {msg.content}" for msg in messages])
    ).content
    
    # Parse the extraction result
    json_match = re.search(r'{.*}', extraction_result, re.DOTALL)
    if json_match:
        try:
            extracted_data = json.loads(json_match.group(0))
            
            # Update the state with the extracted information
            if extracted_data.get("first_name"):
                state.user_info.first_name = extracted_data["first_name"]
            if extracted_data.get("last_name"):
                state.user_info.last_name = extracted_data["last_name"]
            if extracted_data.get("email"):
                state.user_info.email = extracted_data["email"]
            if extracted_data.get("phone"):
                state.user_info.phone = extracted_data["phone"]
            if extracted_data.get("address"):
                state.user_info.address = extracted_data["address"]
        except json.JSONDecodeError:
            # If JSON parsing fails, just continue without updating state
            pass
    
    # Check if all required information has been collected
    all_collected = (
        state.user_info.first_name and 
        state.user_info.email and 
        state.user_info.phone
    )
    
    # Determine the next node
    if all_collected:
        state.next_node = "router"
    else:
        state.next_node = "info_collector"
    
    return state

def tour_scheduler(state: AgentState) -> AgentState:
    """Handles tour scheduling requests."""
    
    # Initialize tour scheduling attempts if not already set
    if not hasattr(state.conversation_state, 'tour_scheduling_attempts'):
        state.conversation_state.tour_scheduling_attempts = 0
    
    # Check if we've exceeded the maximum attempts
    if state.conversation_state.tour_scheduling_attempts >= 3:
        state.next_node = "info_collector"
        state.conversation_state.wants_callback = True
        return state
    
    tour_scheduler_prompt = """
    You are Sophie, a virtual sales specialist at ACME Senior Living.
    
    Current date and time: {current_datetime}
    
    Available tour times:
    - Monday to Friday
    - Between 9:00 AM and 6:00 PM
    
    Your task is to schedule a tour for the potential resident. Follow these steps:
    
    1. If you don't know their availability yet, ask when they would like to schedule the tour.
    2. If they propose a time outside your available hours, politely inform them of your available times and ask for an alternative.
    3. Once a valid time is agreed upon, confirm the tour date and time.
    4. Ask for their name, email, and phone number (one at a time).
    5. Once all information is collected, confirm the tour details and ask if there's anything else they'd like to know.
    
    The user's previous messages and your conversation history will help you determine where they are in this process.
    
    Be conversational and human-like. Respond directly to what they just said.
    """
    
    # Get the conversation history
    messages = state.messages
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", tour_scheduler_prompt.format(current_datetime=CURRENT_DATETIME)),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Get response from LLM
    response = llm.invoke(
        prompt.invoke({"messages": messages}).to_messages()
    )
    
    # Update state with the response
    state.messages.append(AIMessage(content=response.content))
    
    # Extract information from the response to update the state
    extraction_prompt = """
    Based on the conversation so far, extract the following information if available:
    
    1. Has a tour been scheduled? (true/false)
    2. Tour date (if scheduled)
    3. Tour time (if scheduled)
    4. First name
    5. Last name
    6. Email
    7. Phone
    
    Format your response as valid JSON. If a piece of information is not available, use null.
    
    Example:
    {
        "tour_scheduled": true,
        "tour_date": "2025-03-10",
        "tour_time": "14:00",
        "first_name": "John",
        "last_name": "Smith",
        "email": "john@example.com",
        "phone": "555-123-4567"
    }
    """
    
    extraction_result = llm.invoke(
        extraction_prompt + "\n\nConversation:\n" + 
        "\n".join([f"{'User' if isinstance(msg, HumanMessage) else 'Sophie'}: {msg.content}" for msg in messages])
    ).content
    
    # Parse the extraction result
    json_match = re.search(r'{.*}', extraction_result, re.DOTALL)
    if json_match:
        try:
            extracted_data = json.loads(json_match.group(0))
            
            # Update the state with the extracted information
            state.inquiry.tour_scheduled = extracted_data.get("tour_scheduled", False)
            if extracted_data.get("tour_date"):
                state.inquiry.tour_date = extracted_data["tour_date"]
            if extracted_data.get("tour_time"):
                state.inquiry.tour_time = extracted_data["tour_time"]
            if extracted_data.get("first_name"):
                state.user_info.first_name = extracted_data["first_name"]
            if extracted_data.get("last_name"):
                state.user_info.last_name = extracted_data["last_name"]
            if extracted_data.get("email"):
                state.user_info.email = extracted_data["email"]
            if extracted_data.get("phone"):
                state.user_info.phone = extracted_data["phone"]
        except json.JSONDecodeError:
            # If JSON parsing fails, just continue without updating state
            pass
    
    # Increment tour scheduling attempts
    state.conversation_state.tour_scheduling_attempts += 1
    
    # Determine the next node based on the state of the tour scheduling
    if state.inquiry.tour_scheduled and state.user_info.email and state.user_info.phone:
        # All information collected, go to router for any other questions
        state.next_node = "router"
    else:
        # Continue with the tour scheduling
        state.next_node = "tour_scheduler"
    
    return state

def knowledge_base(state: AgentState) -> AgentState:
    """Handles inquiries using the knowledge base."""
    
    # Get the inquiry type from state
    inquiry_type = state.conversation_state.inquiry_type
    
    # Get relevant knowledge from the knowledge base
    knowledge = KNOWLEDGE_BASE.get(inquiry_type, {})
    
    knowledge_base_prompt = """
    You are Sophie, a virtual sales specialist at ACME Senior Living.
    
    Use this information to answer the user's question:
    {knowledge}
    
    If you can answer their question with this information, do so and ask if there's anything else they need.
    
    If you cannot answer their question with this information, apologize and explain that you only have information about pricing, community details, and financing. Ask if they would like to be redirected to a human.
    
    Be conversational and human-like. Respond directly to what they just asked.
    """
    
    # Get the conversation history
    messages = state.messages
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", knowledge_base_prompt.format(knowledge=json.dumps(knowledge, indent=2))),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    # Get response from LLM
    response = llm.invoke(
        prompt.invoke({"messages": messages}).to_messages()
    )
    
    # Update state with the response
    state.messages.append(AIMessage(content=response.content))
    
    # Check if the response suggests redirecting to a human
    redirect_check_prompt = """
    Based on the last AI message, does Sophie suggest redirecting to a human?
    Respond with ONLY "yes" or "no".
    
    Last AI message: {last_ai_message}
    """
    
    redirect_suggested = llm.invoke(
        redirect_check_prompt.format(last_ai_message=response.content)
    ).content.strip().lower()
    
    # Set the next node based on whether a redirect is suggested
    if redirect_suggested == "yes":
        state.next_node = "reattempt_live_contact"
    else:
        state.next_node = "router"
    
    return state

def determine_next_step(state: AgentState) -> str:
    """Determines the next step in the conversation flow based on the state."""
    
    # If the state has a specific next node, use that
    if state.next_node:
        next_node = state.next_node
        state.next_node = None  # Reset for next time
        return next_node
    
    # Default to the router
    return "router"
