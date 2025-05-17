import json
import re
import logging
import time
import datetime
from typing import Optional, Literal, Callable, TypeVar, Any, List, Tuple
from datetime import datetime, timedelta

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

def attempt_transfer():
    # TODO: implement function 
    print("[10 second pause, attempt transfer to live contact...]")
    time.sleep(3)
    return False

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
        "response": "I apologize, but I'm having trouble processing your request. Let me try again.",
        "next_node": "router",
        "failed_parsing": True
    }

def intro(state: AgentState) -> AgentState:
    """First interaction with the user."""

    # Define the system prompt for the initial greeting and routing
    intro_prompt = """
    Last user message: {user_message}
    
    Instructions:
    1. Rephrase the user's last query to confirm understanding: Got it! I can definitely help with that.
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

    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="response", description="The full response message to the user"),
        ResponseSchema(name="next_node", description="The next node to route to (must be either 'intro' or 'router')")
    ])

    last_message = state.messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message should be a HumanMessage")
    
    # Augment last message 
    new_message = intro_prompt.format(
        user_message=last_message.content,
        format_instructions=parser.get_format_instructions()
    )
    
    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=new_message)], parser)

    # Whether success or fail, parsed_response is a dictionary with keys "response" and "next_node"
    response = parsed_response["response"]
    next_node = parsed_response["next_node"]
    failed_parsing = parsed_response.get("failed_parsing", False)

    # Update state with the response
    state.messages += [AIMessage(content=response)]
    state.next_node = next_node
    if failed_parsing:
        state.failed_parsing = True
        state.n_parsing_fails += 1

    # If Sophie has failed to parse the user's message 3 times in a row, apologize and end the conversation
    if state.n_parsing_fails >= 3:
        print("Sophie: I apologize, but I'm having trouble processing your request. Please call 850-445-8362 for assistance.")
        state.next_node = None
        return state

    # If not failed, execute node's logic
    print(f"Sophie: {response}")            

    if next_node == "router" and not failed_parsing:
        # Attempt to transfer to live contact
        transfer_success = attempt_transfer()
        # update time of transfer attempt
        state.conversation_state.time_of_transfer_attempt = datetime.now()
        if transfer_success:
            # end conversation
            state.next_node = None
            return state
        else:
            transfer_failure_message = "Our sales director is not currently available, but I am a virtual assistant, and I am able to answer basic questions about our community. Would you like to speak with me, or leave a message for Jami?"
            print(f"Sophie: {transfer_failure_message}")
            state.messages += [AIMessage(content=transfer_failure_message)]

    return state

def router(state: AgentState) -> AgentState:
    """Routes the conversation to the appropriate handler based on the latest message."""
    
    # Define the system prompt for routing
    routing_prompt = """
    Last user message: {user_message}
    
    Instructions:
    Using the message history and the above message, categorize the user's intent into one of these categories:
        - callback: Callback request or leaving a message
        - tour: Tour request
        - floorplan: Floorplan request
        - frustration: Frustration with AI
        - knowledge: a general question about the community
            - pricing: cost of care
            - amenities: what activities/services are available (including questions about community details, features, or what the community offers)
            - financing: financing options (Medicaid, VA, etc.)
            - phone: asking for the community phone number
            - employment: Employment inquiry
            - uncategorized: For all other cases

    {format_instructions}
    """

    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="category", description="The category of the user's request (must be either callback, tour, floorplan, frustration, or knowledge)"),
        ResponseSchema(name="inquiry_type", description="The type of inquiry for knowledge category (must be either pricing, amenities, financing, phone, employment, or uncategorized)")
    ])
    
    # Get the last message and validate it
    last_message = state.messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message should be a HumanMessage")
    
    # Format the prompt with the last message
    new_message = routing_prompt.format(
        user_message=last_message.content,
        format_instructions=parser.get_format_instructions()
    )
    
    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=new_message)], parser)
    
    # Extract the classification
    category = parsed_response.get("category", "knowledge")
    inquiry_type = parsed_response.get("inquiry_type", "uncategorized")
    
    # Post-process inquiry type for community details
    if inquiry_type == "uncategorized" and any(term in last_message.content.lower() for term in ["community details", "community features", "what does the community offer", "what's in the community"]):
        inquiry_type = "amenities"
    
    # If this is the first message after intro, add disclosure
    if not state.conversation_state.disclosure_given:
        disclosure = "Before I answer, just so you know—This conversation is being recorded for quality purposes and you can leave a voicemail at anytime by pressing 0."
        print(f"Sophie: {disclosure}")
        state.messages.append(AIMessage(content=disclosure))
        state.conversation_state.disclosure_given = True
    
    # Set the next node based on the category
    if category == "callback":
        state.next_node = "reattempt_live_contact"
    elif category == "tour":
        state.next_node = "tour_scheduler"
    elif category == "floorplan":
        state.next_node = "info_collector"
        state.conversation_state.wants_brochure = True
    elif category == "frustration":
        state.next_node = "reattempt_live_contact"
    else:  # knowledge category
        state.next_node = "knowledge_base"
        state.conversation_state.inquiry_type = inquiry_type
    
    return state

def reattempt_live_contact(state: AgentState) -> AgentState:
    """Handles reattempting live contact for frustrated users."""
    
    # Set callback flag
    state.conversation_state.wants_callback = True
    
    TWO_MINUTES = 120
    now = datetime.now()
    last_attempt = state.conversation_state.time_of_transfer_attempt
    if last_attempt is None:
        last_attempt = now - timedelta(seconds=TWO_MINUTES+1)  # force as if enough time has passed
    time_since_last_attempt = now - last_attempt
    
    # Early exit to info_collector if not enough time has passed
    if time_since_last_attempt.total_seconds() <= TWO_MINUTES:
        state.next_node = "info_collector"
        return state
    
    # Else, attempt to transfer to live contact
    transfer_success = attempt_transfer()
    state.conversation_state.time_of_transfer_attempt = now

    if transfer_success:
        # end conversation
        state.next_node = None
        return state
    else:
        transfer_failure_message = "Our sales director is not currently available, but I can take a message and have them call you back."
        print(f"Sophie: {transfer_failure_message}")
        state.messages += [AIMessage(content=transfer_failure_message)]
    
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
        state.next_node = "reattempt_live_contact"
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
