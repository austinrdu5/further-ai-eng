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

from state import AgentState, handle_failed_parsing
from knowledge import KNOWLEDGE_BASE

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
        "failed_parsing": True
    }

def intro(state: AgentState) -> AgentState:
    """First interaction with the user."""
    logger.info("Starting intro node")
    
    # Define the system prompt for the initial greeting and routing
    intro_prompt = """
    Instructions:
    1. Continue the above conversation by rephrasing the user's last query to confirm understanding(e.g. "Got it! I can definitely help with that.")
    2. Analyze the user's question for the following cases:
        - If asking for community phone number or about existing vendor/resident:
            - Community's number is (850)-445-8362 and ask if they have other questions
            - Set next_node to "intro"
        - If asking about employment:
            - Direct user to careers page: https://www.talkfurther.com/events-demo and ask if they have other questions
            - Set next_node to "intro"
        - For questions not related to the community:
            - "I'm sorry, I can only help with information about our community. If you have any questions, I'd be happy to answer them!"
            - Set next_node to "intro"
        - For queries related to the community:
            - "Let me check if my director of sales is available to answer your question. Please hold."
            - Set next_node to "router"
    
    {format_instructions}
    """

    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="response", description="The full response message to the user"),
        ResponseSchema(name="next_node", description="The next node to route to (must be either 'intro' or 'router')")
    ])

    instructions = intro_prompt.format(format_instructions=parser.get_format_instructions())
    
    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=instructions)], parser)

    # If failed parsing, handle it
    if parsed_response.get("failed_parsing"):
        state = handle_failed_parsing(state, logger)
        return state

    # If success, execute node's logic
    response = parsed_response["response"]
    next_node = parsed_response.get("next_node")

    print(f"Sophie: {response}")   

    state.messages.append(AIMessage(content=response))
    state.next_node = next_node

    if next_node == "router":
        # First try to transfer to live contact
        transfer_success = attempt_transfer()
        logger.info("Attempting transfer to live contact")
        state.conversation_state.time_of_transfer_attempt = datetime.now()

        if transfer_success:
            logger.info("Successfully transferred to live contact")
            # End conversation
            state.next_node = None
            return state
        else:
            logger.info("Transfer to live contact failed, continuing with virtual assistant")
            transfer_failure_message = "Our sales director is not currently available, but I am a virtual assistant, and I am able to answer basic questions about our community. Would you like to speak with me, or leave a message for Jami?"
            print(f"Sophie: {transfer_failure_message}")
            state.messages += [AIMessage(content=transfer_failure_message)]

    logger.info(f"Intro node complete, transitioning to {state.next_node}")
    return state

def router(state: AgentState) -> AgentState:
    """Routes the conversation to the appropriate handler based on the latest message."""
    logger.info("Starting router node")
    
    # Define the system prompt for routing
    routing_prompt = """
    Instructions:
    Using the message history and the above message, categorize the user's intent into one of these categories:
        - callback: Callback request or leaving a message
        - tour: Tour request
        - floorplan: Floorplan request
        - frustration: Frustration with AI
        - knowledge: a general question about the community
            - community_info: about community name, phone number, address, smoking policy, care types, room types, capacity, minimum age, entrance fee, cost, price, tour hours
            - amenities: about activities, services, features, rooms, dining, fitness, medical services, etc.
            - policies: about pets, cars, couples, wheelchairs, visiting hours, security, lease term, languages, payment options, etc.
            - employment: about jobs at the community
        - off_topic: not about the community

    {format_instructions}
    """

    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="category", description="The category of the user's request (must be either callback, tour, floorplan, frustration, knowledge, or off_topic)"),
        ResponseSchema(name="inquiry_types", description="The type(s) of inquiry for knowledge category (can be one or more of: community_info, amenities, policies, or employment)", required=False)
    ])
    
    # Get the last message and validate it
    last_message = state.messages[-1]
    if not isinstance(last_message, HumanMessage):
        raise ValueError("Last message should be a HumanMessage")
    
    # Format the prompt with the last message
    instructions = routing_prompt.format(
        format_instructions=parser.get_format_instructions()
    )
    
    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=instructions)], parser)
        
    if parsed_response.get("failed_parsing"):
        state = handle_failed_parsing(state, logger)
        return state
        
    # Extract the classification
    category = parsed_response.get("category", "knowledge")
    logger.info(f"Router classified user intent as: {category}")
    
    # If this is the first message after intro, add disclosure
    if not state.conversation_state.disclosure_given:
        logger.info("First message after intro, adding disclosure")
        disclosure = "Before I answer, just so you know—This conversation is being recorded for quality purposes and you can leave a voicemail at anytime by pressing 0."
        print(f"Sophie: {disclosure}")
        state.messages.append(AIMessage(content=disclosure))
        state.conversation_state.disclosure_given = True
    
    # Set the next node based on the category
    if category == "callback":
        logger.info("Routing to reattempt_live_contact for callback request")
        state.next_node = "reattempt_live_contact"

    elif category == "tour":
        logger.info("Routing to tour_scheduler for tour request")
        state.next_node = "tour_scheduler"

    elif category == "floorplan":
        logger.info("Routing to info_collector for floorplan request")
        state.next_node = "info_collector"
        state.conversation_state.wants_brochure = True

    elif category == "frustration":
        logger.info("Routing to reattempt_live_contact for frustrated user")
        state.next_node = "reattempt_live_contact"

    elif category == "knowledge":
        logger.info("Routing to knowledge_base for general inquiry")
        state.next_node = "knowledge_base"

        # TODO: do we need to trim down to only the relevant knowledge?
        # state.conversation_state.inquiry_types = []
        # inquiry_types = parsed_response.get("inquiry_types", [])
        # logger.info(f"Knowledge inquiry types: {inquiry_types}")
        # for inquiry_type in ["community_info", "amenities", "policies", "employment", "pricing", "financing", "uncategorized", "phone"]:
        #     if inquiry_type in inquiry_types:
        #         state.conversation_state.inquiry_types.append(inquiry_type)

    elif category == "off_topic":
        off_topic_message = "Sophie: I'm sorry, I can only help with information about our community. If you have any questions, I'd be happy to answer them!"
        print(off_topic_message)
        state.messages += [AIMessage(content=off_topic_message)]
        state.next_node = "router"

    else:
        logger.error(f"Invalid category: {category}, defaulting to router")
        state.next_node = "router"
    
    logger.info(f"Router node complete, transitioning to {state.next_node}")
    return state

def reattempt_live_contact(state: AgentState) -> AgentState:
    """Handles reattempting live contact for frustrated users."""
    logger.info("Starting reattempt_live_contact node")
    
    # Set callback flag
    state.conversation_state.wants_callback = True

    TWO_MINUTES = 120
    now = datetime.now()
    time_of_last_attempt = state.conversation_state.time_of_transfer_attempt
    time_since_last_attempt = now - time_of_last_attempt
    
    # Early exit to info_collector if not enough time has passed
    if time_of_last_attempt is not None and time_since_last_attempt.total_seconds() <= TWO_MINUTES:
        logger.info("Not enough time since last transfer attempt, routing to info_collector")
        state.next_node = "info_collector"
        return state
    
    # Else, attempt to transfer to live contact
    logger.info("Attempting transfer to live contact")
    state.conversation_state.time_of_transfer_attempt = now
    transfer_success = attempt_transfer()

    if transfer_success:
        logger.info("Successfully transferred to live contact")
        # End conversation
        state.next_node = None
        return state
    
    # If transfer fails, proceed to info_collector
    logger.info("Transfer to live contact failed, proceeding to info_collector")

    transfer_failure_message = "Our sales director is not currently available, but I can take a message and have them call you back."
    print(f"Sophie: {transfer_failure_message}")

    state.next_node = "info_collector"
    state.messages += [AIMessage(content=transfer_failure_message)]

    return state

def info_collector(state: AgentState) -> AgentState:
    """Collects contact information from the user."""
    logger.info("Starting info_collector node")
    
    user_fields = {
        'first_name': 'first name',
        'last_name': 'last name',
        'email': 'email address',
        'phone': 'phone number',
        'address': 'address',
        'preferred_contact_time': 'preferred contact time',
        'preferred_care_type': 'preferred care type',
        'resident_relationship': 'relationship to the resident',
        'extra_information': 'any additional information'
    }
    
    # Get missing and present info fields from state
    present_info = [
        f"{display_name}: {getattr(state.user_info, field)}"
        for field, display_name in user_fields.items() 
        if getattr(state.user_info, field)
    ]
    
    missing_info = [
        display_name 
        for field, display_name in user_fields.items() 
        if not getattr(state.user_info, field)
    ]
    
    info_collector_prompt = """
    Instructions:
    Your task is to collect and organize the following information from the user:
    
    Information already collected:
    {present_info}
    
    Information missing:
    {missing_info}
    
    Using the conversation history, determine what remaining information you should ask for. 
    Don't be too pushy; only first name, last name, and either (email or phone) are required.
    If the user provides any additional information that's useful for a future contact, add it to the extra_information field.
    
    {format_instructions}
    """
    
    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="response", description="The full response message to the user"),
        ResponseSchema(name="user_info", description="Updated user information", type="object", properties={
            "first_name": {"type": "string", "description": "User's first name"},
            "last_name": {"type": "string", "description": "User's last name"},
            "email": {"type": "string", "description": "User's email address"},
            "phone": {"type": "string", "description": "User's phone number"},
            "address": {"type": "string", "description": "User's address"},
            "preferred_contact_time": {"type": "string", "description": "User's preferred contact time"},
            "preferred_care_type": {"type": "string", "description": "User's preferred care type (assisted_living or independent_living)"},
            "resident_relationship": {"type": "string", "description": "User's relationship to the resident"},
            "extra_information": {"type": "object", "description": "Additional user preferences/requirements"}
        })
    ])
    
    instructions = info_collector_prompt.format(
        present_info="\n".join(present_info) if present_info else "None",
        missing_info="\n".join(missing_info) if missing_info else "None",
        format_instructions=parser.get_format_instructions()
    )

    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=instructions)], parser)

    # If failed parsing, handle it
    if parsed_response.get("failed_parsing"):
        state = handle_failed_parsing(state, logger)
        return state

    # If success, execute node's logic
    response = parsed_response["response"]
    user_info = parsed_response["user_info"]

    print(f"Sophie: {response}")     
    state.messages += [AIMessage(content=response)]

    # Update state with user info
    for field, _ in user_fields.items():
        if field == 'extra_information':
            state.user_info.extra_information.update(user_info.get(field, {}))
        else:
            setattr(state.user_info, field, user_info.get(field, getattr(state.user_info, field)))
    
    # Check if all required information has been collected
    all_collected = (
        state.user_info.first_name and 
        state.user_info.last_name and 
        (state.user_info.email or state.user_info.phone)
    )

    if all_collected:
        logger.info("All required information collected, routing to router")
        state.next_node = "router"
    else:
        logger.info("Missing required information, staying in info_collector")
        state.next_node = "info_collector"
    
    logger.info(f"Info_collector node complete, transitioning to {state.next_node}")
    return state

def tour_scheduler(state: AgentState) -> AgentState:
    """Handles tour scheduling requests."""
    logger.info("Starting tour_scheduler node")
    
    # Early exit if we've exceeded the maximum attempts
    if state.conversation_state.tour_scheduling_attempts >= 3:
        logger.warning("Exceeded maximum tour scheduling attempts, routing to reattempt_live_contact")
        print("Sophie: I apologize, but I'm having trouble scheduling your tour. Let me see if I can transfer you to a live representative.")
        state.next_node = "reattempt_live_contact"
        return state
    
    # Increment tour scheduling attempts
    state.conversation_state.tour_scheduling_attempts += 1
    logger.info(f"Tour scheduling attempt {state.conversation_state.tour_scheduling_attempts}")
    
    tour_scheduler_prompt = """
    Instructions:    
    Your task is to schedule a tour for the potential resident. The user's previous messages and your conversation history will help you determine where they are in this process. Follow these steps:
    
    1. If you don't know their availability yet, ask when they would like to schedule the tour.
    2. If they propose a time outside your available hours, politely inform them of your available times and ask for an alternative.
    3. Once a valid time is agreed upon, confirm the tour date and time.
    
    Additionally, extract the user's information from the conversation history if available:
    - If the user mentions their name (e.g., "my name is John Smith"), extract first_name and last_name.
    - If the user provides an email address, extract it.
    - If the user provides a phone number, extract it.
    
    Current date and time: {current_datetime}
    
    Available tour times:
    - Monday to Friday
    - Between 9:00 AM and 6:00 PM
    
    Valid time formats include:
    - "3pm" or "3 PM" or "3:00 PM"
    - "2pm" or "2 PM" or "2:00 PM"
    - "10am" or "10 AM" or "10:00 AM"
    - "5pm" or "5 PM" or "5:00 PM"
    
    Examples of valid tour requests:
    - "Next Monday at 3pm"
    - "Tomorrow at 2:00 PM"
    - "Friday at 10am"
    
    Be conversational and human-like. Respond directly to what they just said and store response and tour information in your structured output.

    {format_instructions}
    """

    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="response", description="The full response message to the user"),
        ResponseSchema(name="tour_scheduled", description="Whether the tour has been successfully scheduled (true/false)"),
        ResponseSchema(name="tour_date", description="The date of the tour"),
        ResponseSchema(name="tour_time", description="The time of the tour"),
        ResponseSchema(name="user_info", description="Extracted user information", type="object", properties={
            "first_name": {"type": "string", "description": "User's first name"},
            "last_name": {"type": "string", "description": "User's last name"},
            "email": {"type": "string", "description": "User's email address"},
            "phone": {"type": "string", "description": "User's phone number"}
        })
    ])
    
    instructions = tour_scheduler_prompt.format(
        current_datetime=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        format_instructions=parser.get_format_instructions()
    )
    
    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=instructions)], parser)

    # If failed parsing, handle it
    if parsed_response.get("failed_parsing"):
        state = handle_failed_parsing(state, logger)
        return state
    
    # If success, execute node's logic
    response = parsed_response["response"]
    tour_scheduled = parsed_response["tour_scheduled"].lower() == 'true'  # Convert string to boolean
    tour_date = parsed_response["tour_date"]
    tour_time = parsed_response["tour_time"]
    user_info = parsed_response.get("user_info", {})

    print(f"Sophie: {response}")
    state.messages += [AIMessage(content=response)]

    # Update user info if provided by the LLM
    if user_info:
        if user_info.get("first_name"):
            state.user_info.first_name = user_info["first_name"]
        if user_info.get("last_name"):
            state.user_info.last_name = user_info["last_name"]
        if user_info.get("email"):
            state.user_info.email = user_info["email"]
        if user_info.get("phone"):
            state.user_info.phone = user_info["phone"]

    if not tour_scheduled:
        logger.info("Tour not yet scheduled, staying in tour_scheduler")
        state.next_node = "tour_scheduler"
    else:
        logger.info(f"Tour scheduled for {tour_date} at {tour_time}")
        # Update state with the tour information
        state.conversation_state.tour_date = tour_date
        state.conversation_state.tour_time = tour_time
        state.conversation_state.tour_scheduled = True
        state.next_node = "router"

    logger.info(f"Tour_scheduler node complete, transitioning to {state.next_node}")
    return state

def knowledge_base(state: AgentState) -> AgentState:
    """Handles inquiries using the knowledge base."""
    logger.info("Starting knowledge_base node")
    
    knowledge_prompt = """
    Instructions:
    Use this information to answer the user's question:
    {knowledge}
    
    If you can answer their question with this information:
    1. Provide a clear, direct answer
    2. Include relevant details from the knowledge base
    3. End with "Is there anything else you would like to know?"
    
    If you cannot answer their question with this information:
    1. Apologize and explain that you only have information about community details, amenities, and community policies
    2. Ask if they would like to be redirected to a human
        
    Be conversational and human-like. Respond directly to what they just asked.

    {format_instructions}
    """
    
    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="response", description="The full response message to the user"),
    ])
    
    instructions = knowledge_prompt.format(
        knowledge=json.dumps(KNOWLEDGE_BASE, indent=2), 
        format_instructions=parser.get_format_instructions()
    )

    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=instructions)], parser)

    if parsed_response.get("failed_parsing"):
        state = handle_failed_parsing(state, logger)
        return state

    # If success, execute node's logic
    response = parsed_response["response"]
    
    print(f"Sophie: {response}")
    state.messages += [AIMessage(content=response)]

    state.next_node = "router"
    logger.info("Knowledge_base node complete, transitioning to router")

    return state
    
def validator(state: AgentState) -> AgentState:
    """Validates the AI's response."""
    pass
    
