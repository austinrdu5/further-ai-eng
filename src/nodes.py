import json
import re
import logging
import time
import datetime
from typing import Optional, Literal, Callable, TypeVar, Any, List, Tuple
from datetime import datetime, timedelta

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.exceptions import OutputParserException

from state import AgentState, handle_failed_parsing
from knowledge import KNOWLEDGE_BASE

# Set up logging
logger = logging.getLogger(__name__)

def configure_logging(verbose: bool = False):
    """Configure logging based on verbose mode."""
    # Remove all existing handlers first
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    if verbose:
        logger.setLevel(logging.INFO)
        # Create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    else:
        logger.setLevel(logging.WARNING)

# LLM setup
llm = ChatOpenAI(temperature=0.1, model="gpt-4-turbo")  # Lower temperature for more structured outputs

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
    """
    failed_response = ""
    for attempt in range(max_retries):
        try:
            if attempt == 0:
                response = llm.invoke(messages)
                logger.info(f"LLM Response: {response.content}")
            else:
                retry_messages = [AIMessage(content=failed_response),
                                  HumanMessage("Your JSON structure was invalid, please try again.")]
                response = llm.invoke(messages + retry_messages)
                logger.info(f"LLM Retried Response: {response.content}")
                
            parsed = parser.parse(response.content)
            return parsed
        except OutputParserException as e:
            failed_response = response.content
            continue

    logger.error("Failed to get valid response after all retries")
    return {
        "response": "I apologize, but I'm having trouble processing your request. Let me try again.",
        "failed_parsing": True
    }

def intro(state: AgentState) -> AgentState:
    """First interaction with the user."""
    # Define the system prompt for the initial greeting and routing
    intro_prompt = """
    Instructions:
    1. Continue the above conversation by rephrasing the user's last query to confirm understanding(e.g. "Got it! I can definitely help with that.")
    2. Classify the user's intent and format your response and next_node accordingly:
        - For messages not related to the community:
            - "response": "I'm sorry, I can only help with information about our community. If you have any questions, I'd be happy to answer them!"
            - "next_node": "intro"
        - For other queries related to the community, callbacks, tours, or customer service, always try to transfer to director of sales first:
            - "response": "Let me check if the director of sales is available to [help with whatever the user asked]. Please hold."
            - "next_node": "router"
    
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
        state.conversation_state.time_of_transfer_attempt = datetime.now().isoformat()

        if transfer_success:
            state.next_node = None
            return state
        else:
            transfer_failure_message = "Our sales director is not currently available, but I am a virtual assistant, and I am able to answer basic questions about our community. Would you like to speak with me, or leave a message for Jami?"
            print(f"Sophie: {transfer_failure_message}")
            state.messages += [AIMessage(content=transfer_failure_message)]

    state.accepting_user_input = True
    return state

def router(state: AgentState) -> AgentState:
    """Routes the conversation to the appropriate handler based on the latest message."""
    # Define the system prompt for routing
    routing_prompt = """
    Instructions:
    Using the message history, categorize the user's intent using the structured output:
        - User wants to leave a message or get a callback
            - "category": "callback"
        - User wants to schedule a tour
            - "category": "tour"
        - User wants more information about the floorplans
            - "category": "floorplan"
        - User is frustrated with the AI assistant
            - "category": "frustration"
        - User is asking a question about the community name, phone number, address, amenities, policies, employment, or other community details
            - "category": "knowledge"
        - User is asking a question that is not about the community,
            - "category": "off_topic"

    {format_instructions}
    """

    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="category", description="The category of the user's request (must be either callback, tour, floorplan, frustration, knowledge, or off_topic)"),
    ])
    
    instructions = routing_prompt.format(
        format_instructions=parser.get_format_instructions()
    )
    
    # Get structured response with retries
    parsed_response = structured_invoke(llm, state.messages + [HumanMessage(content=instructions)], parser)
        
    if parsed_response.get("failed_parsing"):
        state = handle_failed_parsing(state, logger)
        return state
    
    # If this is the first message after intro, add disclosure
    if not state.conversation_state.disclosure_given:
        disclosure = "Before I answer, just so you know—This conversation is being recorded for quality purposes and you can leave a voicemail at anytime by pressing 0."
        print(f"Sophie: {disclosure}")
        state.messages.append(AIMessage(content=disclosure))
        state.conversation_state.disclosure_given = True
        time.sleep(1)
        
    # Extract the classification
    category = parsed_response["category"]

    if category == "off_topic":
        off_topic_message = "Sophie: I'm sorry, I can only help with information about our community. If you have any questions, I'd be happy to answer them!"
        print(off_topic_message)
        state.messages += [AIMessage(content=off_topic_message)]
        state.accepting_user_input = True
        state.next_node = "router"

        return state
    
    # If not off topic, set the next node based on the category
    if category == "callback":
        state.next_node = "reattempt_live_contact"
    elif category == "tour":
        state.next_node = "tour_scheduler"
    elif category == "floorplan":
        state.next_node = "info_collector"
        state.conversation_state.wants_brochure = True
    elif category == "frustration":
        state.next_node = "reattempt_live_contact"
    elif category == "knowledge":
        state.next_node = "knowledge_base"
    else:
        logger.error(f"Invalid category: {category}, defaulting to knowledge_base")
        state.next_node = "knowledge_base"

    # Router always directly leads into another node
    state.accepting_user_input = False
    
    return state

def reattempt_live_contact(state: AgentState) -> AgentState:
    """Handles reattempting live contact for frustrated users."""
    # Set callback flag
    state.conversation_state.wants_callback = True

    TWO_MINUTES = 120
    now = datetime.now()
    time_of_last_attempt = state.conversation_state.time_of_transfer_attempt
    
    # If there was a previous attempt, check if enough time has passed
    if time_of_last_attempt is not None:
        # Parse from ISO string
        time_of_last_attempt_dt = datetime.fromisoformat(time_of_last_attempt)
        time_since_last_attempt = now - time_of_last_attempt_dt
        # Early exit to info_collector if not enough time has passed
        if time_since_last_attempt.total_seconds() <= TWO_MINUTES:
            state.next_node = "info_collector"
            return state
    
    # Else, attempt to transfer to live contact
    state.conversation_state.time_of_transfer_attempt = now.isoformat()
    transfer_success = attempt_transfer()

    if transfer_success:
        state.next_node = None
        return state
    
    # If transfer fails, proceed to info_collector
    transfer_failure_message = "Our sales director is not currently available, but I can take a message and have them call you back."
    print(f"Sophie: {transfer_failure_message}")

    state.next_node = "info_collector"
    state.messages += [AIMessage(content=transfer_failure_message)]
    state.accepting_user_input = False

    return state

def info_collector(state: AgentState) -> AgentState:
    """Collects contact information from the user."""
    info_collector_prompt = """Instructions:
    Your task is to continue the conversation while collecting the following information from the user. Output a JSON object containing your response and the collected information.

    Information already collected:
    {present_info}

    Information missing:
    {missing_info}

    Using the conversation history, determine what remaining information you should ask for. 
    Don't be too pushy; only first name, last name, and either email or phone are required. If the user wants a brochure, ask fortheir email to send the brochure.
    Ensure the name, email, and phone number are valid. Don't record invalid information.
    If the user provides any additional information that's useful for a future contact, add it to the extra_information field.

    {format_instructions}

    If you don't have any information, you must still output a valid JSON object: {{"response": your response, "first_name": "", "last_name": "", "email": "", "phone": "", "address": "", "preferred_contact_time": "", "preferred_care_type": "", "resident_relationship": "", "extra_information": ""}}.
    """

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
    
    # Create output parser
    parser = StructuredOutputParser.from_response_schemas([
        ResponseSchema(name="response", description="The full response message to the user that continues the conversation"),
        ResponseSchema(name="first_name", description="User's first name"),
        ResponseSchema(name="last_name", description="User's last name"),
        ResponseSchema(name="email", description="User's email address"),
        ResponseSchema(name="phone", description="User's phone number"),
        ResponseSchema(name="address", description="User's address"),
        ResponseSchema(name="preferred_contact_time", description="User's preferred contact time"),
        ResponseSchema(name="preferred_care_type", description="User's preferred care type (assisted_living or independent_living)"),
        ResponseSchema(name="resident_relationship", description="User's relationship to the resident"),
        ResponseSchema(name="extra_information", description="Additional user preferences/requirements")
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
    print(f"Sophie: {response}")

    # Update state with user info
    state.messages += [AIMessage(content=response)]
    for field, _ in user_fields.items():
        if field == 'extra_information':
            state.user_info.extra_information.update(parsed_response.get(field, {}))
        else:
            setattr(state.user_info, field, parsed_response.get(field, getattr(state.user_info, field)))
    
    # Check if all required information has been collected
    all_collected = (
        state.user_info.first_name and 
        state.user_info.last_name and 
        (state.user_info.email or state.user_info.phone)
    )

    if all_collected:
        logger.info("info_collector: All required information collected, routing to router")
        state.next_node = "router"
    else:
        logger.info("info_collector: Missing required information, staying on info_collector")
        state.next_node = "info_collector"
    
    state.accepting_user_input = True
    return state

def tour_scheduler(state: AgentState) -> AgentState:
    """Handles tour scheduling requests."""
    # Early exit if we've exceeded the maximum attempts
    if state.conversation_state.tour_scheduling_attempts >= 3:
        print("Sophie: I apologize, but I'm having trouble scheduling your tour. Let me see if I can transfer you to a live representative.")
        state.next_node = "reattempt_live_contact"
        return state
    
    # Increment tour scheduling attempts
    state.conversation_state.tour_scheduling_attempts += 1
    
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
        ResponseSchema(name="tour_date", description="The date of the tour, in YYYY-MM-DD format"),
        ResponseSchema(name="tour_time", description="The time of the tour, in 24-hour format (e.g., 14:00)"),
        ResponseSchema(name="first_name", description="User's first name"),
        ResponseSchema(name="last_name", description="User's last name"),
        ResponseSchema(name="email", description="User's email address"),
        ResponseSchema(name="phone", description="User's phone number")
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
    first_name = parsed_response["first_name"]
    last_name = parsed_response["last_name"]
    email = parsed_response["email"]
    phone = parsed_response["phone"]

    print(f"Sophie: {response}")
    state.messages += [AIMessage(content=response)]

    # Update user info if provided by the LLM
    if first_name:
        state.user_info.first_name = first_name
    if last_name:
        state.user_info.last_name = last_name
    if email:
        state.user_info.email = email
    if phone:
        state.user_info.phone = phone

    if not tour_scheduled:
        state.next_node = "tour_scheduler"
    else:
        # Update state with the tour information
        state.conversation_state.tour_date = tour_date
        state.conversation_state.tour_time = tour_time
        state.conversation_state.tour_scheduled = True
        state.next_node = "router"

    state.accepting_user_input = True
    return state

def knowledge_base(state: AgentState) -> AgentState:
    """Handles inquiries using the knowledge base."""
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
    state.accepting_user_input = True
    return state
    
def validator(state: AgentState) -> AgentState:
    """Validates the AI's response."""
    pass
    
if __name__ == "__main__":
    state = AgentState()
    state.messages += [HumanMessage(content="How do I get a brochure?")]
    info_collector(state)