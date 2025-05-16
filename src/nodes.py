import json
import re
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from state import AgentState
from knowledge import KNOWLEDGE_BASE, CURRENT_DATETIME

# LLM setup
llm = ChatOpenAI(temperature=0.7, model="gpt-4")

def intro(state: AgentState) -> AgentState:
    """First interaction with the user."""
    
    # If intro is already completed, just pass through
    if state.conversation_state.intro_completed:
        state.next_node = "router"
        return state
    
    # Define the system prompt for the initial greeting
    intro_system_prompt = """
    You are Sophie, a virtual sales specialist at ACME Senior Living. 
    
    For the user's first question:
    1. Greet them warmly with: "Hi, this is ACME Senior Living. My name is Sophie. How may I help you today?"
    2. After they ask a question, paraphrase it briefly and say: "Got it! I can definitely help with that. Let me check if my director of sales is available for a conversation. Please hold."
    3. Then say: "Our sales director is not currently available, but I am a virtual assistant, and I am able to answer basic questions about our community. Would you like to speak with me, or leave a message for Jami."
    4. Add: "Before I answer, just so you know—This conversation is being recorded for quality purposes and you can leave a voicemail at anytime by pressing 0."
    5. Finally, answer their question starting with "About your query on [topic]..." and be helpful and friendly.
    
    Be conversational, concise, and human-like. Use everyday language and don't be robotic.
    """
    
    messages = state.messages
    
    # If this is the very first interaction
    if len(messages) == 1 and isinstance(messages[0], HumanMessage):
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", intro_system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        # Get response from LLM
        response = llm.invoke(
            prompt.invoke({"messages": messages}).to_messages()
        )
        
        # Update state
        state.messages.append(AIMessage(content=response.content))
        state.conversation_state.intro_completed = True
        
        # Check for specific inquiry types
        inquiry_check_prompt = """
        Based on the user's message: "{user_message}"
        
        Determine if the user is:
        1. Asking for community phone number or about an existing vendor/resident
        2. Asking about employment
        3. Wanting a callback
        4. None of the above
        
        Respond with ONLY one of: "phone", "employment", "callback", or "other"
        """
        
        inquiry_type = llm.invoke(
            inquiry_check_prompt.format(user_message=messages[0].content)
        ).content.strip().lower()
        
        # Set the next node based on inquiry type
        if inquiry_type == "phone":
            state.next_node = "intro"  # Stay in intro to handle phone number request
        elif inquiry_type == "employment":
            state.next_node = "intro"  # Stay in intro to handle employment request
        elif inquiry_type == "callback":
            state.next_node = "info_collector"
            state.conversation_state.wants_callback = True
        else:
            state.next_node = "router"
    else:
        # If not the first interaction, proceed to router
        state.next_node = "router"
        
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
