# ACME Senior Living AI Assistant

A conversational AI agent designed to handle inquiries and interactions for ACME Senior Living community. The agent, named Sophie, can handle various tasks including answering questions about the community, scheduling tours, collecting user information, and managing callbacks.

## Quick Run
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run agent.py:
```bash
python src/agent.py
```

## Prerequisites
- Python 3.12! (not yet compatible with Python 3.13)
- Required packages (install via pip):
  - langchain-core
  - langchain-openai
  - langgraph
  - pydantic

## Core Functionality
The agent operates as a state machine with several specialized nodes:

- **Intro**: Initial greeting and conversation routing
- **Router**: Categorizes user intents and directs to appropriate handlers
- **Info Collector**: Gathers user information (name, contact details, preferences)
- **Tour Scheduler**: Handles tour scheduling requests
- **Knowledge Base**: Answers questions about community details and policies
- **Reattempt Live Contact**: Manages transfers to live representatives

## Features
- Natural conversation flow with context awareness
- Structured information collection
- Tour scheduling with availability checking
- Knowledge base integration for community information
- Live agent transfer capabilities
- Conversation recording disclosure
- Error handling and retry mechanisms

## Usage
The agent can be run in two modes:

### Interactive Mode
```bash
python src/agent.py
```

### Example Conversation
```bash
python src/agent.py --example
```

### Verbose Mode
For detailed logging and state information:
```bash
python src/agent.py -v
```

## To-dos
- Current interface is just input() and print() with messages stored in history. These can be replaced with functions that connect to messaging API.
- In the knowledge node, the agent currently just dumps the entire (relatively small) knowledge base into the LLM. We can optimize tokens by having the router identify which parts of the knowledge base are actually necessary.
- Implement "attempt transfer to a live agent." Right now, we're just calling a function that waits 3 seconds and returns False.