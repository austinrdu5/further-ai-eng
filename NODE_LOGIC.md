# Logic for node traversing in agent in LnagGraph
## List of nodes with conditions for transfer between them

1. `intro`: this node takes in the first query, rephrases the query, and chooses the following:
    - if user is asking for community phone number or an exsiting vendor or resident, provide the community phone number and ask if they have other questions (next node is `intro`)
    - if user is asking about employment, direct to careers page and ask for other questions (next node is `intro`)
    - else, attempt a live contact (function call to `attempt_transfer`), and when failed, tells the user that it's a virtual assistant and asks if it wants to leave info for a callback or continue with the conversation

2. `router`: This is an internal routing node that (usually) has no message to the user. However...
    - if state.conversation_state.is_first_message, respond with disclosure: "Before I answer, just so you know—This conversation is being recorded for quality purposes and you can leave a voicemail at anytime by pressing 0."
    - classify the user's question. 
        - if user wants callback, next node is `reattempt_live_contact` with state.conversation_state.wants_callback = True
        - if user is asking for community phone number or an existing vendor or resident, provide the community phone number and ask if they have other questions (next node is `router`)
        - if user is asking about employment, direct to careers page and ask for other questions (next node is `router`)
        - if user wants callback, next node is `info_collector` with state.conversation_state.wants_callback = True
        - if query is about touring, next node is `tour_scheduler`
        - if query is about floorplan, next node is `info_collector` with state.conversation_state.wants_brochure = True
        - if query is about frustration with AI
            - next node is `reattempt_live_contact` 
        - else, next node is `knowledge_base` with state.conversation_state.inquiry_type set to `pricing`, `community_details`, `financing`, or `uncategorized`

3. `reattempt_live_contact`: 
    - set state.conversation_state.wants_callback = True
    - if current_time - state.conversation_state.time_of_transfer_attempt > 2 minutes, this node calls a function `attempt_transfer` that mocks an attempted transfer to a live person. The user will see "[10 second pause]" as this function runs and upon failure, next node will be `info_collector` with state.conversation_state.wants_callback = True
    - else, skip the function call and proceed straight to `info_collector`

4. `info_collector`: This node attempts to populate state.user_info's first_name, last_name, email, phone, and address. 
    - if complete, ask if there's anything else to help with and send to `router`
    - if incomplete, next node is `info_collector` once again

5. `tour_schedule`: this node attempts to schedule a tour given the community availability and the user's availability
    - this node attempts to schedule no more than 3 times, tracked by state.converstaion_state.tour_scheduling_attempts. 
        - If not exceeded, next node is a retry with further dates, next node is `tour_scheduler`
        - If exceeded, next node is `info_collector` with state.converstation_state.wants_callback = True
    - if tour successfully scheduled, set tour_scheduled, tour_date, and tour_time. Ask if there's anything else to answer, next node is `router`

6. `knowledge_base`: this node answers knowledge-based questions from the user
    - upon entry, use state.conversation_state.inquiry_type (should be set to `pricing`, `community_details`, `financing`, or `uncategorized`) to query KNOWLEDGE_BASE
    - query the LLM with the user's question and injected information
        - if question is answerable with the injected facts, answer the question and ask if there's anything else needed. next node is `router`
        - else, apologize and say that you only have information about `pricing`, `community_details`, `financing` and that anything else needs to be handled by a human. ask if user want to be redirected.
            - Yes: next node is `reattempt_live_contact`
            - No: next node is `router`