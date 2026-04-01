import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

#---------- utility function ------------

def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state['thread_id'] = generate_thread_id()
    add_thread(st.session_state['thread_id'])
    st.session_state['messages'] = []

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def load_convo(thread_id):
    return chatbot.get_state(config={
        "configurable": {
            "thread_id": thread_id
        }
    }).values['messages']
#----------------------------------------------------------

st.set_page_config(
    page_title="AI Chatbot",
    layout="wide"
)

st.title("🤖 AI Chatbot using LangGraph & Streamlit")

# ── Single Chat Session ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state['chat_threads'] = []

add_thread(st.session_state['thread_id'])

#-- Sidebar integration ------------------------------
st.sidebar.title("langGraph ChatBot")

if st.sidebar.button("New chat"):
    reset_chat()

st.sidebar.header("Chat History")

for thread_id in st.session_state['chat_threads'][::-1]:
    if st.sidebar.button(str(thread_id)):
        st.session_state['thread_id'] = thread_id
        message_history = load_convo(thread_id)

        temp_messages = []

        for msg in message_history:
            if isinstance(message_history, HumanMessage):
                    role= 'user'
            else:
                    role='assistant'
            temp_messages.append({
                "role": role,
                "content": msg.content
            })
        
        st.session_state['messages'] = temp_messages



# ── Display Messages ───────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User Input ─────────────────────────────────────
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔥 LangGraph config (IMPORTANT)
    config = {
        "configurable": {
            "thread_id": st.session_state['thread_id']
        }
    }

    with st.chat_message("assistant"):
        ai_response = st.write_stream(
            message_chunk.content for message_chunk,metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode='messages'
            )
        )

    # Show AI response
    st.session_state.messages.append({
        "role": "assistant",
        "content": ai_response
    })

