import streamlit as st
from backend import llm
from langchain_core.messages import HumanMessage,AIMessage
import uuid

st.set_page_config(
    page_title="AI Chatbot",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.title("🤖 AI Chatbot using LangGraph & Streamlit")

# ── Initialize Session State ─────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = {}

if "current_chat" not in st.session_state:
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

# ── Sidebar ─────────────────────────────────────────────
st.sidebar.title("💬 Chats")

# ➕ New Chat Button
if st.sidebar.button("➕ New Chat"):
    chat_id = str(uuid.uuid4())
    st.session_state.current_chat = chat_id
    st.session_state.chats[chat_id] = []

# 📜 Show Chat List
for chat_id in st.session_state.chats.keys():
    if st.sidebar.button(f"Chat {chat_id[:5]}"):
        st.session_state.current_chat = chat_id

# 🗑️ End Chat Button (your idea)
if st.sidebar.button("❌ End Chat"):
    new_chat_id = str(uuid.uuid4())
    st.session_state.current_chat = new_chat_id
    st.session_state.chats[new_chat_id] = []

# ── Current Chat Messages ─────────────────────────────
messages = st.session_state.chats[st.session_state.current_chat]

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Input ─────────────────────────────────────────────
user_input = st.chat_input("Type your message...")

if user_input:
    # Save user message
    messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔥 IMPORTANT: unique thread_id per chat
    config = {
        "configurable": {
            "thread_id": st.session_state.current_chat
        }
    }
    
    lc_messages = []

    for msg in messages:
        if msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        else:
            lc_messages.append(AIMessage(content=msg["content"]))

    response = llm.invoke(
        lc_messages,
        config=config
    )

    ai_response = response.content

    # Save AI response
    messages.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)