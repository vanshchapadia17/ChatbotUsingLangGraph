import streamlit as st
from backend import chatbot
from langchain_core.messages import HumanMessage

st.set_page_config(
    page_title="AI Chatbot",
    layout="wide"
)

st.title("🤖 AI Chatbot using LangGraph & Streamlit")

# ── Single Chat Session ─────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

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
            "thread_id": "single-session"
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

