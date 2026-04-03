import os
import re
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq

import sqlite3
#----------------- database -------------

conn = sqlite3.connect(database='chatbot_history.db', check_same_thread=False)


# ── Load Environment ─────────────────────────────
load_dotenv()

# ── LLM Setup ───────────────────────────────────
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=512,
    groq_api_key=os.getenv("GROQ_API_KEY")
)

# ── System Prompt ───────────────────────────────
SYSTEM_MESSAGE = SystemMessage(content=(
    "You are LangGraphBot, a helpful and friendly assistant. "
    "Remember everything the user tells you in the conversation. "
    "Give clear and concise answers."
))

# ── State Definition ────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# ── Clean Groq weird tags ───────────────────────
def clean_response(text: str) -> str:
    return re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()

# ── Chat Node ──────────────────────────────────
def chat_node(state: ChatState):
    full_messages = [SYSTEM_MESSAGE] + state['messages']
    
    response = llm.invoke(full_messages)

    # ✅ IMPORTANT: append response (not replace)
    return {
        'messages': state['messages'] + [response]
    }
checkpointer = SqliteSaver(conn=conn)

# ── Build Graph ────────────────────────────────
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node('chat_node', chat_node)
    graph.add_edge(START, 'chat_node')
    graph.add_edge('chat_node', END)

    return graph.compile(checkpointer=checkpointer)

# ── Export Chatbot (used in Streamlit) ─────────
chatbot = build_graph()

def retrieve_thread():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])

    return list(all_threads)