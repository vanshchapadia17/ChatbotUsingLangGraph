import os
import re
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing import TypedDict, Annotated

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


# ── Environment ──────────────────────────────────────────────
load_dotenv()


# ── LLM Setup ────────────────────────────────────────────────
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    max_new_tokens=512,
    do_sample=False,
    repetition_penalty=1.03,
    provider="auto",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

chat_model = ChatHuggingFace(llm=llm, verbose=False)


# ── System Prompt ─────────────────────────────────────────────
SYSTEM_MESSAGE = SystemMessage(content=(
    "You are LangGraphBot, a helpful and friendly assistant. "
    "Remember everything the user tells you in the conversation. "
    "Give clear and concise answers."
))


# ── State Definition ──────────────────────────────────────────
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ── Clean DeepSeek <tool_call> tags ───────────────────────────
def clean_response(text: str) -> str:
    return re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL).strip()


# ── Chat Node ────────────────────────────────────────────────
def chat_node(state: ChatState):
    full_messages = [SYSTEM_MESSAGE] + state['messages']
    response = chat_model.invoke(full_messages)
    return {'messages': [response]}


# ── Build Graph ──────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node('chat_node', chat_node)
    graph.add_edge(START, 'chat_node')
    graph.add_edge('chat_node', END)

    return graph.compile(checkpointer=MemorySaver())


# ── GLOBAL Chatbot Instance (IMPORTANT) ───────────────────────
chatbot = build_graph()
config = {'configurable': {'thread_id': '1'}}


# ── CLI Runner (Optional) ─────────────────────────────────────
def main():
    print("🤖 LangGraph Chatbot started! Type 'bye' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("👋 Goodbye!")
            break

        response = chatbot.invoke(
            {'messages': [HumanMessage(content=user_input)]},
            config=config
        )

        raw = response['messages'][-1].content
        print(f"LangGraphBot: {clean_response(raw)}\n")


# ── Run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    main()