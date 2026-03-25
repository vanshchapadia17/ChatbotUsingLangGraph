from langgraph.graph import StarGraph,START,END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict,Annotated
from langchain_groq import Groq

llm = Groq()

class ChatState(TypedDict):
    
    messages: Annotated[list[BaseMessage], add_messages] 

def chat_node(state:ChatState):
    #take users query 
    messages=state['messages']

    #generate response using llm
    response = llm.invoke(messages)

    #response to store
    return {'messages': [response]}



graph = StarGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile()