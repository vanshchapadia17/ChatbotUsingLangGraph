import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph,START,END, StateGraph
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage
from typing import TypedDict,Annotated
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",  # or any free model
    task="text-generation",
    max_new_tokens=512,
    temperature=0.5,
    do_sample=False,
    repetition_penalty=1.03,
    provider="novita",
    huggingfacehub_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

chat_model = ChatHuggingFace(llm=llm,verbose=True) 

class ChatState(TypedDict):
    
    messages: Annotated[list[BaseMessage], add_messages] 

def chat_node(state:ChatState):
    #take users query 
    messages=state['messages']

    #generate response using llm
    response = chat_model.invoke(messages)

    #response to store
    return {'messages': [response]}



graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)

chatbot = graph.compile()

while True:
    user_input = input("User: ")

    if user_input.strip() in ['exit', 'quit','bye']:
        print("Exiting chatbot.")
        break

    # Run the chatbot with the user's message
    response = chatbot.invoke({'messages': [HumanMessage(content=user_input)]})

    # Print the chatbot's response
    print('AI:', response['messages'][-1].content)