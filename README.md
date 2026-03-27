"# LangGraph Chatbot with Persistent Memory

A conversational AI chatbot built using **LangGraph** , powered by the DeepSeek-R1 model from HuggingFace. This chatbot maintains conversation history and provides intelligent responses with support for persistent memory across sessions.

## Features

- 🤖 **AI-Powered Conversations** - Powered by DeepSeek-R1 language model via HuggingFace
- 💾 **Persistent Memory** - Conversation history is maintained and remembered within sessions using LangGraph's MemorySaver
- 🔄 **State Management** - Built with LangGraph's StateGraph for robust conversation flow
- 🧹 **Clean Output** - Automatically removes inference artifacts (e.g., `<think>` tags) from responses
- 📚 **Full LangChain Integration** - Leverages LangChain and LangChain-Community libraries
- 🔐 **Environment-Based Configuration** - Uses `.env` for secure API token management

## Project Structure

```
ChatbotLangGraph/
├── chatbot.py                 # Basic LangGraph setup and imports
├── usingpersistant.py        # Main chatbot implementation with persistent memory
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup configuration
├── README.md                 # This file
└── CropAI.egg-info/         # Package metadata
```

## Prerequisites

- Python 3.8+
- HuggingFace API Token (free account at [huggingface.co](https://huggingface.co))
- Internet connection for LLM API calls

## Installation

1. **Clone or download the repository**
   ```bash
   cd ChatbotLangGraph
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a `.env` file in the project root
   ```env
   HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_token_here
   ```

## Dependencies

- **langgraph** - Graph-based orchestration for LLM applications
- **langchain** & **langchain-core** - Core LLM framework
- **langchain-community** - Community integrations
- **langchain-groq** - Groq LLM integration support
- **langchain-huggingface** - HuggingFace model integration
- **huggingface_hub** - HuggingFace Hub Python client
- **python-dotenv** - Environment variable management

## Usage

### Run the Chatbot

Execute the main chatbot with persistent memory:
```bash
python usingpersistant.py
```

### Interactive Chat

```
LangGraph Chatbot started! Type 'bye' to exit.

You: Hello! What can you do?
LangGraphBot: I'm a helpful assistant built with LangGraph. I can help you with a wide range of topics...

You: Tell me about AI
LangGraphBot: Artificial Intelligence is a field of computer science...

You: bye
👋 Goodbye!
```

### Commands

- **Start chatting** - Type any message to interact with the bot
- **Exit** - Type `exit`, `quit`, or `bye` to end the session
- **Skip empty input** - Empty lines are automatically ignored

## How It Works

### Architecture

1. **State Management** - Uses `ChatState` TypedDict to manage conversation messages
2. **LLM Integration** - Connects to HuggingFace's DeepSeek-R1 model
3. **Graph-Based Flow** - Single-node graph with START → chat_node → END flow
4. **Memory Persistence** - MemorySaver checkpoint ensures conversation history is retained
5. **Message Processing** - Uses `add_messages` to append new messages to conversation history
6. **Response Cleaning** - Removes inference artifacts from raw model output

### Key Components

- **System Prompt** - Defines bot behavior as a helpful assistant
- **Chat Node** - Processes user messages with full conversation context
- **Thread ID** - Session identifier for conversation tracking
- **Model** - DeepSeek-R1 (512 token context window, deterministic output)

## Configuration

### Modify LLM Parameters

Edit [usingpersistant.py](usingpersistant.py#L9-L16) to adjust:

```python
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",      # Model to use
    task="text-generation",
    max_new_tokens=512,                      # Response length
    do_sample=False,                         # Deterministic output
    repetition_penalty=1.03,                 # Penalize repetition
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
```

### Change System Prompt

Modify the system message in [usingpersistant.py](usingpersistant.py#L24-L28) to customize bot behavior:

```python
SYSTEM_MESSAGE = SystemMessage(content=(
    "Your custom instructions here..."
))
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `HUGGINGFACEHUB_API_TOKEN` not found | Ensure `.env` file exists with valid token |
| Response contains `<think>` tags | Already handled by `clean_response()` function |
| Empty responses | Check API token validity and rate limits |
| Connection timeout | Verify internet connection and HuggingFace API availability |

## Future Enhancements

- 🔀 Multi-turn document QA with retrieval augmentation
- 📁 Database persistence (SQLite/PostgreSQL)
- 🎯 Fine-tuning support for custom models
- 🌐 Web UI (Streamlit/Gradio integration)
- 🔍 Semantic search for context retrieval
- 📊 Analytics and conversation logging

## Author

**Vanshchapadia** ([chapadiav01@gmail.com](mailto:chapadiav01@gmail.com))

## License

This project is part of the CropAI package. Check the included license for details.

## References

- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [HuggingFace Models](https://huggingface.co/models)
- [DeepSeek-R1 Model](https://huggingface.co/deepseek-ai/DeepSeek-R1)" 
