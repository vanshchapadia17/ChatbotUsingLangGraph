# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

The repo uses a local venv at `venv/` and expects a `.env` file at the project root with:

- `HUGGINGFACEHUB_API_TOKEN` — required by `chatbot.py` and `usingpersistant.py` (DeepSeek-R1 via HuggingFace Inference Endpoint)
- `GROQ_API_KEY` — required by both Streamlit variants (`streamlit_UI/`, `SQlite/`) which use `ChatGroq` with `llama-3.1-8b-instant`

Install once with:

```bash
pip install -r requirements.txt
```

`requirements.txt` ends with `-e .`, so this also installs the project itself as an editable package per `setup.py` (package name `LangraphChatbot`).

## Running the four entry points

There are four independent runnable variants. Each uses the same LangGraph pattern (see "Architecture") but differs in UI, LLM provider, and persistence backend. **Working directory matters** — the Streamlit variants use sibling imports and relative DB paths.

| Entry point | Command | LLM | Persistence |
|---|---|---|---|
| `chatbot.py` | `python chatbot.py` | HF DeepSeek-R1 | none |
| `usingpersistant.py` | `python usingpersistant.py` | HF DeepSeek-R1 | in-memory (`MemorySaver`), thread_id `"1"` |
| `streamlit_UI/frontend.py` | `cd streamlit_UI && streamlit run frontend.py` | Groq Llama 3.1 | in-memory (`MemorySaver`), threads lost on restart |
| `SQlite/SQ_frontend.py` | `cd SQlite && streamlit run SQ_frontend.py` | Groq Llama 3.1 | SQLite (`chatbot_history.db`) |

Notes:
- Both Streamlit frontends do `from backend import chatbot` / `from SQ_backend import chatbot` — they MUST be launched from inside their own directory, otherwise the import fails.
- `SQ_backend.py` opens SQLite with the relative path `chatbot_history.db`, so launching from elsewhere also creates/uses the wrong DB file.
- There are no tests, linter, or build step configured.

## Architecture

Every variant implements the same minimal LangGraph shape:

1. `ChatState = TypedDict` with `messages: Annotated[list[BaseMessage], add_messages]` — `add_messages` is what appends new turns to history instead of overwriting.
2. A single node `chat_node(state)` that prepends a `SystemMessage` to `state['messages']` and calls `llm.invoke(...)`.
3. `StateGraph(ChatState)` wired `START → chat_node → END`, compiled with an optional `checkpointer`.
4. Invocation uses `config={'configurable': {'thread_id': <id>}}` — the checkpointer keys conversation history by `thread_id`.

Variant-specific details worth knowing before editing:

- **`chat_node` return contract differs between files.** `usingpersistant.py` returns `{'messages': [response]}` (relies on `add_messages` to append), while `streamlit_UI/backend.py` and `SQlite/SQ_backend.py` return `{'messages': state['messages'] + [response]}` (manually concatenates). Both work with `add_messages`, but don't mix the two styles in one file without thinking about it.
- **DeepSeek-R1 emits `<think>` / `<tool_call>` artifacts.** `clean_response()` regex-strips these. The regex in `usingpersistant.py` is actually buggy (`<tool_call>.*?<tool_call>` instead of `</tool_call>`) — fix only if the task calls for it.
- **SQLite checkpointer thread listing.** `SQ_backend.retrieve_thread()` iterates `checkpointer.list(None)` and dedupes thread IDs to populate the sidebar — this is how the SQLite variant restores past conversations across restarts.
- **Streaming in the UI.** Frontends use `chatbot.stream(..., stream_mode='messages')` and unpack `(message_chunk, metadata)` tuples — don't switch `stream_mode` without updating the unpacking.
- **Logging/exception infra (`logger.py`, `exception.py`) is only wired into the top-level CLI scripts** (`chatbot.py`, `usingpersistant.py`), not the Streamlit variants. `logger.py` creates `logs/<timestamp>.log/<timestamp>.log` — note it calls `os.makedirs(logs_path, ...)` where `logs_path` is the full file path, so the "log file" is actually a directory with a file of the same name inside it.

## Conventions

- `.env` is loaded via `python-dotenv` at the top of every backend module; do not hardcode keys.
- The system prompt is defined inline as a module-level `SYSTEM_MESSAGE = SystemMessage(...)` constant in each backend — change it there, not at call sites.
- Thread IDs in the Streamlit variants are `uuid.uuid4()` objects (not strings) stored in `st.session_state['chat_threads']`; they're passed directly into the LangGraph `configurable` dict.
