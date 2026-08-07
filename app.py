import uuid
from pathlib import Path

from agentic_rag import rebuild_rag_index
from graph import chatbot
from memory import delete_thread, list_threads, save_thread, update_thread_title
from settings import DOCUMENTS_FOLDER, MODEL_NAME, SUPPORTED_IMAGE_EXTENSIONS
import streamlit as st
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from openai import LengthFinishReasonError

st.set_page_config(page_title="Agentic BPMN RAG", page_icon=":speech_balloon:")

st.title("Agentic BPMN RAG")

title_llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0,
    max_completion_tokens=24,
    timeout=(10, 30),
    max_retries=2,
    streaming=False,
)


def new_thread_id():
    return str(uuid.uuid4())


def new_thread():
    return {
        "title": "",
        "title_generated": False,
        "has_user_message": False,
        "pending_interrupt": None,
        "messages": [],
    }


def create_thread():
    thread_id = new_thread_id()
    st.session_state["threads"][thread_id] = new_thread()
    st.session_state["thread_order"].append(thread_id)
    st.session_state["thread_id"] = thread_id


def active_thread():
    return st.session_state["threads"][st.session_state["thread_id"]]


def thread_config(thread_id):
    return {"configurable": {"thread_id": thread_id}}


def langgraph_messages_to_ui(messages):
    ui_messages = []

    for message in messages:
        if isinstance(message, HumanMessage):
            ui_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            if isinstance(message.content, str) and message.content:
                ui_messages.append({"role": "assistant", "content": message.content})

    return ui_messages


def load_thread_messages(thread_id):
    state = chatbot.get_state(config=thread_config(thread_id))
    messages = state.values.get("messages", [])
    return langgraph_messages_to_ui(messages)


def has_user_message(messages):
    return any(message["role"] == "user" for message in messages)


def normalize_threads():
    for thread in st.session_state["threads"].values():
        thread.setdefault("title", "")
        thread["title"] = clean_title(thread["title"]) if thread["title"] else ""
        thread.setdefault("title_generated", bool(thread["title"]))
        thread.setdefault("messages", [])
        thread.setdefault("has_user_message", has_user_message(thread["messages"]))
        thread.setdefault("pending_interrupt", None)


def title_messages(messages_history):
    langchain_messages = []

    for message in messages_history:
        if message["role"] == "user":
            langchain_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_messages.append(AIMessage(content=message["content"]))

    return langchain_messages


def clean_title(title):
    cleaned = title.strip().strip('"').strip("'")
    cleaned = cleaned.replace("**", "").replace("`", "")
    cleaned = cleaned.splitlines()[0] if cleaned else ""
    cleaned = " ".join(cleaned.split())

    if len(cleaned) > 48:
        cleaned = cleaned[:45].rstrip() + "..."

    return cleaned or "Untitled conversation"


def thread_label(thread_id):
    thread = st.session_state["threads"][thread_id]
    return thread["title"] or "Untitled conversation"


def interrupt_value(interrupt):
    return getattr(interrupt, "value", interrupt)


def interrupt_question(interrupt):
    value = interrupt_value(interrupt)

    if isinstance(value, dict):
        return value.get("question") or "Mi serve un chiarimento prima di procedere."

    return str(value)


def response_interrupts(response):
    if isinstance(response, dict):
        return response.get("__interrupt__", ())

    return ()


def stream_assistant_response(user_input, config, resume=False):
    try:
        graph_input = Command(resume=user_input) if resume else {"messages": [HumanMessage(content=user_input)]}
        streamed_content = False

        for stream_mode, chunk in chatbot.stream(
            graph_input,
            config=config,
            stream_mode=["messages", "updates"],
        ):
            if stream_mode == "updates":
                interrupts = response_interrupts(chunk)

                if interrupts:
                    interrupt = interrupts[0]
                    active_thread()["pending_interrupt"] = interrupt_value(interrupt)
                    yield interrupt_question(interrupt)
                    return

                continue

            if stream_mode != "messages":
                continue

            message_chunk, metadata = chunk

            if metadata.get("langgraph_node") != "answer":
                continue

            if not isinstance(message_chunk, (AIMessage, AIMessageChunk)):
                continue

            content = message_chunk.content

            if isinstance(content, str) and content:
                streamed_content = True
                yield content

        active_thread()["pending_interrupt"] = None

        if not streamed_content:
            response = chatbot.get_state(config=thread_config(st.session_state["thread_id"]))
            interrupts = getattr(response, "interrupts", ())

            if interrupts:
                interrupt = interrupts[0]
                active_thread()["pending_interrupt"] = interrupt_value(interrupt)
                yield interrupt_question(interrupt)
    except LengthFinishReasonError:
        graph_input = Command(resume=user_input) if resume else {"messages": [HumanMessage(content=user_input)]}
        response = chatbot.invoke(graph_input, config=config)
        interrupts = response_interrupts(response)

        if interrupts:
            interrupt = interrupts[0]
            active_thread()["pending_interrupt"] = interrupt_value(interrupt)
            yield interrupt_question(interrupt)
            return

        message = response["messages"][-1]

        if isinstance(message, AIMessage) and isinstance(message.content, str):
            active_thread()["pending_interrupt"] = None
            yield message.content
        else:
            yield "La risposta e' stata interrotta per limite di lunghezza. Prova con una richiesta piu' specifica."


def latest_assistant_content(thread_id):
    state = chatbot.get_state(config=thread_config(thread_id))
    messages = state.values.get("messages", [])

    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str):
            if message.content:
                return message.content

    return ""


def generate_thread_title(messages):
    title_prompt = SystemMessage(
        content=(
            "Generate a short title for this conversation. "
            "Use the same language as the user. "
            "Maximum 4 words. Return only the title, with no quotes, no Markdown, no explanation."
        )
    )

    response = title_llm.invoke([title_prompt, *messages[:1]])
    return clean_title(response.content if isinstance(response.content, str) else str(response.content))


def save_uploaded_documents(uploaded_files):
    documents_folder = Path(DOCUMENTS_FOLDER)
    documents_folder.mkdir(parents=True, exist_ok=True)
    saved_files = []

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name).name
        extension = Path(file_name).suffix.lower()

        if extension != ".pdf" and extension not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        destination = documents_folder / file_name
        destination.write_bytes(uploaded_file.getvalue())
        saved_files.append(file_name)

    return saved_files


if "threads" not in st.session_state:
    saved_threads = list_threads()

    if saved_threads:
        st.session_state["threads"] = {}
        st.session_state["thread_order"] = []

        for saved_thread in reversed(saved_threads):
            thread_id = saved_thread["thread_id"]
            title = saved_thread["title"]
            messages = load_thread_messages(thread_id)

            st.session_state["threads"][thread_id] = {
                "title": clean_title(title),
                "title_generated": bool(title),
                "has_user_message": has_user_message(messages),
                "messages": messages,
            }
            st.session_state["thread_order"].append(thread_id)

        st.session_state["thread_id"] = saved_threads[0]["thread_id"]
    else:
        first_thread_id = st.session_state.get("thread_id", new_thread_id())
        previous_messages = st.session_state.get("messages_history", [])
        st.session_state["thread_id"] = first_thread_id
        st.session_state["thread_order"] = [first_thread_id]
        st.session_state["threads"] = {
            first_thread_id: {
                "title": "",
                "title_generated": False,
                "has_user_message": False,
                "messages": previous_messages,
            }
        }

normalize_threads()

if st.sidebar.button("New chat"):
    create_thread()
    st.rerun()

st.sidebar.caption("Recent conversations")

for thread_id in reversed(st.session_state["thread_order"]):
    thread = st.session_state["threads"][thread_id]

    if not thread["has_user_message"]:
        continue

    if st.sidebar.button(thread["title"], key=f"thread_{thread_id}"):
        st.session_state["thread_id"] = thread_id
        st.rerun()

deletable_thread_ids = [
    thread_id
    for thread_id in reversed(st.session_state["thread_order"])
    if st.session_state["threads"][thread_id]["has_user_message"]
]

if deletable_thread_ids:
    st.sidebar.divider()
    selected_delete_thread = st.sidebar.selectbox(
        "Delete conversation",
        options=[""] + deletable_thread_ids,
        format_func=lambda thread_id: "Select a conversation" if not thread_id else thread_label(thread_id),
    )

    if selected_delete_thread and st.sidebar.button("Delete selected chat", type="secondary"):
        delete_thread(selected_delete_thread)
        st.session_state["threads"].pop(selected_delete_thread, None)
        st.session_state["thread_order"] = [
            thread_id
            for thread_id in st.session_state["thread_order"]
            if thread_id != selected_delete_thread
        ]

        if not st.session_state["thread_order"]:
            create_thread()
        elif st.session_state["thread_id"] == selected_delete_thread:
            st.session_state["thread_id"] = st.session_state["thread_order"][-1]

        st.rerun()

current_title = st.sidebar.empty()
if active_thread()["title"]:
    current_title.markdown(f"**{active_thread()['title']}**")

for message in active_thread()["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

pending_interrupt = active_thread().get("pending_interrupt")

if pending_interrupt:
    st.button("In attesa di chiarimento", disabled=True)

chat_input = st.chat_input(
    placeholder="Rispondi al chiarimento..." if pending_interrupt else "Type your message here...",
    accept_file=False if pending_interrupt else "multiple",
    file_type=["pdf", "png", "jpg", "jpeg"],
)

if chat_input:
    if isinstance(chat_input, str):
        user_input = chat_input
        uploaded_files = []
    else:
        user_input = chat_input.text
        uploaded_files = chat_input.files

    thread = active_thread()

    if user_input:
        thread["has_user_message"] = True
        thread["messages"].append({"role": "user", "content": user_input})
        save_thread(st.session_state["thread_id"], thread["title"])

        with st.chat_message("user"):
            st.markdown(user_input)

    is_resume = bool(thread.get("pending_interrupt"))
    uploaded_file_names = [] if is_resume else save_uploaded_documents(uploaded_files)

    if uploaded_file_names:
        with st.spinner("Updating RAG index..."):
            indexed_files = rebuild_rag_index()

        status_message = (
            "Updated RAG index with: "
            + ", ".join(uploaded_file_names)
            + f". Indexed files: {', '.join(indexed_files)}."
        )
        thread["messages"].append({"role": "assistant", "content": status_message})

        with st.chat_message("assistant"):
            st.markdown(status_message)

    if not user_input:
        save_thread(st.session_state["thread_id"], thread["title"])
        st.rerun()

    config = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_trace",
    }

    with st.chat_message("assistant"):
        ai_message = st.write_stream(stream_assistant_response(user_input, config, resume=is_resume))

        if not ai_message:
            ai_message = latest_assistant_content(st.session_state["thread_id"])

            if ai_message:
                st.markdown(ai_message)

    thread["messages"].append({"role": "assistant", "content": ai_message})

    if not thread["title_generated"]:
        thread["title"] = generate_thread_title(title_messages(thread["messages"]))
        thread["title_generated"] = True
        current_title.markdown(f"**{thread['title']}**")
        update_thread_title(st.session_state["thread_id"], thread["title"])
    else:
        save_thread(st.session_state["thread_id"], thread["title"])
