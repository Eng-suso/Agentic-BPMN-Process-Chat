import uuid
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from openai import LengthFinishReasonError
from pydantic import BaseModel, Field

from agentic_rag import indexed_document_names, tool_node
from memory import connection, list_threads, save_thread, setup_thread_metadata, update_thread_title
from settings import ANSWER_MAX_CONTINUATIONS, FINAL_MAX_COMPLETION_TOKENS, MODEL_NAME

llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0.25,
    max_completion_tokens=FINAL_MAX_COMPLETION_TOKENS,
    timeout=(10, 60),
    max_retries=3,
    stream_usage=True,
)

router_llm = ChatOpenAI(
    model_name=MODEL_NAME,
    temperature=0,
    max_completion_tokens=1200,
    timeout=(10, 30),
    max_retries=2,
    streaming=False,
    disable_streaming=True,
    stream_usage=False,
)


class RouteDecision(BaseModel):
    """Structured routing decision for one user turn."""

    route: Literal["rag", "web", "direct", "clarify"] = Field(
        description="The next action: use local RAG, use web search, answer directly, or ask a clarification."
    )
    query: str = Field(
        default="",
        description="The rewritten query to send to the selected tool, or the original user request."
    )
    reason: str = Field(default="", description="Very short reason.")


class ChatState(TypedDict, total=False):
    """Conversation state managed by LangGraph."""

    messages: Annotated[list[BaseMessage], add_messages]
    route: str
    route_query: str
    route_reason: str


SYSTEM_PROMPT = """You are an expert BPMN 2.0 process modeling assistant.

The user has provided BPMN/process resources as local documents.
Use retrieved document context when the answer depends on uploaded/local documents,
BPMN examples, process diagrams, process mapping context, or document-specific evidence.

Your job is to help the user map business processes into BPMN-compliant models.

Core rules:
- Use retrieved document context as evidence when available.
- Do not claim something comes from local documents unless it is in retrieved context.
- Use web-search context only for current/external facts.
- Distinguish clearly between business-process discovery and BPMN notation rules.
- Ask follow-up questions when the business process is missing key details.
- Prefer BPMN 2.0 terminology: start event, end event, task, subprocess, exclusive gateway, parallel gateway, event-based gateway, sequence flow, message flow, pool, lane, participant, data object, data store.
- When mapping a business process, produce both a business-readable structure and a BPMN-oriented structure.
- Include a Mermaid diagram only when the user asks to map/model/visualize a process, asks for a graph/diagram, or when a visual flow is necessary to make the answer useful.
- Do not include Mermaid for simple explanations, definitions, troubleshooting, factual answers, or short conceptual answers unless the user explicitly asks for a diagram.
- When you do include a process graph or diagram, use a fenced Mermaid code block. Do not represent the diagram only as a Markdown bullet list or table.
- Use `flowchart TD` for high-level process flows unless a sequence-style interaction is more appropriate.
- Keep Mermaid node labels concise and quote labels that contain spaces or punctuation.
- Finish every answer completely. If the answer would be long, be concise and complete instead of ending mid-sentence.
- Do not end with an unfinished section, unfinished Mermaid block, or dangling sentence.

For process mapping, identify:
- Process name
- Goal
- Trigger / start event
- Participants, pools, lanes, roles
- Tasks and subprocesses
- Decision points and gateway types
- Events
- Sequence flows
- Message flows
- Inputs, outputs, data objects
- Exceptions and alternative paths
- End events
- Missing information

When mapping a process, output the sections that are useful:
1. Business process summary
2. Mermaid process diagram, only if needed
3. BPMN elements to use
4. BPMN flow
5. Open questions
6. Modeling risks or ambiguities
"""

ROUTER_PROMPT = """Route the latest user request.

Return one route:
- rag: local/uploaded documents, BPMN examples, diagrams, process mapping evidence.
- web: current/recent/external facts, news, regulations, products, companies.
- direct: simple stable general question.
- clarify: process mapping request is too vague and cannot be grounded in available local resources or prior conversation.

If the user refers to an uploaded file, local document, diagram, image, example, or previous object using pronouns like "it/this/lo/quello", route to rag when local resources or prior context are available.
If local resources are indexed and the user is asking to map, understand, improve, or explain a business process, route to rag before asking clarifying questions.
Use clarify only when retrieval would not add useful context.
Keep query empty unless a short rewritten search query is clearly useful.
Never copy a long user message into query.
Keep reason under 8 words.
"""


def router_node(state: ChatState):
    router = router_llm.with_structured_output(RouteDecision)
    latest_text = latest_user_text(state["messages"])

    try:
        decision = router.invoke(
            [
                SystemMessage(content=ROUTER_PROMPT),
                HumanMessage(content=router_input_text(state["messages"])),
            ],
            config={"callbacks": []},
        )
    except LengthFinishReasonError:
        decision = RouteDecision(
            route="rag",
            query="",
            reason="router length fallback",
        )
    except Exception:
        decision = RouteDecision(route="rag", query="", reason="router fallback")

    decision = normalize_route_decision(decision)

    return {
        "route": decision.route,
        "route_query": decision.query or latest_text,
        "route_reason": decision.reason,
    }


def normalize_route_decision(decision: RouteDecision) -> RouteDecision:
    """Keep the LLM router agentic while preventing it from skipping local evidence."""

    if decision.route == "clarify" and has_indexed_resources():
        return RouteDecision(
            route="rag",
            query=decision.query,
            reason="clarify promoted to rag",
        )

    return decision


def has_indexed_resources() -> bool:
    try:
        return bool(indexed_document_names())
    except FileNotFoundError:
        return False


def rag_call_node(state: ChatState):
    query = state.get("route_query") or latest_user_text(state["messages"])
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "rag_tool",
                        "args": {"query": query},
                        "id": f"rag_{uuid.uuid4().hex}",
                    }
                ],
            )
        ]
    }


def web_call_node(state: ChatState):
    query = state.get("route_query") or latest_user_text(state["messages"])
    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "web_search_tool",
                        "args": {"query": query},
                        "id": f"web_{uuid.uuid4().hex}",
                    }
                ],
            )
        ]
    }


def answer_node(state: ChatState):
    route = state.get("route", "direct")
    system_prompt = SYSTEM_PROMPT

    if route == "clarify":
        system_prompt += "\nAsk concise follow-up questions needed to map the process. Do not invent the missing process details."
    elif route in {"rag", "web"}:
        system_prompt += "\nUse the tool result in the conversation as the primary evidence for this answer."

    system_prompt += "\nUse Mermaid only when a process diagram is requested or genuinely helpful. If you include one, output it in a fenced ```mermaid block."

    messages = [SystemMessage(content=system_prompt), *messages_for_model(state["messages"])]
    response = invoke_complete_answer(messages)
    return {"messages": [response]}


def invoke_complete_answer(messages: list[BaseMessage]) -> AIMessage:
    response = llm.invoke(messages)
    answer_text = message_text(response)

    for _ in range(ANSWER_MAX_CONTINUATIONS):
        if finish_reason(response) != "length":
            break

        continuation_messages = [
            *messages,
            AIMessage(content=answer_text),
            HumanMessage(
                content=(
                    "Continue exactly from where the previous answer stopped. "
                    "Do not repeat earlier content. Finish the answer completely, "
                    "and close any open Markdown or Mermaid code block."
                )
            ),
        ]
        response = llm.invoke(continuation_messages)
        answer_text += message_text(response)

    return AIMessage(content=answer_text)


def finish_reason(message: BaseMessage) -> str:
    metadata = getattr(message, "response_metadata", {}) or {}
    finish = metadata.get("finish_reason")

    if finish:
        return str(finish)

    generations = metadata.get("generations", [])
    if generations and isinstance(generations[0], dict):
        return str(generations[0].get("finish_reason", ""))

    return ""


def route_after_router(state: ChatState):
    route = state.get("route", "direct")

    if route == "rag":
        return "rag_call"

    if route == "web":
        return "web_call"

    return "answer"


def latest_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)

    return ""


def available_resources_text() -> str:
    try:
        names = indexed_document_names()
    except FileNotFoundError:
        names = []

    if not names:
        return "No local resources are currently indexed."

    return "Indexed local resources: " + ", ".join(names[:12])


def message_text(message: BaseMessage) -> str:
    content = message.content

    if isinstance(content, str):
        return content

    return str(content)


def router_input_text(messages: list[BaseMessage]) -> str:
    recent_lines = []

    for message in messages_for_router(messages)[-4:]:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        text = message_text(message).replace("\n", " ").strip()

        if len(text) > 500:
            text = text[:500].rstrip() + "..."

        recent_lines.append(f"{role}: {text}")

    return (
        f"{available_resources_text()}\n\n"
        f"Recent conversation:\n" + "\n".join(recent_lines) + "\n\n"
        f"Latest user request: {latest_user_text(messages)}"
    )


def messages_for_router(messages: list[BaseMessage]) -> list[BaseMessage]:
    compact_messages = []

    for message in messages:
        if isinstance(message, ToolMessage):
            continue

        if isinstance(message, AIMessage) and getattr(message, "tool_calls", None):
            continue

        compact_messages.append(message)

    return compact_messages[-6:]


def messages_for_model(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep tool cycles valid while avoiding replay of old bulky tool outputs."""

    if messages and isinstance(messages[-1], ToolMessage):
        for index in range(len(messages) - 1, -1, -1):
            if isinstance(messages[index], HumanMessage):
                return messages[index:]

    compact_messages = []

    for message in messages:
        if isinstance(message, ToolMessage):
            continue

        if isinstance(message, AIMessage):
            has_content = isinstance(message.content, str) and bool(message.content.strip())
            has_tool_calls = bool(getattr(message, "tool_calls", None))

            if has_tool_calls:
                continue

            if not has_content:
                continue

        compact_messages.append(message)

    return compact_messages[-8:]


setup_thread_metadata()

checkpoint = SqliteSaver(connection)
graph = StateGraph(ChatState)
graph.add_node("tools", tool_node)
graph.add_node("router", router_node)
graph.add_node("rag_call", rag_call_node)
graph.add_node("web_call", web_call_node)
graph.add_node("answer", answer_node)

graph.add_edge(START, "router")
graph.add_conditional_edges(
    "router",
    route_after_router,
    {
        "rag_call": "rag_call",
        "web_call": "web_call",
        "answer": "answer",
    },
)
graph.add_edge("rag_call", "tools")
graph.add_edge("web_call", "tools")
graph.add_edge("tools", "answer")
graph.add_edge("answer", END)

# Compiled LangGraph app with SQLite checkpointing enabled.
# The configurable thread_id controls which conversation memory is loaded.
chatbot = graph.compile(checkpointer=checkpoint)


def run_cli():
    thread_id = "conversation_1"

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["exit", "quit"]:
            break

        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"messages": [HumanMessage(content=user_input)]}
        response = chatbot.invoke(initial_state, config=config)

        print("Assistant:", response["messages"][-1].content)


if __name__ == "__main__":
    run_cli()


# Helper functions for Streamlit frontend
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)
