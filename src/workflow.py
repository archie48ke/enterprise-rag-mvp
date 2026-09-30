"""
LangGraph workflow:

    START -> Manager Node -> Retrieval Node -> Answer Node -> END

Exactly two LLM-oriented nodes (Manager, Answer). Retrieval is plain Python.
"""

from typing import List, TypedDict

from langgraph.graph import StateGraph, END

from src.agents import classify_domain, generate_answer
from src.retriever import retrieve, DomainIndexNotFound
from src.schemas import Chunk, Citation
from src.config import NO_ANSWER_MESSAGE, TOP_K


class GraphState(TypedDict):
    question: str
    domain: str
    retrieved_chunks: List[Chunk]
    answer: str
    citations: List[Citation]
    error: str


def manager_node(state: GraphState) -> GraphState:
    """Determine the domain for the incoming question. Does not answer it."""
    domain = classify_domain(state["question"])
    state["domain"] = domain
    return state


def retrieval_node(state: GraphState) -> GraphState:
    """Search only the FAISS index for the routed domain."""
    try:
        chunks = retrieve(state["question"], state["domain"], top_k=TOP_K)
        state["retrieved_chunks"] = chunks
    except DomainIndexNotFound as exc:
        state["retrieved_chunks"] = []
        state["error"] = str(exc)
    return state


def answer_node(state: GraphState) -> GraphState:
    """Generate a grounded answer from the retrieved chunks and build citations."""
    if state.get("error"):
        state["answer"] = (
            "The knowledge base for this department has not been built yet. "
            "Please run: python scripts/build_indexes.py"
        )
        state["citations"] = []
        return state

    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        state["answer"] = NO_ANSWER_MESSAGE
        state["citations"] = []
        return state

    answer_text = generate_answer(state["question"], chunks)
    state["answer"] = answer_text

    # Citations are built from retrieval metadata, never invented by the LLM.
    # De-duplicate (filename, page) pairs while preserving order.
    seen = set()
    citations: List[Citation] = []
    for c in chunks:
        key = (c.filename, c.page_number)
        if key not in seen:
            seen.add(key)
            citations.append(Citation(filename=c.filename, page_number=c.page_number))
    state["citations"] = citations

    return state


def build_workflow():
    """Compile the LangGraph app: Manager -> Retrieval -> Answer."""
    graph = StateGraph(GraphState)

    graph.add_node("manager", manager_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("answer", answer_node)

    graph.set_entry_point("manager")
    graph.add_edge("manager", "retrieval")
    graph.add_edge("retrieval", "answer")
    graph.add_edge("answer", END)

    return graph.compile()


_app = None


def get_app():
    global _app
    if _app is None:
        _app = build_workflow()
    return _app


def _initial_state(question: str) -> GraphState:
    return {
        "question": question,
        "domain": "",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "error": "",
    }


def run_workflow(question: str) -> GraphState:
    return get_app().invoke(_initial_state(question))


def stream_workflow(question: str):
    """
    Same pipeline as run_workflow, but yields (node_name, state) after each
    node finishes so the UI can show live progress for every step.
    """
    for update in get_app().stream(_initial_state(question), stream_mode="updates"):
        for node_name, state in update.items():
            yield node_name, state
