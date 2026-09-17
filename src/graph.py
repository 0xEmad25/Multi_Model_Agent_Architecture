from typing import TypedDict, Literal, Any
from langgraph.graph import StateGraph, START, END

class SupportState(TypedDict, total=False):
    user_message: str
    route: str
    intent: str
    confidence: float
    context: list[dict]
    tool_results: list[dict]
    answer: str
    escalate: bool
    trace_id: str


def route_node(state: SupportState):
    decision = baseline_router(state["user_message"])
    return {
        "route": decision.get("route") or decision.get("intent"),
        "intent": decision.get("intent", ""),
        "confidence": decision.get("confidence", 0.0),
    }


def qa_node(state: SupportState):
    # Retrieve context first, then call Model B.
    kb = knowledge_base_search.invoke({"query": state["user_message"]})
    answer = run_qa_model(
        question=state["user_message"],
        contexts=kb["passages"],
    )
    return {"context": kb["passages"], "answer": answer}


def tool_node(state: SupportState):
    result = diagnostic_runbook.invoke({
        "issue_type": state.get("intent", "unknown"),
        "service": infer_service(state["user_message"]),
        "symptom": state["user_message"],
    })
    return {"tool_results": [result]}


def support_node(state: SupportState):
    answer = run_support_model(
        user_message=state["user_message"],
        context=state.get("context", []),
        tool_results=state.get("tool_results", []),
    )
    return {"answer": answer}


def escalation_node(state: SupportState):
    result = escalate_to_human.invoke({
        "reason": "Policy or unresolved technical incident",
        "evidence": str(state),
    })
    return {"answer": f"Escalated to human support: {result}"}


def choose_after_router(state: SupportState) -> Literal["qa", "tools", "support", "escalate"]:
    # TODO(student 8): implement the hybrid routing policy you measured earlier.
    if state.get("route") == "escalate":
        return "escalate"
    if state.get("route") == "qa":
        return "qa"
    if state.get("route") in {"database", "gpu", "deployment", "network"}:
        return "tools"
    return "support"

builder = StateGraph(SupportState)
builder.add_node("route", route_node)
builder.add_node("qa", qa_node)
builder.add_node("tools", tool_node)
builder.add_node("support", support_node)
builder.add_node("escalate", escalation_node)

builder.add_edge(START, "route")
builder.add_conditional_edges("route", choose_after_router)
builder.add_edge("qa", END)
builder.add_edge("tools", "support")
builder.add_edge("support", END)
builder.add_edge("escalate", END)

graph = builder.compile()