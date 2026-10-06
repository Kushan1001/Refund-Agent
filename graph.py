from __future__ import annotations

import json
import operator
from collections import Counter
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import (AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage,
                                     ToolMessage, trim_messages)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from tools import TOOL_SCHEMAS, NeedsApproval, Toolbox

SYSTEM_PROMPT = """You are a refund support agent for an online shop.

How to work:
- Look up the order with lookup_order before deciding anything about it.
- The full refund policy is included below. Follow it, and quote it when the customer
  asks about a rule. search_policy is also available.
- The tools enforce the policy. If a tool returns an error, read it and change
  what you do. Do not repeat the same call hoping for a different result.
- Never tell the customer a refund has happened unless issue_refund returned
  status "refunded".
- If you are blocked or unsure, call escalate_to_human with a short summary.
Keep replies short and plain."""

NOT_RUN = {"ok": False, "error": "Not run: the agent was stopped."}
REPEAT = {"ok": False, "error": "You just made this exact call and nothing has changed since. "
                                "Use the earlier result or take a different step."}
REJECTED = {"ok": False, "error": "A human reviewer rejected this refund. Tell the customer. Do not retry."}

# --- the critic ---------------------------------------------------------------
# A second model call that runs once, just before the customer sees a reply.
# It reviews everything the agent did this turn, plus the reply itself.

CRITIC_PROMPT = """You review a refund support agent's work before the customer sees its reply.
You are given the shop's refund policy (below), the recent conversation, every tool the
agent called this turn with the results, the actions that really happened, and the draft reply.
Judge everything against the policy and the tool results, not against your own assumptions.

Reject if the agent made a wrong decision:
- refunded with a reason ("damaged", "wrong_item" or "changed_mind") that the customer's
  words anywhere in the recent conversation do not support,
- refunded an order the customer did not ask to refund,
- refunded less than the customer is owed, when they did not ask for less. The full refund is
  item_price + shipping_fee for "damaged" or "wrong_item", and item_price for "changed_mind".
  (item_price + shipping_fee is allowed for those reasons; it is not "more than the item price".)
  Example: item_price 1299, shipping_fee 49, reason "damaged": the full refund is 1348, so
  1299 is too little. The agent cannot refund the same order twice, so it must call
  escalate_to_human to have the difference refunded,
- did something the policy does not allow, or
- gave up or refused when the policy and tool results show it could have helped.
Reject if the reply is wrong:
- it says something happened that is in neither "Actions taken earlier in this session" nor
  "Actions taken this turn" (an action from earlier in the session really did happen),
- it states an amount or rule that contradicts the tool results or the policy,
- it promises something the policy does not allow,
- it reveals anything about another customer's orders, or
- it does not answer what the customer asked.
Otherwise approve. Wording and tone are not reasons to reject.
When you reject, say what is wrong and what the agent should do. A refund that already
happened cannot be undone by the agent: tell it to call escalate_to_human so a person can fix it."""


class Verdict(BaseModel):
    """The critic's decision."""
    approved: bool
    problem: str = Field("", description="If rejected: what is wrong and how to fix it, in one or two "
                                         "sentences. Empty if approved.")


def merge(old, new):
    return {**old, **new}


class State(TypedDict, total=False):
    """Everything the checkpointer saves. Annotated fields are merged, others replaced."""

    messages: Annotated[list[AnyMessage], add_messages]  # full transcript, never trimmed
    customer_email: str
    orders: Annotated[dict, merge]            # orders verified with lookup_order
    actions: Annotated[list, operator.add]    # side effects that really happened
    trace: Annotated[list, operator.add]      # one line per tool call, for debugging
    # Reset at the start of every customer message:
    steps: int                                # model calls so far this turn
    seen: dict                                # identical tool calls since the last success
    stop_reason: Optional[str]
    pending: list                             # tool calls waiting for a human
    turn_start: int                           # len(actions) when the turn began
    critique: Optional[dict]                  # critic's objection to the last draft reply
    revisions: int                            # drafts the critic rejected this turn
    reply: str                                # what the customer is shown


def text_of(message):
    """Message text, whether the provider returns a string or content blocks."""
    content = message.content
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") for b in content if isinstance(b, dict))


def describe(action, internal=False):
    """One action in plain words. internal=True adds notes meant for the critic, not the customer."""
    amount = f"₹{action['amount']:,.2f} " if "amount" in action else ""
    kind = action["type"]
    if kind == "refund":
        return f"refund of {amount}on {action['order_id']} ({action['id']})"
    if kind == "refund_rejected":
        return f"refund of {amount}on {action['order_id']} rejected by a human, no money moved"
    text = f"handed to a human ({action['id']})"
    if internal and action.get("summary"):
        text += f", summary for the human: {action['summary']}"
    return text


def build_context(state, max_messages=30, keep_recent=8, policy=""):
    """The messages actually sent to the model, kept to a bounded size.

    The full transcript stays in the checkpoint. What the model sees is:
      1. a system prompt with the full policy, and the verified orders and actions,
      2. the most recent whole turns that fit in max_messages,
      3. with older tool outputs shortened.
    Because facts are restated in (1), dropping old messages in (2) and (3)
    does not make the model forget an order it already looked up.
    """
    messages = state["messages"]
    # start_on="human" cuts only at the start of a turn, so a tool reply is
    # never separated from the assistant message that asked for it.
    visible = trim_messages(messages, strategy="last", token_counter=len,
                            max_tokens=max_messages, start_on="human")
    if not visible:  # the current turn alone is over the limit: keep all of it
        last_human = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
        visible = messages[last_human:]

    shortened = []
    for i, m in enumerate(visible):
        old = i < len(visible) - keep_recent
        if old and isinstance(m, ToolMessage) and len(m.content) > 160:
            m = m.model_copy(update={"content": m.content[:120] + " ...[trimmed]"})
        shortened.append(m)

    system = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Refund policy:\n{policy.strip()}\n\n"
        f"Current customer: {state['customer_email']}\n"
        f"Orders verified this session: {json.dumps(state.get('orders', {}))}\n"
        f"Actions already taken: {json.dumps(state.get('actions', []))}"
    )
    dropped = len(messages) - len(visible)
    if dropped:
        system += f"\n({dropped} earlier messages are not shown. Rely on the facts above.)"
    critique = state.get("critique")
    if critique:
        system += ("\n\nA reviewer rejected your last draft reply, so the customer has not seen it.\n"
                   f"Draft: {critique['draft']}\nProblem: {critique['problem']}\n"
                   "Fix the problem (call tools first if needed, e.g. escalate_to_human), then write a new reply.")
    return [SystemMessage(system)] + shortened


def render_for_critic(messages, earlier=6):
    """The conversation as plain text, for the critic to read.

    Earlier turns are shown as just the customer's and agent's words (the last
    `earlier` of them), so a reason given before still counts. The current
    turn is shown in full, with every tool call and result.
    """
    start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    said = [m for m in messages[:start]
            if isinstance(m, (HumanMessage, AIMessage)) and text_of(m).strip()][-earlier:]
    lines = ["Earlier in the conversation:"] if said else []
    lines += [f"  {'Customer' if isinstance(m, HumanMessage) else 'Agent'}: {text_of(m)}" for m in said]
    lines.append("This turn:")
    for m in messages[start:]:
        if isinstance(m, HumanMessage):
            lines.append(f"Customer: {text_of(m)}")
        elif isinstance(m, ToolMessage):
            lines.append(f"Tool result: {text_of(m)}")
        elif isinstance(m, AIMessage):
            lines += [f"Agent called {c['name']}({json.dumps(c['args'])})" for c in m.tool_calls]
    return "\n".join(lines)


def build_graph(model, store, checkpointer=None, max_steps=8, max_messages=30, keep_recent=8,
                critic=None, max_revisions=2):
    """model is any LangChain chat model (or anything with bind_tools and invoke).

    critic is an optional second chat model (it may be the same one) that reviews
    the agent's decisions and reply before the customer sees anything.
    """
    llm = model.bind_tools(TOOL_SCHEMAS)
    judge = critic.with_structured_output(Verdict) if critic else None

    def review(material):
        """Ask the critic. Returns a Verdict, or None if the call failed."""
        try:
            verdict = judge.invoke([SystemMessage(f"{CRITIC_PROMPT}\n\nRefund policy:\n{store.policy_text}"),
                                    HumanMessage(material)])
        except Exception:
            return None
        return verdict if isinstance(verdict, Verdict) else None

    def toolbox(state, config):
        return Toolbox(store, state["customer_email"], config["configurable"]["thread_id"],
                       state.get("orders", {}))

    def start_turn(state: State):
        return {"steps": 0, "seen": {}, "stop_reason": None, "pending": [],
                "turn_start": len(state.get("actions", [])), "critique": None, "revisions": 0}

    def call_model(state: State):
        reply = llm.invoke(build_context(state, max_messages, keep_recent, store.policy_text))
        return {"messages": [reply], "steps": state["steps"] + 1}

    def after_model(state: State):
        last = state["messages"][-1]
        if last.tool_calls or last.invalid_tool_calls:
            return "tools"
        if not text_of(last).strip():
            return "give_up"
        return "critic" if judge else "finish"

    def run_tools(state: State, config: RunnableConfig):
        last = state["messages"][-1]
        box = toolbox(state, config)
        seen = Counter(state["seen"])
        stop, replies, pending, trace = None, [], [], []

        def answer(call, result):
            # Every tool call gets a reply, even when skipped, so the
            # transcript stays valid for the next model call.
            replies.append(ToolMessage(json.dumps(result), tool_call_id=call["id"] or ""))
            trace.append({"tool": call["name"], "args": call["args"], "ok": result["ok"],
                          "error": result.get("error")})

        # invalid_tool_calls are calls whose arguments LangChain could not parse.
        for call in list(last.tool_calls) + list(last.invalid_tool_calls):
            signature = f"{call['name']}:{json.dumps(call['args'], sort_keys=True)}"
            seen[signature] += 1
            if stop:
                answer(call, NOT_RUN)
            elif seen[signature] >= 3:
                stop = f"it repeated the same {call['name']} call three times"
                answer(call, NOT_RUN)
            elif seen[signature] == 2:
                answer(call, REPEAT)
            else:
                try:
                    result = box.call(call["name"], call["args"])
                except NeedsApproval as need:
                    pending.append({**call, "request": need.request})  # answered in `approval`
                    continue
                if result["ok"]:
                    # Something changed, so retrying an earlier failed call is
                    # now reasonable (e.g. refund again after the lookup).
                    seen = Counter({signature: seen[signature]})
                answer(call, result)

        if stop:
            for call in pending:
                answer(call, NOT_RUN)
            pending = []
        return {"messages": replies, "orders": box.new_orders, "actions": box.actions,
                "trace": trace, "seen": dict(seen), "stop_reason": stop, "pending": pending}

    def after_tools(state: State):
        if state["stop_reason"]:
            return "give_up"
        if state["pending"]:
            return "approval"
        return "give_up" if state["steps"] >= max_steps else "model"

    def approval(state: State, config: RunnableConfig):
        # interrupt() saves the state and stops the run. When a human resumes
        # it, this node starts again from the top and interrupt() returns their
        # answer, so nothing with a side effect may come before these calls.
        decisions = [interrupt(call["request"]) for call in state["pending"]]
        box = toolbox(state, config)
        replies, trace = [], []
        for call, approved in zip(state["pending"], decisions):
            if approved is True:
                # Guards run again: the order may have changed while we waited.
                result = box.call(call["name"], call["args"], approved=True)
            else:
                result = REJECTED
                box.actions.append({"type": "refund_rejected", **{k: call["request"][k] for k in ("order_id", "amount")}})
            replies.append(ToolMessage(json.dumps(result), tool_call_id=call["id"]))
            trace.append({"tool": call["name"], "args": call["args"], "ok": result["ok"],
                          "error": result.get("error"), "human": "approved" if approved is True else "rejected"})
        return {"messages": replies, "orders": box.new_orders, "actions": box.actions, "trace": trace, "pending": []}

    def after_approval(state: State):
        return "give_up" if state["steps"] >= max_steps else "model"

    def critic_node(state: State):
        """Review this turn's decisions and the draft reply before the customer sees it."""
        draft = state["messages"][-1]
        # Earlier actions too, so "yes, your refund went through" is not mistaken for a false claim.
        earlier = [describe(a, internal=True) for a in state["actions"][:state["turn_start"]]]
        done = [describe(a, internal=True) for a in state["actions"][state["turn_start"]:]]
        # Refunds reach the critic through the action lists. Leaving existing_refund in the
        # orders made it read this turn's refund as an earlier one, i.e. a double refund.
        orders = {oid: {k: v for k, v in o.items() if k != "existing_refund"}
                  for oid, o in state.get("orders", {}).items()}
        material = (f"{render_for_critic(state['messages'])}\n\n"
                    f"Orders verified this session: {json.dumps(orders, ensure_ascii=False)}\n"
                    f"Actions taken earlier in this session: {json.dumps(earlier or ['none'], ensure_ascii=False)}\n"
                    f"Actions taken this turn: {json.dumps(done or ['none'], ensure_ascii=False)}\n\n"
                    f"Draft reply to review:\n{text_of(draft)}")
        verdict = review(material)
        if verdict is None:
            # Fail open: the footer built from `actions` still shows the truth.
            return {"critique": None, "trace": [{"tool": "(critic) review", "args": "", "ok": True,
                                                 "error": "critic unavailable, reply sent unchecked"}]}
        if verdict.approved:
            return {"critique": None, "trace": [{"tool": "(critic) review", "args": "", "ok": True, "error": None}]}
        # Rejected: drop the draft from the transcript and tell the model why.
        revisions = state.get("revisions", 0) + 1
        update = {"messages": [RemoveMessage(id=draft.id)], "revisions": revisions,
                  "critique": {"draft": text_of(draft), "problem": verdict.problem},
                  "trace": [{"tool": "(critic) review", "args": text_of(draft)[:80], "ok": False,
                             "error": verdict.problem}]}
        if revisions > max_revisions:
            update["stop_reason"] = f"the critic rejected its work {revisions} times"
        return update

    def after_critic(state: State):
        if state["critique"] is None:
            return "finish"
        if state["stop_reason"] or state["steps"] >= max_steps:
            return "give_up"
        return "model"

    def give_up(state: State, config: RunnableConfig):
        """The model could not finish. Stop safely and hand over to a person."""
        if state["stop_reason"]:
            reason = state["stop_reason"]
        elif isinstance(state["messages"][-1], AIMessage):
            reason = "the model returned an empty reply"
        else:
            reason = f"it used all {max_steps} steps without finishing"
        entry = store.add("escalation", "ESC", order_id=None, summary=f"Agent stopped because {reason}.",
                          session_id=config["configurable"]["thread_id"])
        text = ("I wasn't able to finish this safely, so I've passed it to a human colleague "
                f"(ticket {entry['id']}). They will follow up with you.")
        return {"messages": [AIMessage(text)],
                "actions": [{"type": "escalation", "id": entry["id"], "summary": f"Agent stopped because {reason}."}],
                "trace": [{"tool": "(agent stopped)", "args": reason, "ok": False, "error": reason}]}

    def finish(state: State):
        # The model's words are never the record of what happened. The footer
        # is built from the action log, so a reply that claims a refund that
        # was never issued is visibly contradicted.
        done = state["actions"][state["turn_start"]:]
        footer = "; ".join(describe(a) for a in done) if done else "none"
        return {"reply": f"{text_of(state['messages'][-1]).strip()}\n\n[Actions recorded: {footer}]"}

    graph = StateGraph(State)
    graph.add_node("start_turn", start_turn)
    graph.add_node("model", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("approval", approval)
    graph.add_node("critic", critic_node)
    graph.add_node("give_up", give_up)
    graph.add_node("finish", finish)
    graph.add_edge(START, "start_turn")
    graph.add_edge("start_turn", "model")
    graph.add_conditional_edges("model", after_model, ["tools", "critic", "finish", "give_up"])
    graph.add_conditional_edges("tools", after_tools, ["model", "approval", "give_up"])
    graph.add_conditional_edges("approval", after_approval, ["model", "give_up"])
    graph.add_conditional_edges("critic", after_critic, ["model", "finish", "give_up"])
    graph.add_edge("give_up", "finish")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)


# --- small helpers for callers ---------------------------------------------

def thread(session_id):
    # recursion_limit is LangGraph's own backstop; max_steps should stop the run first.
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 60}


def send(app, session_id, email, text):
    """Run one customer message. Returns the state, or a paused run to approve."""
    return app.invoke({"messages": [HumanMessage(text)], "customer_email": email}, thread(session_id))


def waiting_for_approval(app, session_id):
    """The refund requests a paused run is waiting on (empty if it is not paused)."""
    return [i.value for i in app.get_state(thread(session_id)).interrupts]


def decide(app, session_id, approved):
    """Answer the next pending approval as the human reviewer."""
    return app.invoke(Command(resume=bool(approved)), thread(session_id))
