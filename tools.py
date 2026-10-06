"""The tools the agent can call, and the guards that run before any side effect.

The important idea: policy lives in this file, in code. The prompt asks the
model to follow the policy, but nothing here trusts that it will. Every guard
returns a plain-English error to the model so it can correct itself.

Tool arguments are declared as pydantic models. LangChain sends their JSON
schema to the model, and the same models validate what the model sends back.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

RETURN_WINDOW_DAYS = 30
APPROVAL_THRESHOLD = 5000  # rupees; refunds above this need a human
NON_REFUNDABLE = {"gift_card", "digital"}
SHIPPING_REFUND_REASONS = {"damaged", "wrong_item"}


class _Args(BaseModel):
    # forbid: unknown arguments are an error. strict: "1299" is not a number.
    model_config = ConfigDict(extra="forbid", strict=True)


class LookupOrder(_Args):
    """Fetch one order belonging to the current customer. Call this before any refund."""
    order_id: str = Field(description="For example ORD-1001")


class SearchPolicy(_Args):
    """Search the refund policy. Use it when unsure whether something is allowed."""
    query: str


class IssueRefund(_Args):
    """Refund money for an order. This moves real money. Only call it after lookup_order."""
    order_id: str
    amount: float = Field(gt=0, allow_inf_nan=False, description="Amount in rupees")
    reason: Literal["damaged", "wrong_item", "changed_mind"]


class EscalateToHuman(_Args):
    """Hand the case to a human agent when policy blocks you or you are unsure."""
    summary: str
    order_id: str = ""


ARGS = {
    "lookup_order": LookupOrder,
    "search_policy": SearchPolicy,
    "issue_refund": IssueRefund,
    "escalate_to_human": EscalateToHuman,
}
# OpenAI-style tool definitions; model.bind_tools() accepts these for any provider.
TOOL_SCHEMAS = [
    {"type": "function",
     "function": {"name": name, "description": model.__doc__, "parameters": model.model_json_schema()}}
    for name, model in ARGS.items()
]


class ToolError(Exception):
    """A problem the model can fix. The message is sent back to the model."""


class NeedsApproval(Exception):
    """A valid refund that is too large for the agent to issue on its own."""

    def __init__(self, request):
        super().__init__("needs approval")
        self.request = request


class Store:
    """Orders, the policy text, and a ledger of everything that changed.

    The ledger is a JSON file shared by all conversations, so the
    one-refund-per-order check works across sessions and restarts.
    (One process at a time; there is no file locking.)
    """

    def __init__(self, data_dir, ledger_path):
        data_dir = Path(data_dir)
        orders = json.loads((data_dir / "orders.json").read_text())
        self.orders = {o["order_id"]: o for o in orders}
        self.policy_text = (data_dir / "policy.md").read_text()  # the whole policy, for the critic
        self.policy = [s.strip() for s in re.split(r"^## ", self.policy_text, flags=re.M) if s.strip()]
        self.ledger_path = Path(ledger_path)
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []

    def add(self, kind, prefix, **fields):
        number = sum(1 for e in self.ledger if e["type"] == kind) + 1
        entry = {"id": f"{prefix}-{number:04d}", "type": kind, **fields}
        self.ledger.append(entry)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.ledger, indent=2))
        os.replace(tmp, self.ledger_path)  # atomic: never half a file
        return entry

    def refund_for(self, order_id):
        return next((e for e in self.ledger if e["type"] == "refund" and e["order_id"] == order_id), None)


class Toolbox:
    """Runs tool calls for one graph step.

    It reads the verified orders from the graph state and collects what
    changed (new_orders, actions) for the node to return as a state update.
    call() returns {"ok": True, ...} or {"ok": False, "error": "..."}; the only
    thing it raises is NeedsApproval.
    """

    def __init__(self, store, customer_email, session_id, verified_orders):
        self.store = store
        self.customer_email = customer_email
        self.session_id = session_id
        self.orders = dict(verified_orders)  # orders looked up in this session
        self.new_orders = {}
        self.actions = []

    def call(self, name, raw_args, approved=False):
        try:
            if name not in ARGS:
                raise ToolError(f"Unknown tool '{name}'. Available tools: {sorted(ARGS)}.")
            if not isinstance(raw_args, dict):  # LangChain could not parse the JSON
                raise ToolError("Arguments were not valid JSON. Send a JSON object.")
            try:
                args = ARGS[name].model_validate(raw_args).model_dump()
            except ValidationError as e:
                problems = "; ".join(f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors())
                raise ToolError(f"Bad arguments. {problems}")
            if name == "issue_refund":
                args["approved"] = approved
            return {"ok": True, **getattr(self, name)(**args)}
        except ToolError as e:
            return {"ok": False, "error": str(e)}

    # --- read-only tools -------------------------------------------------

    def lookup_order(self, order_id):
        order_id = order_id.strip().upper()
        order = self.store.orders.get(order_id)
        # Same message for "does not exist" and "belongs to someone else",
        # so the agent cannot be used to probe other customers' orders.
        if order is None or order["customer_email"] != self.customer_email:
            raise ToolError(f"No order {order_id} found for this customer. Ask them to check the order ID.")
        view = {k: v for k, v in order.items() if k != "customer_email"}
        existing = self.store.refund_for(order_id)
        view["existing_refund"] = existing["id"] if existing else None
        self.orders[order_id] = self.new_orders[order_id] = view
        return {"order": view}

    def search_policy(self, query):
        words = {w for w in re.findall(r"[a-z]+", query.lower()) if len(w) > 2}
        scored = []
        for section in self.store.policy:
            hits = sum(1 for w in words if w in section.lower())
            if hits:
                scored.append((hits, section))
        scored.sort(key=lambda pair: -pair[0])
        if not scored:
            return {"results": [], "note": "No matching policy section. If unsure, escalate_to_human."}
        return {"results": [section for _, section in scored[:2]]}

    # --- tools with side effects ----------------------------------------

    def issue_refund(self, order_id, amount, reason, approved=False):
        order_id = order_id.strip().upper()
        if order_id not in self.orders:
            raise ToolError(
                f"Call lookup_order for {order_id} first. Refunds are only allowed on orders verified in this session."
            )
        order = self.store.orders[order_id]
        if order["status"] != "delivered":
            raise ToolError("This order has not been delivered, so it cannot be refunded here. Use escalate_to_human.")
        if order["category"] in NON_REFUNDABLE:
            raise ToolError(f"Category '{order['category']}' is non-refundable. Explain the policy to the customer.")
        if order["delivered_days_ago"] > RETURN_WINDOW_DAYS:
            raise ToolError(
                f"Delivered {order['delivered_days_ago']} days ago, outside the {RETURN_WINDOW_DAYS}-day window. "
                "Do not refund. Use escalate_to_human if the customer wants a review."
            )
        existing = self.store.refund_for(order_id)
        if existing:
            raise ToolError(f"{order_id} already has refund {existing['id']}. Only one refund per order.")
        limit = order["item_price"] + (order["shipping_fee"] if reason in SHIPPING_REFUND_REASONS else 0)
        amount = round(amount, 2)
        if amount > limit:
            raise ToolError(f"Amount {amount} is more than the {limit} refundable for reason '{reason}'. Check the order.")

        # Every guard has passed. Large refunds still stop here for a person.
        if amount > APPROVAL_THRESHOLD and not approved:
            raise NeedsApproval({"order_id": order_id, "item": order["item"], "amount": amount,
                                 "reason": reason, "customer": self.customer_email})
        entry = self.store.add("refund", "RF", order_id=order_id, amount=amount, reason=reason,
                               session_id=self.session_id, approved_by="human" if approved else "policy")
        self.actions.append({"type": "refund", "id": entry["id"], "order_id": order_id, "amount": amount})
        # Keep the verified copy current, so later turns do not see "existing_refund": null.
        self.orders[order_id] = self.new_orders[order_id] = {**self.orders[order_id], "existing_refund": entry["id"]}
        return {"status": "refunded", "refund_id": entry["id"], "amount": amount}

    def escalate_to_human(self, summary, order_id=""):
        entry = self.store.add("escalation", "ESC", order_id=order_id or None, summary=summary,
                               session_id=self.session_id)
        self.actions.append({"type": "escalation", "id": entry["id"], "summary": summary})
        return {"status": "escalated", "ticket_id": entry["id"]}
