from __future__ import annotations

import argparse
import os
import sqlite3
import uuid
from pathlib import Path

from graph import build_graph, decide, send, thread, waiting_for_approval
from tools import Store

ROOT = Path(__file__).parent
DATA = ROOT / "data"
SESSIONS = ROOT / "sessions"

OPENAI_MODEL = "openai:gpt-4o-mini"
CRITIC_MODEL = "openai:gpt-4.1"  
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  


def print_trace(trace, start=0):
    for step in trace[start:]:
        mark = "ok     " if step["ok"] else "BLOCKED"
        human = f"  [human {step['human']}]" if step.get("human") else ""
        print(f"   {mark} {step['tool']} {step['args']}{human}")
        if step["error"]:
            print(f"           -> {step['error']}")


def ask_human(app, session_id):
    state = None
    while requests := waiting_for_approval(app, session_id):
        r = requests[0]
        answer = input(f"\nSUPERVISOR: approve ₹{r['amount']:,.2f} refund on {r['order_id']} "
                       f"({r['item']}, {r['reason']}) for {r['customer']}? [y/n] ")
        state = decide(app, session_id, approved=answer.strip().lower() == "y")
    return state


def chat(email, session_id, use_critic=True):
    from langchain.chat_models import init_chat_model
    from langgraph.checkpoint.sqlite import SqliteSaver

    session_id = session_id or uuid.uuid4().hex[:8]
    SESSIONS.mkdir(exist_ok=True)
    saver = SqliteSaver(sqlite3.connect(SESSIONS / "checkpoints.db", check_same_thread=False))
    model = init_chat_model(OPENAI_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    critic = init_chat_model(CRITIC_MODEL, temperature=0, api_key=OPENAI_API_KEY) if use_critic else None
    app = build_graph(model, Store(DATA, SESSIONS / "ledger.json"), saver, critic=critic)

    saved = app.get_state(thread(session_id)).values
    email = saved.get("customer_email", email)
    shown = len(saved.get("trace", []))
    print(f"Session {session_id} for {email}, critic {'on' if use_critic else 'off'}. Type 'quit' to stop.")
    state = ask_human(app, session_id)  # a run left paused by an earlier session
    while True:
        if state:
            print_trace(state["trace"], shown)
            shown = len(state["trace"])
            print(f"\nAgent: {state['reply']}")
        try:
            text = input("\nYou: ").strip()
        except EOFError:
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            state = None
            continue
        state = send(app, session_id, email, text)
        state = ask_human(app, session_id) or state


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", default="aarav@example.com")
    parser.add_argument("--session")
    parser.add_argument("--no-critic", action="store_true", help="run without the critic model")
    args = parser.parse_args()
    if not OPENAI_API_KEY:
        raise SystemExit("Set OPENAI_API_KEY first, e.g.  export OPENAI_API_KEY=sk-...")
    chat(args.email, args.session, use_critic=not args.no_critic)


if __name__ == "__main__":
    main()
