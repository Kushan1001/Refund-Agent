# Refund Support Agent

An AI support agent for a mock online shop. It looks up orders, issues refunds and escalates to humans, built with **LangGraph**, **LangChain** and OpenAI models.

**The idea: the model proposes, code decides.** The model is never trusted with money on its own word:

1. **Checks in code.** Every refund passes policy checks written in Python before anything is recorded.
2. **Human approval.** Refunds above ₹5,000 pause until a person says yes.
3. **A critic.** A second, stronger model reviews each reply before the customer sees it.
4. **A ledger-built footer.** Every reply ends with the actions that really happened, taken from the ledger and not from the model's words.

## Demo

```
You: ORD-1001 arrived broken, refund it
   ok      lookup_order {'order_id': 'ORD-1001'}
   ok      issue_refund {'order_id': 'ORD-1001', 'amount': 1299, 'reason': 'damaged'}
   BLOCKED (critic) review -> Refunded only the item price; the ₹49 shipping fee is also owed
                              for a damaged item. Escalate to a human.
   ok      escalate_to_human {'summary': 'Shipping fee of ₹49 still owed…'}
   ok      (critic) review
Agent: Your ₹1,299 refund is done. I've escalated the ₹49 shipping fee to a colleague.
[Actions recorded: refund of ₹1,299.00 on ORD-1001 (RF-0001); handed to a human (ESC-0001)]
```
<sub>Shortened from a real run.</sub>


- **Tools:** `lookup_order`, `search_policy`, `issue_refund`, `escalate_to_human`. Arguments are strict pydantic models, so unknown fields and wrong types are rejected.
- **State:** a SQLite checkpointer saves after every step. Conversations, and runs paused for approval, survive a restart. A separate ledger enforces one refund per order across sessions.
- **Context:** the model sees the last 30 messages, trimmed at turn boundaries, with older tool outputs shortened. The policy, the orders looked up and the actions taken are restated in the system prompt on every call, so trimming never makes it forget a fact.
- **Errors:** blocked calls go back to the model as plain-English tool results, so it usually corrects itself.

### Refund checks, in order

`issue_refund` only records a refund if all of these pass ([`tools.py`](tools.py)):

1. The order was looked up in this session and belongs to the current customer.
2. It has been delivered.
3. Its category is refundable (not `gift_card` or `digital`).
4. It was delivered within the last 30 days.
5. It has no earlier refund in the ledger.
6. The amount is at most the item price, plus the shipping fee when the reason is `damaged` or `wrong_item`.
7. If the amount is above ₹5,000, a human approves it. The checks run again after approval, in case something changed while waiting.
## The critic

The critic (`gpt-4.1`) checks what code can't: does the refund reason match what the customer said, and is the reply true?

- It sees the **same facts** as the main model (`gpt-4o-mini`): the full policy, the verified orders, all actions, and this turn's tool calls with their results.
- It has **no tools**, and returns a structured `{approved, problem}`.
- **Rejected:** the draft is removed from the transcript and the model is told why. After 3 rejections in one turn the case goes to a human.
- **Critic unreachable:** the reply is sent anyway and flagged in the trace. The footer still shows what really happened.

## Evaluation

[`eval.py`](eval.py) runs 11 difficult scenarios against the real models. `--sabotage` tells the main model to make mistakes, to test whether the critic catches them. There was one run per configuration, so treat the results as indicative:


## Quick start

Requires Python 3.10+ and an OpenAI API key.

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...                        # read from the environment, never from a file

python run.py --email aarav@example.com --session demo1
```

| Flag | Meaning |
|---|---|
| `--email` | The customer you are chatting as. Default `aarav@example.com`. |
| `--session` | A name for the conversation. Reuse it to resume; a resumed session keeps its original email. Random if omitted. |
| `--no-critic` | Run without the critic model. |

Type `quit` to stop. If you stop while a refund is waiting for approval, the supervisor prompt comes back the next time you open that session.

### Things to try

| Message | What happens |
|---|---|
| `ORD-1001 arrived broken, refund it` | Refund, then the critic spots the missing shipping fee |
| `I changed my mind about ORD-1001` | Changed-mind refund: item price only |
| `refund my laptop ORD-1002` | Pauses for supervisor approval |
| `refund ORD-1003` | Blocked: outside the 30-day window |
| `refund ORD-2001` | "Not found": the order belongs to another customer |

An order can only be refunded once, across all sessions. Reset between attempts with `rm -rf sessions/`.


## Project structure

```
graph.py                state, nodes, routing, context, critic
tools.py                tools, policy checks, ledger
run.py                  terminal chat and model settings
eval.py                 scenario evaluation
data/                   orders.json, policy.md
sessions/               created at runtime: checkpoints.db, ledger.json
```

