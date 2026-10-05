"""Vestwise chat UI (spec FR-19, FR-9).

Run:  streamlit run ui/streamlit_app.py      (or bash scripts/run_all.sh for API + UI)

Streamlit re-runs this whole script top to bottom on every interaction; anything
that must survive a rerun (the conversation, the dilution result) lives in
`st.session_state`. The UI only calls the FastAPI backend (ui/api_client.py);
it never touches Mongo or the LLM. Which user is "signed in" is just the
X-User-Id header it sends; the API enforces what that user may see.
"""

import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

# `streamlit run ui/streamlit_app.py` puts ui/ (not the repo root) on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.config import settings  # noqa: E402
from app.rag.prompts import ACCESS_DENIED_MESSAGE, NOT_FOUND_MESSAGE  # noqa: E402
from ui import api_client as api  # noqa: E402

DEFAULT_AS_OF = date(2026, 10, 3)  # spec §8.4 demo date
HISTORY_TURNS = 6  # FR-16; the agent also trims to 6
MAX_ARG_CHARS = 40


@dataclass(frozen=True)
class DemoUser:
    """One entry in the user switcher. `role` only decides which tabs to show; the API enforces access."""

    user_id: str
    name: str
    role: str


USERS = [
    DemoUser("u_priya", "Priya Sharma", "employee"),
    DemoUser("u_rahul", "Rahul Verma", "employee"),
    DemoUser("u_arjun", "Arjun Mehta", "admin"),
]
USERS_BY_ID = {u.user_id: u for u in USERS}


# --- session state ---

def init_state() -> None:
    """Create session keys on the first run of a session."""
    st.session_state.setdefault("user_id", USERS[0].user_id)
    st.session_state.setdefault("as_of", DEFAULT_AS_OF)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("dilution_result", None)  # not "dilution": that key belongs to the form


def reset_conversation() -> None:
    """Called when the user changes: a new person must never see the previous person's chat."""
    st.session_state.messages = []
    st.session_state.dilution_result = None


def history_for_api(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Earlier turns to send as context: user/assistant text only, errors left out, last 6."""
    turns = [{"role": m["role"], "content": m["content"]} for m in messages if m.get("kind") != "error"]
    return turns[-HISTORY_TURNS:]


def kind_of(answer: str) -> str:
    """How to style an answer: the API's exact refusal sentences get a calm, distinct box."""
    if answer == ACCESS_DENIED_MESSAGE:
        return "refused"
    if answer == NOT_FOUND_MESSAGE:
        return "not_found"
    return "answer"


def assistant_message(result: api.ApiResult) -> dict[str, Any]:
    """Turn an API result into a stored chat message."""
    if not result.ok:
        return {"role": "assistant", "kind": "error", "content": result.error or "Something went wrong."}
    data = result.data
    return {
        "role": "assistant",
        "kind": kind_of(data["answer"]),
        "content": data["answer"],
        "citations": data.get("citations", []),
        "tool_calls": data.get("tool_calls", []),
        "latency_ms": data.get("latency_ms"),
        "audit_id": result.audit_id,
    }


# --- rendering ---

def format_tool_call(call: dict[str, Any]) -> str:
    """get_vesting_status(as_of=2026-11-03); long values are shortened."""
    def short(value: Any) -> str:
        text = str(value)
        return text if len(text) <= MAX_ARG_CHARS else text[: MAX_ARG_CHARS - 1] + "…"
    args = ", ".join(f"{k}={short(v)}" for k, v in call.get("args", {}).items() if v is not None)
    return f"{call['name']}({args})"


def caption_for(message: dict[str, Any]) -> str:
    """Tools used · latency · audit id."""
    tools = ", ".join(format_tool_call(c) for c in message.get("tool_calls", [])) or "none"
    parts = [f"Tools: {tools}"]
    if message.get("latency_ms") is not None:
        parts.append(f"{message['latency_ms'] / 1000:.1f} s")
    if message.get("audit_id"):
        parts.append(f"audit {message['audit_id']}")
    return " · ".join(parts)


def render_message(message: dict[str, Any]) -> None:
    """One chat bubble: the text (styled by kind), then citations and the caption."""
    with st.chat_message(message["role"]):
        kind = message.get("kind", "answer")
        if kind == "refused":
            st.info(message["content"], icon=":material/lock:")
        elif kind == "not_found":
            st.info(message["content"], icon=":material/search_off:")
        elif kind == "error":
            st.error(message["content"], icon=":material/error:")
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant" and kind != "error":
            for c in message.get("citations", []):
                with st.expander(f"{c['doc_title']}, p. {c['page']} · {c['section']}", icon=":material/description:"):
                    st.markdown(f"> {c['snippet']}…")
            st.caption(caption_for(message))


def chat_panel(user: DemoUser, as_of: date) -> None:
    """Conversation so far, then the input; a new message is sent to /chat and the page reruns."""
    if not st.session_state.messages:
        st.caption(f"Ask {'about any grant, the cap table or the ESOP policy' if user.role == 'admin' else 'about your options or the ESOP policy'}.")
    for message in st.session_state.messages:
        render_message(message)

    prompt = st.chat_input(f"Ask as {user.name.split()[0]}…")
    if not prompt:
        return
    history = history_for_api(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "kind": "question", "content": prompt})
    render_message(st.session_state.messages[-1])
    with st.chat_message("assistant"), st.spinner("Thinking… (free-tier models can take up to a minute)"):
        result = api.chat(settings.api_url, user.user_id, prompt, history, as_of)
    st.session_state.messages.append(assistant_message(result))
    st.rerun()


def cap_table_panel(user: DemoUser) -> None:
    """Holders with issued and fully diluted %, plus the pool buckets (GET /captable)."""
    result = api.cap_table(settings.api_url, user.user_id)
    if not result.ok:
        st.error(result.error)
        return
    table = result.data
    pool = table["pool"]
    cols = st.columns(5)
    for col, (label, key) in zip(cols, [("Pool size", "pool_size"), ("Outstanding", "outstanding"),
                                        ("Exercised", "exercised"), ("Lapsed", "lapsed"),
                                        ("Unallocated", "unallocated")], strict=True):
        col.metric(label, f"{pool[key]:,}")
    frame = pd.DataFrame(table["rows"])[["name", "shares", "outstanding_options", "fully_diluted",
                                         "issued_pct", "fully_diluted_pct"]]
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "name": "Holder",
        "shares": st.column_config.NumberColumn("Shares", format="localized"),
        "outstanding_options": st.column_config.NumberColumn("Outstanding options", format="localized"),
        "fully_diluted": st.column_config.NumberColumn("Fully diluted", format="localized"),
        "issued_pct": st.column_config.NumberColumn("Issued %", format="%.2f%%"),
        "fully_diluted_pct": st.column_config.NumberColumn("Fully diluted %", format="%.2f%%"),
    })
    st.caption(f"Total issued {table['total_issued']:,} · fully diluted {table['total_fully_diluted']:,}. "
               "Lapsed options are shown for reference; they're back in the unallocated pool.")


def dilution_panel(user: DemoUser) -> None:
    """Form for POST /captable/simulate; the result is kept in session state across reruns."""
    with st.form("dilution"):
        left, right = st.columns(2)
        new_shares = left.number_input("New shares", min_value=1, value=2_000_000, step=100_000)
        investor = right.text_input("Investor name", value="Horizon Capital", max_chars=200)
        submitted = st.form_submit_button("Simulate", type="primary")
    if submitted:
        with st.spinner("Simulating…"):
            st.session_state.dilution_result = api.simulate(settings.api_url, user.user_id, int(new_shares), investor.strip())
    result = st.session_state.dilution_result
    if result is None:
        return
    if not result.ok:
        st.error(result.error)
        return
    data = result.data
    st.caption(f"{data['new_shares']:,} new shares to {data['investor_name']}: fully diluted total "
               f"{data['total_fully_diluted_before']:,} → {data['total_fully_diluted_after']:,}")
    frame = pd.DataFrame(data["rows"])[["name", "shares", "issued_pct_before", "issued_pct_after",
                                        "fully_diluted_pct_before", "fully_diluted_pct_after"]]
    pct = {"format": "%.2f%%"}
    st.dataframe(frame, hide_index=True, width="stretch", column_config={
        "name": "Holder",
        "shares": st.column_config.NumberColumn("Shares", format="localized"),
        "issued_pct_before": st.column_config.NumberColumn("Issued % before", **pct),
        "issued_pct_after": st.column_config.NumberColumn("Issued % after", **pct),
        "fully_diluted_pct_before": st.column_config.NumberColumn("Fully diluted % before", **pct),
        "fully_diluted_pct_after": st.column_config.NumberColumn("Fully diluted % after", **pct),
    })


def audit_panel(user: DemoUser) -> None:
    """The latest 20 audit records (GET /audit)."""
    st.button("Refresh", icon=":material/refresh:")  # any click reruns the script, which refetches
    result = api.audit(settings.api_url, user.user_id)
    if not result.ok:
        st.error(result.error)
        return
    if not result.data:
        st.caption("No requests logged yet.")
        return
    frame = pd.DataFrame([{
        "time (UTC)": r["ts"][:19].replace("T", " "),
        "user": r["user_id"],
        "outcome": r["outcome"],
        "latency ms": r["latency_ms"],
        "question": r["question"],
        "tools": ", ".join(c["name"] for c in r["tool_calls"]),
        "chunks": len(r["chunk_ids"]),
        "answer": (r["answer"] or r["error"] or "")[:120],
        "audit id": r["id"],
    } for r in result.data])
    st.dataframe(frame, hide_index=True, width="stretch")
    with st.expander("Raw records (JSON)"):
        st.json(result.data, expanded=False)


def sidebar() -> tuple[DemoUser, date]:
    """User switcher and as-of date; changing the user clears the conversation."""
    with st.sidebar:
        st.title("Vestwise")
        st.selectbox("Signed in as", options=list(USERS_BY_ID), key="user_id", on_change=reset_conversation,
                     format_func=lambda uid: f"{USERS_BY_ID[uid].name} ({USERS_BY_ID[uid].role})")
        st.date_input("As of", key="as_of", help="Sent as `as_of` on every question (demo date 2026-10-03).")
        if st.button("Clear conversation", icon=":material/delete_sweep:"):
            reset_conversation()
        st.caption(f"Simulated login: sends `X-User-Id: {st.session_state.user_id}` to {settings.api_url}. "
                   "The API decides what this user may see.")
    return USERS_BY_ID[st.session_state.user_id], st.session_state.as_of


def main() -> None:
    """Page layout: sidebar, then chat (plus admin tabs for admins)."""
    st.set_page_config(page_title="Vestwise", page_icon=":material/savings:", layout="wide")
    init_state()
    user, as_of = sidebar()
    if user.role != "admin":
        chat_panel(user, as_of)
        return
    chat_tab, cap_tab, dilution_tab, audit_tab = st.tabs(
        ["Chat", "Cap table", "Dilution", "Audit log"], key="admin_tab", on_change="rerun")
    # Lazy tabs: only the open tab's code runs, so a chat message doesn't refetch the cap table.
    if chat_tab.open:
        with chat_tab:
            chat_panel(user, as_of)
    if cap_tab.open:
        with cap_tab:
            cap_table_panel(user)
    if dilution_tab.open:
        with dilution_tab:
            dilution_panel(user)
    if audit_tab.open:
        with audit_tab:
            audit_panel(user)


main()
