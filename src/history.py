"""
Local conversation history.

Conversations are stored as JSON in history/conversations.json so they survive
page reloads and app restarts on the machine running the app. This is a simple
single-user store: there is no authentication, so every user of the same
deployment sees the same history.
"""

import json
import uuid
from datetime import datetime
from typing import List, Optional

from src.config import PROJECT_ROOT

HISTORY_FILE = PROJECT_ROOT / "history" / "conversations.json"


def load_conversations() -> List[dict]:
    """All saved conversations, newest first. Returns [] if the file is missing or unreadable."""
    try:
        data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return sorted(data, key=lambda c: c.get("updated_at", ""), reverse=True)


def _write(conversations: List[dict]) -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(HISTORY_FILE)  # atomic swap so a crash never leaves a half-written file


def new_conversation_id() -> str:
    return uuid.uuid4().hex[:12]


def save_conversation(conv_id: str, messages: List[dict]) -> None:
    """Create or update a conversation. The title is the first question asked."""
    if not messages:
        return
    now = datetime.now().isoformat(timespec="seconds")
    conversations = load_conversations()
    existing = next((c for c in conversations if c["id"] == conv_id), None)
    if existing:
        existing["messages"] = messages
        existing["updated_at"] = now
    else:
        first_question = next((m["content"] for m in messages if m["role"] == "user"), "New chat")
        conversations.append(
            {"id": conv_id, "title": first_question, "created_at": now, "updated_at": now, "messages": messages}
        )
    _write(conversations)


def get_conversation(conv_id: str) -> Optional[dict]:
    return next((c for c in load_conversations() if c["id"] == conv_id), None)


def delete_conversation(conv_id: str) -> None:
    _write([c for c in load_conversations() if c["id"] != conv_id])


def conversation_to_markdown(conv: dict) -> str:
    lines = [f"## {conv['title']}", "", f"_Started {conv['created_at'].replace('T', ' ')}_", ""]
    for msg in conv["messages"]:
        if msg["role"] == "user":
            lines += [f"**Question:** {msg['content']}", ""]
        elif msg.get("error"):
            lines += [f"**Error:** {msg['error']}", ""]
        else:
            lines += [f"**Answer:** {msg['answer']}", ""]
    return "\n".join(lines)


def export_markdown(conversations: List[dict]) -> str:
    header = "# AI Enterprise Knowledge Assistant — Conversations\n\n"
    return header + "\n---\n\n".join(conversation_to_markdown(c) for c in conversations)
