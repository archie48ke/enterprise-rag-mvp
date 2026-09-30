"""
AI Enterprise Knowledge Assistant — Streamlit UI.

Run with:
    streamlit run app.py
"""

import html
import sys
import time
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import suggest_follow_ups
from src.config import (
    GROQ_API_KEY,
    EMBEDDING_MODEL,
    INDEX_DIR,
    DOMAINS,
    DOMAIN_FOLDER,
    NO_ANSWER_MESSAGE,
)
from src.history import (
    export_markdown,
    get_conversation,
    load_conversations,
    new_conversation_id,
    save_conversation,
)
from src.retriever import get_embedding_model
from src.workflow import stream_workflow

st.set_page_config(
    page_title="AI Enterprise Knowledge Assistant",
    page_icon=":material/auto_awesome:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Presentation constants
# ---------------------------------------------------------------------------

def _svg(paths: str) -> str:
    return (
        '<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )


DOMAIN_STYLE = {
    "HR": {
        "rgb": "192,132,252",
        "label": "HR",
        "desc": "Leave, benefits and employee policies",
        "svg": _svg('<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
                    '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>'),
        "material": ":material/groups:",
    },
    "TECHNICAL": {
        "rgb": "96,165,250",
        "label": "Technical",
        "desc": "Deployment, setup and infrastructure",
        "svg": _svg('<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>'),
        "material": ":material/terminal:",
    },
    "PROJECTS": {
        "rgb": "244,114,182",
        "label": "Projects",
        "desc": "Scope, goals, status and teams",
        "svg": _svg('<path d="M4 20h16a2 2 0 0 0 2-2V8a2 2 0 0 0-2-2h-7.9a2 2 0 0 1-1.7-.9l-.8-1.2A2 2 0 0 0 '
                    '7.9 3H4a2 2 0 0 0-2 2v13c0 1.1.9 2 2 2Z"/>'),
        "material": ":material/folder:",
    },
    "GENERAL": {
        "rgb": "52,211,153",
        "label": "General",
        "desc": "Working hours, offices and holidays",
        "svg": _svg('<rect x="4" y="2" width="16" height="20" rx="2"/><path d="M9 22v-4h6v4"/>'
                    '<path d="M8 6h.01M12 6h.01M16 6h.01M8 10h.01M12 10h.01M16 10h.01M8 14h.01M12 14h.01M16 14h.01"/>'),
        "material": ":material/apartment:",
    },
}

EXAMPLE_QUESTIONS = [
    ("HR", "How many casual leaves do I get?"),
    ("TECHNICAL", "How do I deploy Project Alpha?"),
    ("PROJECTS", "What is Project Alpha?"),
    ("GENERAL", "What are the company working hours?"),
]

ICONS = {
    "activity": _svg('<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>'),
    "cpu": _svg('<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/>'
                '<path d="M15 2v2M15 20v2M2 15h2M2 9h2M20 15h2M20 9h2M9 2v2M9 20v2"/>'),
    "database": _svg('<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14a9 3 0 0 0 18 0V5"/>'
                     '<path d="M3 12a9 3 0 0 0 18 0"/>'),
    "zap": _svg('<path d="M13 2 3 14h9l-1 8 10-12h-9l1-8z"/>'),
    "history": _svg('<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/><path d="M3 3v5h5"/>'
                    '<path d="M12 7v5l4 2"/>'),
}

MAX_HISTORY_ITEMS = 5

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/auto_awesome:"

BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg: #07070b;
    --line: rgba(255, 255, 255, 0.09);
    --line-strong: rgba(192, 132, 252, 0.45);
    --text: #ececf1;
    --muted: #9b98a8;
    --violet-soft: #c084fc;
    --glass: linear-gradient(145deg, rgba(255,255,255,.085), rgba(255,255,255,.02));
    --glass-border: rgba(255, 255, 255, 0.13);
    --glass-shadow: inset 0 1px 0 rgba(255,255,255,.14), 0 10px 34px rgba(0,0,0,.38);
}

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}
.mono { font-family: 'JetBrains Mono', ui-monospace, monospace !important; }
/* keep Streamlit's icon font — the global font override above must not replace it */
[data-testid="stIconMaterial"], span[class*="material-symbols"] { font-family: 'Material Symbols Rounded' !important; }

/* ---------- Aurora background ---------- */
.stApp { background: var(--bg); }
.stApp::before {
    content: ""; position: fixed; inset: -25%; z-index: 0; pointer-events: none;
    background:
        radial-gradient(38% 32% at 52% 30%, rgba(168, 85, 247, .55), transparent 70%),
        radial-gradient(26% 30% at 40% 62%, rgba(99, 102, 241, .38), transparent 70%),
        radial-gradient(22% 24% at 64% 55%, rgba(217, 70, 239, .30), transparent 70%);
    filter: blur(40px);
    opacity: var(--aurora-opacity, 1);
    animation: aurora 22s ease-in-out infinite alternate;
}
@keyframes aurora {
    0%   { transform: translate3d(0, 0, 0) rotate(0deg) scale(1); }
    50%  { transform: translate3d(3%, -2%, 0) rotate(8deg) scale(1.06); }
    100% { transform: translate3d(-3%, 2%, 0) rotate(-6deg) scale(1); }
}
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: transparent; position: relative; z-index: 1; }
header[data-testid="stHeader"] { background: transparent; }
[data-testid="stBottom"], [data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stBottom"] > div { background: linear-gradient(180deg, transparent, rgba(7,7,11,.94) 40%) !important; }

.block-container { max-width: 860px; padding-top: 2rem; padding-bottom: 7rem; }

/* ---------- Sidebar: fixed desktop panel, fits one screen ---------- */
[data-testid="stSidebar"] {
    background: rgba(10, 9, 16, 0.78) !important;
    backdrop-filter: blur(18px);
    border-right: 1px solid var(--line);
    min-width: 290px !important; max-width: 290px !important; width: 290px !important;
}
[data-testid="stSidebarContent"] { overflow: hidden !important; padding: 0 !important; }
[data-testid="stSidebarUserContent"] { width: 100% !important; box-sizing: border-box; padding: 1.4rem 1.1rem 1rem 1.1rem !important; }
[data-testid="stSidebarResizeHandle"] { display: none; }
@media (max-height: 760px) {
    .landing { padding-top: 0; margin-bottom: 1rem; }
    .landing h1 { font-size: 2.3rem; margin-bottom: .6rem; }
    .landing p { font-size: .92rem; }
    .eyebrow { margin-bottom: .7rem; }
    .section-label { margin: .9rem 0 .55rem 0; }
    .kpi-card { min-height: 118px; padding: .8rem .9rem; }
    .kpi-name { margin-top: .5rem; }
    .block-container { padding-top: .8rem; }
}
@media (min-width: 992px) {
    [data-testid="stSidebarHeader"] { height: 0 !important; min-height: 0 !important; padding: 0 !important; overflow: hidden; }
    [data-testid="stSidebarCollapseButton"], [data-testid="stExpandSidebarButton"] { display: none !important; }
}

.brand { display: flex; align-items: center; gap: .75rem; padding: .1rem .1rem .5rem .1rem; }
.orb {
    width: 36px; height: 36px; border-radius: 50%; flex: none;
    background: radial-gradient(circle at 35% 30%, #f5d0fe, #a855f7 45%, #4c1d95 80%);
    box-shadow: 0 0 22px rgba(168, 85, 247, .7);
}
.brand-name { font-weight: 600; font-size: 1rem; color: var(--text); line-height: 1.2; }
.brand-sub { font-size: .74rem; color: var(--muted); }

.glass {
    background: var(--glass); border: 1px solid var(--glass-border); box-shadow: var(--glass-shadow);
    backdrop-filter: blur(18px) saturate(140%); -webkit-backdrop-filter: blur(18px) saturate(140%);
}
.sb-card { border-radius: 14px; padding: .8rem .9rem; margin-bottom: .5rem; }
.sb-title { font-size: .66rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin-bottom: .45rem; }
.sb-row { display: flex; align-items: center; justify-content: space-between; gap: .6rem; font-size: .84rem; color: var(--text); padding: .22rem 0; }
.sb-row .left { display: flex; align-items: center; gap: .55rem; white-space: nowrap; }
.sb-row .right { color: var(--muted); font-size: .75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex: none; }
.dot.ok { background: #34d399; box-shadow: 0 0 10px rgba(52,211,153,.8); }
.dot.bad { background: #f87171; box-shadow: 0 0 10px rgba(248,113,113,.8); }
.sb-title { display: flex; align-items: center; gap: .45rem; }
.sb-title svg { width: 13px; height: 13px; color: var(--violet-soft); }
.row-icon { display: inline-flex; width: 24px; height: 24px; border-radius: 7px; align-items: center; justify-content: center;
            background: rgba(255,255,255,.05); border: 1px solid var(--line); color: #d8d4e6; }
.row-icon svg { width: 13px; height: 13px; }
.pill { display: inline-flex; align-items: center; gap: .35rem; font-size: .7rem; padding: .12rem .5rem; border-radius: 999px; }
.pill.ok { color: #6ee7b7; background: rgba(52,211,153,.10); border: 1px solid rgba(52,211,153,.25); }
.pill.bad { color: #fca5a5; background: rgba(248,113,113,.10); border: 1px solid rgba(248,113,113,.25); }
.history-title { margin: .4rem 0 0 .15rem; }
.history-title .count { margin-left: auto; font-size: .66rem; color: var(--muted); padding: .05rem .45rem;
                        border-radius: 999px; border: 1px solid var(--line); letter-spacing: 0; }
.history-empty { font-size: .78rem; color: var(--muted); padding: 0 .2rem; }
[class*="st-key-h_"] button { justify-content: flex-start !important; min-height: 2.1rem; padding: .25rem .7rem !important;
                              font-size: .82rem !important; border-radius: 10px !important; background: rgba(255,255,255,.02) !important; }
[class*="st-key-h_"] button > div { min-width: 0; justify-content: flex-start !important; width: 100%; }
[class*="st-key-h_"] button [data-testid="stIconMaterial"] { flex: none; }
[class*="st-key-h_"] button [data-testid="stMarkdownContainer"] { min-width: 0; overflow: hidden; }
[class*="st-key-h_"] button p { white-space: nowrap !important; overflow: hidden; text-overflow: ellipsis; text-align: left; }
[class*="st-key-h_"] [data-testid="stIconMaterial"] { font-size: 1rem; color: var(--muted); }
[class*="st-key-export_"] button { font-size: .8rem !important; padding: .35rem .5rem !important; min-height: 2.3rem; }
[class*="st-key-export_"] button p { text-align: center; white-space: nowrap; }

/* ---------- Buttons ---------- */
.stButton > button, .stDownloadButton > button {
    border-radius: 12px; border: 1px solid var(--glass-border); background: var(--glass);
    color: var(--text); transition: all .2s ease; backdrop-filter: blur(10px);
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--line-strong); background: rgba(168,85,247,.12); color: #fff;
    box-shadow: 0 0 24px -6px rgba(168,85,247,.6);
}
.stButton > button p { white-space: normal; text-align: left; line-height: 1.35; }
.st-key-new_chat button {
    background: linear-gradient(135deg, rgba(168,85,247,.30), rgba(99,102,241,.22)) !important;
    border: 1px solid var(--line-strong) !important; font-weight: 600;
}
.st-key-new_chat button p { text-align: center; }

/* ---------- Landing ---------- */
.landing { text-align: center; padding-top: 4vh; margin-bottom: 1.4rem; }
.eyebrow {
    display: inline-flex; align-items: center; gap: .5rem; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase;
    color: var(--violet-soft); padding: .35rem .9rem; border-radius: 999px;
    border: 1px solid rgba(192,132,252,.25); background: rgba(168,85,247,.08); margin-bottom: 1.2rem;
}
.eyebrow .pulse { width: 6px; height: 6px; border-radius: 50%; background: #c084fc; box-shadow: 0 0 10px #c084fc; }
.landing h1 {
    font-size: 2.8rem; font-weight: 400; letter-spacing: -0.035em; line-height: 1.12;
    color: #f4f3f8; margin: 0 0 .8rem 0; padding: 0;
}
.landing h1 .fade { background: linear-gradient(90deg, #f4f3f8, #c4b5fd); -webkit-background-clip: text; background-clip: text; color: transparent; }
.landing h1 .q { color: var(--violet-soft); }
.landing p {
    color: var(--muted); font-size: .98rem; max-width: 540px; margin: 0 auto; line-height: 1.6;
    text-align: justify; text-align-last: center; hyphens: auto;
}

/* ---------- Chat input: glowing capsule that grows downward ---------- */
[data-testid="stChatInput"] {
    position: relative;
    border-radius: 26px !important;
    background: rgba(16, 12, 26, .78) !important;
    border: 1px solid var(--line-strong) !important;
    box-shadow: 0 0 0 1px rgba(168,85,247,.12), 0 0 48px -10px rgba(168,85,247,.6), inset 0 1px 0 rgba(255,255,255,.06) !important;
    backdrop-filter: blur(14px);
    padding: 0 !important;
    transition: border-color .2s ease, box-shadow .2s ease;
}
[data-testid="stChatInput"]:focus-within {
    border-color: rgba(216,180,254,.75) !important;
    box-shadow: 0 0 0 3px rgba(168,85,247,.18), 0 0 56px -8px rgba(168,85,247,.75), inset 0 1px 0 rgba(255,255,255,.08) !important;
}
/* leading sparkle icon */
[data-testid="stChatInput"]::before {
    content: ""; position: absolute; left: 20px; top: 18px; width: 20px; height: 20px; pointer-events: none;
    background: no-repeat center / contain url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23c084fc' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z'/%3E%3Cpath d='M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z'/%3E%3C/svg%3E");
}
[data-testid="stChatInput"] > div { background: transparent !important; border: none !important; padding: 12px 12px 12px 52px !important; }
/* Streamlit stacks the send button under long text; keep it on the right instead. */
[data-testid="stChatInput"] > div > div { flex-direction: row !important; flex-wrap: nowrap !important; align-items: flex-end !important; gap: 0 !important; }
[data-testid="stChatInput"] > div > div > div:first-child { flex: 1 1 auto !important; min-width: 0; }
[data-testid="stChatInput"] > div > div > div:last-child { flex: 0 0 auto !important; width: auto !important; margin-left: .6rem; }
[data-testid="stChatInput"] textarea {
    background: transparent !important; border: none !important; color: var(--text) !important;
    font-size: 1rem !important; line-height: 1.5 !important; max-height: 7.5rem !important; overflow-y: auto !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #8d8a9b !important; }
[data-testid="stChatInputSubmitButton"] {
    width: 34px !important; height: 34px !important; border-radius: 50% !important;
    background: linear-gradient(135deg, #a855f7, #6366f1) !important; color: #fff !important;
    box-shadow: 0 0 18px rgba(168,85,247,.55);
}
[data-testid="stChatInputSubmitButton"]:disabled { opacity: .45; box-shadow: none; }
.st-key-chat_home { max-width: 660px; margin: 0 auto; }

[class*="st-key-ex_"] button { border-radius: 999px !important; font-size: .86rem !important; padding: .5rem 1rem !important; min-height: 2.6rem; }
[class*="st-key-ex_"] button p { text-align: center; }

.section-label {
    font-size: .68rem; font-weight: 600; letter-spacing: .14em; text-transform: uppercase;
    color: var(--muted); margin: 1.6rem 0 .8rem 0; text-align: center;
}

/* ---------- Glassmorphism KPI cards ---------- */
.kpi-card {
    position: relative; overflow: hidden; border-radius: 18px; padding: 1rem 1rem .95rem 1rem;
    min-height: 128px; box-sizing: border-box; display: flex; flex-direction: column;
    transition: transform .25s ease, box-shadow .25s ease, border-color .25s ease;
}
.kpi-card::before {
    content: ""; position: absolute; width: 120px; height: 120px; right: -40px; top: -50px; border-radius: 50%;
    background: radial-gradient(circle, rgba(var(--rgb), .45), transparent 70%); filter: blur(8px); pointer-events: none;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(var(--rgb), .5); box-shadow: inset 0 1px 0 rgba(255,255,255,.18), 0 14px 40px -10px rgba(var(--rgb), .55); }
.kpi-icon {
    width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center;
    color: rgb(var(--rgb)); background: rgba(var(--rgb), .14); border: 1px solid rgba(var(--rgb), .32);
}
.kpi-name { font-weight: 600; margin-top: .7rem; color: var(--text); font-size: .95rem; }
.kpi-desc { font-size: .78rem; color: var(--muted); margin-top: .15rem; line-height: 1.45; flex: 1; }


/* ---------- Chat messages ---------- */
[data-testid="stChatMessage"] {
    background: var(--glass) !important; border: 1px solid var(--glass-border); border-radius: 18px;
    box-shadow: var(--glass-shadow); backdrop-filter: blur(16px); padding: 1rem 1.2rem; margin-bottom: .9rem;
}
/* Icon avatars are all "stChatMessageAvatarCustom", so user turns carry a .msg-user marker. */
[data-testid="stChatMessage"]:has(.msg-user) {
    background: linear-gradient(135deg, rgba(168,85,247,.18), rgba(99,102,241,.10)) !important;
    border-color: rgba(192,132,252,.28);
}
/* Paragraphs are justified; list items stay left-aligned because long tokens
   (URLs, ENV_VARS) would otherwise open wide gaps in short lines. */
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { text-align: justify; hyphens: auto; line-height: 1.65; }
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li p { text-align: left; line-height: 1.65; }
[data-testid="stChatMessageAvatarCustom"] {
    border-radius: 10px; background: linear-gradient(135deg, #a855f7, #6366f1) !important; color: #fff !important;
    box-shadow: 0 0 16px rgba(168,85,247,.55);
}
[data-testid="stChatMessage"]:has(.msg-user) [data-testid="stChatMessageAvatarCustom"] {
    background: rgba(168,85,247,.22) !important; color: #e9d5ff !important; box-shadow: none;
}
.msg-user { display: none; }
.abstain {
    padding: .85rem 1rem; border-radius: 12px; color: #fde68a; text-align: justify;
    border: 1px solid rgba(251,191,36,.3); background: rgba(251,191,36,.08);
}

/* related questions */
.related { font-size: .68rem; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); font-weight: 600; margin: 1.2rem 0 0 0; padding-bottom: .3rem; }
[data-testid="stMarkdownContainer"]:has(> .related) { margin-bottom: 0 !important; }
[class*="st-key-fu_"] button {
    justify-content: flex-start !important; border-radius: 12px !important; min-height: 2.5rem;
    padding: .45rem .9rem !important; font-size: .88rem !important; color: #ddd6fe !important;
}
[class*="st-key-fu_"] button > div { justify-content: flex-start !important; width: 100%; }
[class*="st-key-fu_"] button p { text-align: left; }
[data-testid="stChatMessage"] [data-testid="stVerticalBlock"] { gap: .45rem; }
</style>
"""

LANDING_CSS = "<style>.block-container { padding-bottom: 1.5rem; }</style>"
# In chat mode the aurora dims so long answers stay readable.
CHAT_MODE_CSS = "<style>.stApp { --aurora-opacity: .35; }</style>"


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model…")
def load_embedding_model():
    """Load Sentence Transformers once per server process."""
    get_embedding_model()
    return True


def indexes_exist() -> bool:
    return any((INDEX_DIR / folder / "index.faiss").exists() for folder in DOMAIN_FOLDER.values())


def friendly_error(exc: Exception) -> str:
    """Translate Groq / runtime exceptions into a message a user can act on."""
    name = type(exc).__name__
    if name == "AuthenticationError":
        return "The Groq API key was rejected. Check GROQ_API_KEY in your .env or Streamlit secrets."
    if name == "RateLimitError":
        return "The Groq API rate limit was reached. Please wait a few seconds and try again."
    if name in ("APIConnectionError", "APITimeoutError"):
        return "Could not reach the Groq API. Check your internet connection and try again."
    return f"Something went wrong while answering your question ({name}). Please try again."


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def status_row(icon: str, label: str, ok: bool, detail: str) -> str:
    state = "ok" if ok else "bad"
    return (
        f'<div class="sb-row"><span class="left"><span class="row-icon">{icon}</span>{label}</span>'
        f'<span class="pill {state}"><span class="dot {state}"></span>{detail}</span></div>'
    )


def short_title(title: str, limit: int = 26) -> str:
    return title if len(title) <= limit else title[: limit - 1].rstrip() + "…"


def start_new_chat():
    st.session_state.messages = []
    st.session_state.conversation_id = new_conversation_id()


def render_sidebar(model_ready: bool):
    has_indexes = indexes_exist()
    conversations = load_conversations()
    current_id = st.session_state.conversation_id

    with st.sidebar:
        st.markdown(
            '<div class="brand"><div class="orb"></div><div>'
            '<div class="brand-name">Knowledge Assistant</div>'
            '<div class="brand-sub">Enterprise RAG</div></div></div>',
            unsafe_allow_html=True,
        )

        if st.button("New chat", key="new_chat", icon=":material/add:", width="stretch"):
            start_new_chat()
            st.rerun()

        st.markdown(
            f'<div class="sb-card glass"><div class="sb-title">{ICONS["activity"]}System status</div>'
            + status_row(ICONS["cpu"], "Embeddings", model_ready, "ready" if model_ready else "error")
            + status_row(ICONS["database"], "FAISS indexes", has_indexes, "loaded" if has_indexes else "missing")
            + status_row(ICONS["zap"], "Groq API", bool(GROQ_API_KEY), "online" if GROQ_API_KEY else "no key")
            + "</div>",
            unsafe_allow_html=True,
        )
        if not has_indexes:
            st.caption("Run `python scripts/build_indexes.py`")

        st.markdown(
            f'<div class="sb-title history-title">{ICONS["history"]}Recent chats'
            f'<span class="count">{len(conversations)}</span></div>',
            unsafe_allow_html=True,
        )
        if not conversations:
            st.markdown('<div class="history-empty">Your conversations will appear here.</div>', unsafe_allow_html=True)
        for conv in conversations[:MAX_HISTORY_ITEMS]:
            if st.button(
                short_title(conv["title"]),
                key=f"h_{conv['id']}",
                icon=":material/chat_bubble:",
                width="stretch",
            ):
                st.session_state.messages = conv["messages"]
                st.session_state.conversation_id = conv["id"]
                st.rerun()
        if len(conversations) > MAX_HISTORY_ITEMS:
            st.markdown(
                f'<div class="history-empty">+{len(conversations) - MAX_HISTORY_ITEMS} older, included in All chats export</div>',
                unsafe_allow_html=True,
            )
        # Highlight the conversation that is currently open.
        st.markdown(
            f"<style>.st-key-h_{current_id} button {{ border-color: var(--line-strong) !important;"
            f" background: rgba(168,85,247,.16) !important; color: #fff !important; }}</style>",
            unsafe_allow_html=True,
        )

        current = get_conversation(current_id)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "This chat",
                data=export_markdown([current]) if current else "",
                file_name="conversation.md",
                mime="text/markdown",
                icon=":material/download:",
                width="stretch",
                disabled=current is None,
                key="export_one",
            )
        with col2:
            st.download_button(
                "All chats",
                data=export_markdown(conversations) if conversations else "",
                file_name="all_conversations.md",
                mime="text/markdown",
                icon=":material/library_books:",
                width="stretch",
                disabled=not conversations,
                key="export_all",
            )


# ---------------------------------------------------------------------------
# Landing screen
# ---------------------------------------------------------------------------

def render_landing_top():
    st.markdown(
        """
<div class="landing">
  <div class="eyebrow"><span class="pulse"></span>Enterprise Knowledge Assistant</div>
  <h1>What's on your mind <span class="fade">today</span><span class="q">?</span></h1>
  <p>Ask about HR policies, deployments, projects or company information. Every answer is
  grounded in internal company documents.</p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_landing_bottom():
    st.write("")
    cols = st.columns(2)
    for i, (domain, question) in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(question, key=f"ex_{i}", icon=DOMAIN_STYLE[domain]["material"], width="stretch"):
            st.session_state.pending_question = question
            st.rerun()

    st.markdown('<div class="section-label">Knowledge domains</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, domain in zip(cols, DOMAINS):
        style = DOMAIN_STYLE[domain]
        col.markdown(
            f"""
<div class="kpi-card glass" style="--rgb:{style['rgb']};">
  <div class="kpi-icon">{style['svg']}</div>
  <div class="kpi-name">{style['label']}</div>
  <div class="kpi-desc">{style['desc']}</div>
</div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Assistant message
# ---------------------------------------------------------------------------

def render_user(text: str):
    st.markdown('<span class="msg-user"></span>', unsafe_allow_html=True)
    st.markdown(text)


def fallback_suggestions(question: str) -> list:
    """When nothing was found, suggest the sample questions the documents can answer."""
    return [q for _, q in EXAMPLE_QUESTIONS if q != question][:3]


def render_assistant(msg: dict, index: int, is_latest: bool, clear_stale: bool = False):
    """Render one assistant turn: the answer only, plus related questions on the latest turn."""
    if msg.get("error"):
        st.error(msg["error"], icon=":material/error:")
        return

    if NO_ANSWER_MESSAGE in msg["answer"] and not msg.get("citations"):
        st.markdown(f'<div class="abstain">{html.escape(msg["answer"])}</div>', unsafe_allow_html=True)
    else:
        st.markdown(msg["answer"])

    follow_ups = msg.get("follow_ups") or []
    if is_latest and follow_ups:
        st.markdown('<div class="related">Related questions</div>', unsafe_allow_html=True)
        for i, q in enumerate(follow_ups):
            if st.button(q, key=f"fu_{index}_{i}", icon=":material/subdirectory_arrow_right:", width="stretch"):
                st.session_state.pending_question = q
                st.rerun()
    elif clear_stale:
        # This turn showed related questions on the previous run. Blank those slots
        # now, otherwise Streamlit leaves them faded on screen while the next answer loads.
        for _ in range(len(follow_ups) + 1):
            st.empty()


# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

def answer_question(question: str) -> dict:
    """Run the LangGraph pipeline with live per-node progress, then suggest follow-ups."""
    record = {"role": "assistant", "domain": "GENERAL", "answer": "", "citations": [], "follow_ups": []}
    chunks = []

    with st.status("Routing your question…", expanded=True) as status:
        start = time.perf_counter()
        try:
            for node, state in stream_workflow(question):
                if node == "manager":
                    record["domain"] = state["domain"]
                    st.write(f":material/check_circle: **Manager Agent** routed the question to **{state['domain']}**")
                    status.update(label=f"Searching the {state['domain']} index…", expanded=True)
                elif node == "retrieval":
                    chunks = state.get("retrieved_chunks", [])
                    st.write(f":material/check_circle: **FAISS Retriever** found {len(chunks)} relevant chunk(s)")
                    status.update(label="Generating a grounded answer…", expanded=True)
                elif node == "answer":
                    record["answer"] = state["answer"]
                    record["citations"] = [c.model_dump() for c in state.get("citations", [])]
                    st.write(":material/check_circle: **Answer Agent** generated the response")

            if chunks and NO_ANSWER_MESSAGE not in record["answer"]:
                status.update(label="Suggesting related questions…", expanded=True)
                record["follow_ups"] = suggest_follow_ups(question, record["answer"], chunks)
            if not record["follow_ups"]:
                record["follow_ups"] = fallback_suggestions(question)
        except Exception as exc:
            print(f"[app] Pipeline error: {exc!r}")
            status.update(label="Something went wrong", state="error", expanded=False)
            return {"role": "assistant", "error": friendly_error(exc)}

        status.update(label=f"Done in {time.perf_counter() - start:.2f}s", state="complete", expanded=False)

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if "messages" not in st.session_state:
        start_new_chat()

    st.markdown(BASE_CSS, unsafe_allow_html=True)

    model_ready = False
    try:
        model_ready = load_embedding_model()
    except Exception as exc:
        st.error(f"Could not load the embedding model '{EMBEDDING_MODEL}': {exc}", icon=":material/error:")

    render_sidebar(model_ready)

    if not GROQ_API_KEY:
        render_landing_top()
        st.error(
            "**GROQ_API_KEY is not configured.** Copy `.env.example` to `.env` "
            "and add your Groq API key, then restart the app.",
            icon=":material/key_off:",
        )
        return

    if not indexes_exist():
        st.warning(
            "No FAISS indexes were found. Add PDFs under `data/<department>/` "
            "and run `python scripts/build_indexes.py` before asking questions.",
            icon=":material/warning:",
        )

    pending = st.session_state.pop("pending_question", None)

    if not st.session_state.messages:
        # Landing: the chat box sits inline in the centre of the page.
        st.markdown(LANDING_CSS, unsafe_allow_html=True)
        landing = st.empty()
        with landing.container():
            render_landing_top()
            question = st.chat_input("Ask anything about your company…", key="chat_home")
            render_landing_bottom()
        question = pending or question
        if not question:
            return
        landing.empty()  # clear the landing screen immediately once a question arrives
    else:
        # Conversation: the chat box is pinned to the bottom of the page.
        typed = st.chat_input("Ask a follow-up question…", key="chat_main")
        question = pending or typed

    st.markdown(CHAT_MODE_CSS, unsafe_allow_html=True)

    messages = st.session_state.messages
    for i, msg in enumerate(messages):
        with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR):
            if msg["role"] == "user":
                render_user(msg["content"])
            else:
                is_last = i == len(messages) - 1
                render_assistant(msg, i, is_latest=is_last and not question, clear_stale=is_last and bool(question))

    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user", avatar=USER_AVATAR):
            render_user(question)

        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            record = answer_question(question)
            render_assistant(record, len(messages), is_latest=False)

        messages.append(record)
        save_conversation(st.session_state.conversation_id, messages)
        # Rerun so the related-question buttons, history and exports include this turn.
        st.rerun()


if __name__ == "__main__":
    main()
