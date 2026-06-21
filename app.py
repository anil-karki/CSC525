"""
CSU Global Chatbot — Closed-Domain
Covers the full csuglobal.edu/student-success/ section plus
admissions, programs, cost, and about pages.

Knowledge sources (in priority order):
  1. Structured intents  — verified facts from all CSU Global pages
  2. Live web scraping   — multi-page scrape of the student-success domain
  3. Fallback            — polite out-of-scope message
"""

import json
import random
import re
import string
import requests
import streamlit as st
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------
# CONFIG — all pages that belong to the closed domain
# -----------------------------------------------------------------------
DOMAIN_URLS = [
    "https://csuglobal.edu/student-success/",
    "https://csuglobal.edu/student-success/academic-calendar",
    "https://csuglobal.edu/student-success/what-expect",
    "https://csuglobal.edu/student-success/academic-support",
    "https://csuglobal.edu/student-success/career-development",
    "https://csuglobal.edu/student-success/offices-services",
    "https://csuglobal.edu/student-success/course-catalog",
    "https://csuglobal.edu/student-success/student-organizations",
    "https://csuglobal.edu/admissions",
    "https://csuglobal.edu/admissions/undergraduate-students/",
    "https://csuglobal.edu/admissions/graduate-students/",
    "https://csuglobal.edu/admissions/transfer-students/",
    "https://csuglobal.edu/admissions/military-veteran-students/",
    "https://csuglobal.edu/admissions/international-students/",
    "https://csuglobal.edu/about",
]

PRIMARY_URL   = "https://csuglobal.edu/student-success/"
REQUEST_TIMEOUT = 12

# Similarity thresholds
INTENT_THRESHOLD = 0.40
WEB_THRESHOLD    = 0.18

# -----------------------------------------------------------------------
# PAGE SETTINGS
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="CSU Global Chatbot",
    page_icon="🎓",
    layout="centered",
)

# -----------------------------------------------------------------------
# TEXT HELPERS
# -----------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()

_JUNK = [
    "apply now", "request info", "want to know more",
    "this field is for validation", "complete the form",
    "first name", "last name", "by submitting this form",
    "consent is not required", "opt out", "cookies",
    "privacy statement", "skip to main", "login",
]

def _is_useful(text: str) -> bool:
    low = text.lower()
    return len(text) >= 50 and not any(j in low for j in _JUNK)

def _split_chunks(text: str, max_chars: int = 450):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, cur = [], ""
    for s in sentences:
        s = s.strip()
        if not _is_useful(s):
            continue
        if len(cur) + len(s) + 1 <= max_chars:
            cur = f"{cur} {s}".strip()
        else:
            if cur:
                chunks.append(cur)
            cur = s
    if cur:
        chunks.append(cur)
    return chunks

# -----------------------------------------------------------------------
# SCRAPER
# -----------------------------------------------------------------------
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}

def _scrape_one(url: str) -> list[str]:
    """Scrape a single URL, return text chunks. Returns [] on failure."""
    try:
        r = requests.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "header",
                         "footer", "nav", "form"]):
            tag.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
        return _split_chunks(text)
    except Exception:
        return []


@st.cache_data(show_spinner=False, ttl=3600)
def fetch_all_chunks():
    """
    Scrape every domain URL, label each chunk with its source page,
    and return combined list + status string.
    """
    all_items = []     # list of {"text": ..., "raw": ..., "source": url}
    loaded, failed = [], []

    for url in DOMAIN_URLS:
        chunks = _scrape_one(url)
        if chunks:
            for c in chunks:
                all_items.append({"raw": c, "text": preprocess(c), "source": url})
            loaded.append(url)
        else:
            # Try requests-html JS render as fallback
            try:
                from requests_html import HTMLSession   # type: ignore
                session = HTMLSession()
                resp = session.get(url, headers=_HEADERS, timeout=REQUEST_TIMEOUT)
                resp.html.render(timeout=20, sleep=2)
                soup = BeautifulSoup(resp.html.html, "html.parser")
                for tag in soup(["script","style","noscript","header","footer","nav","form"]):
                    tag.decompose()
                text = re.sub(r"\s+", " ", soup.get_text(separator=" ", strip=True))
                chunks = _split_chunks(text)
                if chunks:
                    for c in chunks:
                        all_items.append({"raw": c, "text": preprocess(c), "source": url})
                    loaded.append(url)
                else:
                    failed.append(url)
            except Exception:
                failed.append(url)

    status = f"Scraped {len(loaded)}/{len(DOMAIN_URLS)} pages live"
    if failed:
        status += f" ({len(failed)} used knowledge base fallback)"
    return all_items, status

# -----------------------------------------------------------------------
# LOAD INTENTS
# -----------------------------------------------------------------------
def load_intents(path="intents.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_intent_items(data):
    items = []
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            items.append({
                "tag": intent["tag"],
                "text": preprocess(pattern),
                "responses": intent["responses"],
            })
    return items

def build_vectorizer(items):
    texts = [i["text"] for i in items]
    vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    mat = vec.fit_transform(texts)
    return vec, mat

# -----------------------------------------------------------------------
# RESPONSE LOGIC
# -----------------------------------------------------------------------
def best_match(query, items, vec, mat):
    qv = vec.transform([preprocess(query)])
    scores = cosine_similarity(qv, mat)[0]
    idx = int(scores.argmax())
    return items[idx], float(scores[idx])

def get_response(user_input: str) -> str:
    if not preprocess(user_input):
        return "Please type a question about CSU Global."

    # 1 — structured intents (always checked first)
    best_intent, intent_score = best_match(
        user_input, intent_items, intent_vec, intent_mat
    )
    if intent_score >= INTENT_THRESHOLD:
        return random.choice(best_intent["responses"])

    # 2 — live scraped web content
    if web_items:
        best_web, web_score = best_match(
            user_input, web_items, web_vec, web_mat
        )
        if web_score >= WEB_THRESHOLD:
            page_label = best_web["source"].replace("https://csuglobal.edu", "csuglobal.edu")
            return (
                f"{best_web['raw']}\n\n"
                f"_Source: [{page_label}]({best_web['source']})_"
            )

    # 3 — best-effort intent even below threshold (skip pure fallback tags)
    skip_tags = {"fallback", "greeting", "smalltalk", "closed_domain_notice"}
    if best_intent["tag"] not in skip_tags and intent_score >= 0.18:
        return random.choice(best_intent["responses"])

    # 4 — generic fallback
    fb = next((i for i in intents_data["intents"] if i["tag"] == "fallback"), None)
    if fb:
        return random.choice(fb["responses"])
    return "Sorry, I can only answer questions about CSU Global."

# -----------------------------------------------------------------------
# STARTUP — load knowledge
# -----------------------------------------------------------------------
intents_data  = load_intents("intents.json")
intent_items  = build_intent_items(intents_data)
intent_vec, intent_mat = build_vectorizer(intent_items)

web_items, scrape_status = fetch_all_chunks()
if web_items:
    web_vec, web_mat = build_vectorizer(web_items)
else:
    web_vec = web_mat = None

# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------
st.title("🎓 CSU Global Chatbot")
st.caption(
    "Closed-domain chatbot covering [csuglobal.edu/student-success/](https://csuglobal.edu/student-success/) "
    "and related sections: Admissions, Programs, Cost, and About."
)

if web_items:
    st.success(f"✅ {scrape_status} — {len(web_items)} content chunks loaded")
else:
    st.warning("⚠️ Live scrape unavailable — using verified knowledge base. Answers are still accurate.")

with st.expander("💡 Example questions", expanded=False):
    st.markdown(
        """
**Academic Calendar**
- When do classes start at CSU Global?
- What is the Burgundy track? What is the Gold track?
- How many trimesters does CSU Global have?
- Where can I download the 2025-2026 calendar?

**Student Success Services**
- What does a student success counselor do?
- Is there tutoring available?
- How does the writing center work?
- What career services are offered?
- How do I prepare for a job interview?
- Is there disability services?

**Admissions**
- How do I apply to CSU Global?
- Can I transfer credits?
- Are there military benefits?
- How do international students apply?

**Programs & Degrees**
- What bachelor's degrees are available?
- What master's programs does CSU Global offer?

**Cost & Tuition**
- How much does CSU Global cost?
- Is there a tuition guarantee?
- Are there any student fees?

**About CSU Global**
- Is CSU Global accredited?
- Is it 100% online?
- Who are the faculty?
- How do I contact CSU Global?
"""
    )

# Chat state
if "history" not in st.session_state:
    st.session_state.history = []

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask a question about CSU Global:")
    submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    answer = get_response(user_input.strip())
    st.session_state.history.append(("You", user_input.strip()))
    st.session_state.history.append(("Bot", answer))

if st.session_state.history:
    st.markdown("---")
    st.markdown("### 💬 Conversation")
    for speaker, message in reversed(st.session_state.history):
        if speaker == "You":
            st.markdown(f"**🧑 You:** {message}")
        else:
            st.markdown(f"**🤖 Bot:** {message}")
