
import os
import re
import html
import tempfile
import streamlit as st
from dotenv import load_dotenv

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

load_dotenv()

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Nexa AI | Video Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# CUSTOM UI DESIGN
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        --bg: #0b1020;
        --panel: #131b30;
        --panel2: #18223a;
        --purple: #8b5cf6;
        --blue: #60a5fa;
        --text: #f1f5f9;
        --muted: #a5b4cc;
        --border: rgba(148,163,184,0.17);
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 15% 0%, rgba(99,102,241,0.14), transparent 38%),
            radial-gradient(ellipse at 95% 15%, rgba(139,92,246,0.12), transparent 35%),
            #0b1020;
        color: var(--text);
    }

    [data-testid="stHeader"] {
        background: rgba(11,16,32,0.85);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #11182b, #0c1222);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 4px 0 25px 0;
    }

    .brand-icon {
        width: 45px;
        height: 45px;
        display: flex;
        justify-content: center;
        align-items: center;
        border-radius: 14px;
        background: linear-gradient(135deg, #8b5cf6, #3b82f6);
        font-size: 24px;
        box-shadow: 0 8px 24px rgba(139,92,246,0.28);
    }

    .brand-name {
        font-size: 21px;
        font-weight: 800;
        letter-spacing: -0.8px;
        color: white;
    }

    .brand-caption {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 2px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 35px 34px;
        border: 1px solid rgba(167,139,250,0.27);
        border-radius: 24px;
        background:
            radial-gradient(circle at 90% 10%, rgba(96,165,250,0.24), transparent 30%),
            linear-gradient(120deg, #171b3a 0%, #242052 50%, #152a4c 100%);
        box-shadow: 0 18px 50px rgba(0,0,0,0.22);
        margin: 6px 0 26px 0;
    }

    .hero-label {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 30px;
        border: 1px solid rgba(196,181,253,0.3);
        background: rgba(139,92,246,0.16);
        color: #ddd6fe;
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 16px;
    }

    .hero h1 {
        font-size: clamp(29px, 4vw, 43px);
        line-height: 1.17;
        font-weight: 800;
        letter-spacing: -1.6px;
        margin: 0 0 13px 0;
        color: #ffffff;
    }

    .hero h1 span {
        background: linear-gradient(90deg, #c4b5fd, #93c5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #c0cbe0;
        font-size: 14px;
        line-height: 1.8;
        max-width: 700px;
        margin: 0;
    }

    .section-title {
        font-size: 20px;
        font-weight: 750;
        letter-spacing: -0.5px;
        color: #f8fafc;
        margin: 10px 0 5px 0;
    }

    .section-subtitle {
        color: var(--muted);
        font-size: 13px;
        margin-bottom: 20px;
    }

    .feature-card {
        min-height: 180px;
        padding: 22px 20px;
        border: 1px solid var(--border);
        border-radius: 18px;
        background: linear-gradient(145deg, #172139, #11192c);
        box-shadow: 0 8px 26px rgba(0,0,0,0.12);
        transition: border-color 0.2s ease, transform 0.2s ease;
    }

    .feature-icon {
        width: 43px;
        height: 43px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        font-size: 21px;
        margin-bottom: 16px;
        background: rgba(139,92,246,0.15);
    }

    .feature-card h3 {
        color: #f8fafc;
        font-size: 16px;
        margin: 0 0 8px 0;
    }

    .feature-card p {
        color: #a8b5cc;
        font-size: 12px;
        line-height: 1.75;
        margin: 0;
    }

    .metric-card {
        padding: 18px;
        min-height: 112px;
        border-radius: 16px;
        background: linear-gradient(145deg, #18233b, #11192b);
        border: 1px solid var(--border);
    }

    .metric-label {
        color: #a5b4cc;
        font-size: 12px;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 24px;
        font-weight: 800;
        letter-spacing: -0.7px;
        overflow-wrap: anywhere;
    }

    .result-panel {
        background: linear-gradient(145deg, #172139, #11192b);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 22px;
        margin: 8px 0 16px 0;
    }

    .result-label {
        font-size: 11px;
        color: #c4b5fd;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .stButton > button {
        border-radius: 11px;
        font-weight: 650;
        min-height: 43px;
        transition: all 0.2s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(110deg, #7c3aed, #4f46e5);
        border: 1px solid rgba(196,181,253,0.4);
        color: white;
    }

    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(110deg, #8b5cf6, #6366f1);
        border-color: #c4b5fd;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        border: 1px solid rgba(167,139,250,0.35);
        background: #1b2440;
        color: #e9d5ff;
        font-weight: 600;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 7px;
        border-bottom: 1px solid var(--border);
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(24,34,58,0.7);
        border-radius: 10px 10px 0 0;
        padding: 10px 15px;
        color: #b7c4da;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(139,92,246,0.18) !important;
        color: #ddd6fe !important;
    }

    .stTextInput input, .stTextArea textarea {
        background: #111a2d;
        color: #f1f5f9;
        border: 1px solid #34415d;
        border-radius: 10px;
    }

    .stSelectbox [data-baseweb="select"] > div {
        background: #111a2d;
        border-color: #34415d;
        border-radius: 10px;
    }

    .footer {
        text-align: center;
        color: #75839c;
        font-size: 11px;
        padding: 28px 0 10px 0;
    }

    hr {
        border-color: var(--border);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

if "quick_query" not in st.session_state:
    st.session_state.quick_query = None


def generate_markdown_report(result):
    return f"""# {result.get('title', 'AI Video Summary')}

## Executive Summary
{result.get('summary', 'N/A')}

## Action Items
{result.get('action_items', 'N/A')}

## Key Decisions
{result.get('key_decisions', 'N/A')}

## Open Questions
{result.get('open_questions', 'N/A')}

## Full Transcript
{result.get('transcript', 'N/A')}
"""


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-icon">✦</div>
            <div>
                <div class="brand-name">Nexa AI</div>
                <div class="brand-caption">VIDEO INTELLIGENCE STUDIO</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🎬 Create a new analysis")
    st.caption("Turn any supported recording into searchable insights.")

    input_mode = st.radio(
        "Choose your source",
        ["YouTube URL", "Local File Upload"],
    )

    source_path = ""
    uploaded_file = None

    if input_mode == "YouTube URL":
        yt_url = st.text_input(
            "YouTube link",
            placeholder="Paste your video URL here...",
        )
        if yt_url:
            source_path = yt_url.strip()

    else:
        uploaded_file = st.file_uploader(
            "Upload a recording",
            type=[
                "mp4", "mp3", "wav", "m4a",
                "mov", "webm", "ogg", "aac",
            ],
            help="Choose a supported audio or video file.",
        )

    st.markdown("---")
    st.markdown("### ⚙️ Analysis settings")

    language_choice = st.selectbox(
        "Spoken language",
        ["English", "Hinglish"],
    ).lower()

    run_button = st.button(
        "✦  Generate AI Insights",
        type="primary",
        use_container_width=True,
    )

    if st.session_state.pipeline_result is not None:
        if st.button("＋  Start a new analysis", use_container_width=True):
            st.session_state.pipeline_result = None
            st.session_state.chat_messages = []
            st.session_state.quick_query = None
            st.rerun()

    with st.expander("How it works"):
        st.markdown(
            """
            1. **Audio processing** — prepares your recording.
            2. **Transcription** — converts speech to text.
            3. **AI analysis** — creates summaries and insights.
            4. **Smart extraction** — identifies actions and decisions.
            5. **RAG chat** — answers questions using the transcript.
            """
        )

    st.markdown("---")
    st.caption("Powered by Whisper · Groq · LangChain · ChromaDB")


# --------------------------------------------------
# HERO HEADER
# --------------------------------------------------
st.markdown(
    """
    <div class="hero">
        <div class="hero-label">✦ AI-POWERED WORKSPACE</div>
        <h1>Turn recordings into<br><span>actionable intelligence.</span></h1>
        <p>
            Summarize long videos, uncover key decisions, identify next steps,
            and ask questions about your content — all from one intelligent workspace.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# RUN THE AI PIPELINE
# --------------------------------------------------
if run_button:
    if input_mode == "YouTube URL" and not source_path:
        st.warning("Please enter a YouTube URL first.")

    elif input_mode == "Local File Upload" and uploaded_file is None:
        st.warning("Please upload an audio or video file first.")

    else:
        temp_path = None

        try:
            if input_mode == "Local File Upload":
                suffix = os.path.splitext(uploaded_file.name)[1] or ".wav"

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=suffix
                ) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_path = temp_file.name

                source_path = temp_path

            with st.status(
                "Preparing your AI analysis...",
                expanded=True,
            ) as status:
                st.write("🎧 **1/5 — Processing audio**")
                chunks = process_input(source_path)

                st.write("🗣️ **2/5 — Transcribing speech**")
                transcript = transcribe_all(chunks, language_choice)

                if not transcript or not transcript.strip():
                    raise ValueError(
                        "Transcription returned empty text. "
                        "Try another recording or check its audio."
                    )

                st.write("✨ **3/5 — Generating summary and title**")
                title = generate_title(transcript)
                summary = summarize(transcript)

                st.write("📌 **4/5 — Extracting key insights**")
                action_items = extract_action_items(transcript)
                decisions = extract_key_decisions(transcript)
                questions = extract_questions(transcript)

                st.write("🧠 **5/5 — Preparing transcript Q&A**")
                rag_chain = build_rag_chain(transcript)

                st.session_state.pipeline_result = {
                    "title": title,
                    "transcript": transcript,
                    "summary": summary,
                    "action_items": action_items,
                    "key_decisions": decisions,
                    "open_questions": questions,
                    "rag_chain": rag_chain,
                    "language": language_choice,
                    "source": (
                        uploaded_file.name
                        if input_mode == "Local File Upload"
                        else source_path
                    ),
                }

                st.session_state.chat_messages = []
                st.session_state.quick_query = None

                status.update(
                    label="Your AI insights are ready!",
                    state="complete",
                    expanded=False,
                )

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.caption(
                "If this is a Groq rate-limit error (429), "
                "wait before retrying. Your input may need fewer tokens."
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------
result = st.session_state.pipeline_result

if result is None:
    st.markdown(
        '<div class="section-title">Your intelligent workspace</div>'
        '<div class="section-subtitle">'
        'Everything you need to understand your recordings, faster.'
        '</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    features = [
        (
            col1,
            "🎯",
            "Smart summaries",
            "Convert long recordings into clear, readable summaries "
            "with the main ideas highlighted.",
        ),
        (
            col2,
            "⚡",
            "Actionable insights",
            "Extract action items, confirmed decisions, and unresolved "
            "questions from the transcript.",
        ),
        (
            col3,
            "💬",
            "Chat with your video",
            "Ask questions about the recording and receive answers "
            "based on the indexed transcript.",
        ),
    ]

    for column, icon, heading, description in features:
        with column:
            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-icon">{icon}</div>
                    <h3>{heading}</h3>
                    <p>{description}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    left, right = st.columns([1.4, 1])

    with left:
        st.markdown(
            """
            <div class="result-panel">
                <div class="result-label">GET STARTED</div>
                <h3 style="color:#f8fafc; margin:0 0 10px 0;">
                    Your next insight is one recording away.
                </h3>
                <p style="color:#a5b4cc; font-size:13px; line-height:1.8;">
                    Choose a YouTube link or upload a recording in the sidebar,
                    select the spoken language, and generate your analysis.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="result-panel">
                <div class="result-label">YOUR TOOLKIT</div>
                <p>🎙️ Speech-to-text transcription</p>
                <p>📝 AI-generated summaries</p>
                <p>✅ Action item extraction</p>
                <p>🔑 Decision tracking</p>
                <p>🔎 Transcript search and Q&A</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

else:
    title = result.get("title", "Video Insights")
    transcript = result.get("transcript", "")
    words_count = len(transcript.split())

    st.markdown(
        '<div class="section-title">Analysis overview</div>'
        '<div class="section-subtitle">'
        'Your recording has been processed. Explore the insights below.'
        '</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)

    metrics = [
        (m1, "📝  Transcript words", f"{words_count:,}"),
        (m2, "🌐  Language", result.get("language", "English").capitalize()),
        (m3, "📌  Action items", "Ready"),
        (m4, "🧠  AI analysis", "Complete"),
    ]

    for column, label, value in metrics:
        with column:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{html.escape(str(value))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    top_left, top_right = st.columns([3, 1])

    with top_left:
        st.markdown(
            f"""
            <div class="result-panel">
                <div class="result-label">VIDEO / MEETING REPORT</div>
                <h2 style="color:#f8fafc; margin:0;">
                    {html.escape(str(title))}
                </h2>
                <p style="color:#a5b4cc; font-size:12px; margin-top:10px;">
                    Source: {html.escape(str(result.get("source", "Unknown")))}
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with top_right:
        st.markdown("<br>", unsafe_allow_html=True)
        report = generate_markdown_report(result)

        filename = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            str(title),
        ).strip("_")[:50] or "video_report"

        st.download_button(
            "⬇️ Download full report",
            data=report,
            file_name=f"{filename}.md",
            mime="text/markdown",
            use_container_width=True,
        )

    tab_summary, tab_actions, tab_decisions, tab_questions, tab_chat, tab_transcript = (
        st.tabs([
            "✨ Summary",
            "✅ Action items",
            "🔑 Decisions",
            "❓ Open questions",
            "💬 AI chat",
            "📜 Transcript",
        ])
    )

    with tab_summary:
        st.markdown("### Executive summary")
        st.caption("A concise overview of the main content.")

        summary_text = result.get("summary", "")
        if summary_text:
            st.markdown(
                f'<div class="result-panel">'
                f'<div class="result-label">KEY TAKEAWAYS</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(summary_text)
        else:
            st.info("No summary was generated.")

    with tab_actions:
        st.markdown("### Action items and next steps")
        st.caption("Tasks identified in the recording.")

        actions = result.get("action_items", "")
        if actions:
            st.markdown(actions)
        else:
            st.info("No action items were identified.")

    with tab_decisions:
        st.markdown("### Key decisions")
        st.caption("Decisions identified in the recording.")

        decisions = result.get("key_decisions", "")
        if decisions:
            st.markdown(decisions)
        else:
            st.info("No decisions were identified.")

    with tab_questions:
        st.markdown("### Open questions")
        st.caption("Unresolved questions or follow-up topics.")

        questions = result.get("open_questions", "")
        if questions:
            st.markdown(questions)
        else:
            st.info("No open questions were identified.")

    with tab_chat:
        st.markdown("### Chat with your recording")
        st.caption(
            "Ask a question about the transcript. "
            "Answers are generated using your indexed recording."
        )

        q1, q2, q3 = st.columns(3)

        if q1.button("🔍 Key takeaways", use_container_width=True):
            st.session_state.quick_query = (
                "What are the key takeaways from this recording?"
            )

        if q2.button("👥 Tasks and owners", use_container_width=True):
            st.session_state.quick_query = (
                "What tasks and owners are mentioned in this recording?"
            )

        if q3.button("💡 Important concepts", use_container_width=True):
            st.session_state.quick_query = (
                "What are the most important concepts explained?"
            )

        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        user_input = st.chat_input("Ask something about your recording...")

        query = st.session_state.quick_query
        if query:
            st.session_state.quick_query = None
        elif user_input:
            query = user_input

        if query:
            st.session_state.chat_messages.append(
                {"role": "user", "content": query}
            )

            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching the transcript and generating an answer..."):
                    try:
                        answer = ask_question(result["rag_chain"], query)
                        st.markdown(answer)

                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": answer}
                        )

                    except Exception as exc:
                        st.error(f"Could not answer this question: {exc}")

    with tab_transcript:
        st.markdown("### Full transcript")
        st.caption("Search the transcript or download it as a text file.")

        search_term = st.text_input(
            "Search transcript",
            placeholder="Enter a word or phrase...",
            key="transcript_search",
        )

        if search_term:
            matches = [
                line for line in transcript.splitlines()
                if search_term.lower() in line.lower()
            ]

            if matches:
                st.success(f"Found {len(matches)} matching line(s).")
                st.text("\n".join(matches))
            else:
                st.info("No matching lines found.")
        else:
            st.text_area(
                "Transcript text",
                value=transcript,
                height=400,
                disabled=True,
            )

        st.download_button(
            "⬇️ Download transcript",
            data=transcript,
            file_name="video_transcript.txt",
            mime="text/plain",
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <div class="footer">
        NEXA AI · VIDEO INTELLIGENCE STUDIO<br>
        Transforming recordings into knowledge and action.
    </div>
    """,
    unsafe_allow_html=True,
)
