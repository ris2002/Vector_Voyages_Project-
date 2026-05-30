import streamlit as st
import hashlib
import uuid
import logging
import json
import os
import subprocess
import tempfile

from rag_backend.ocr import run_ocr
from rag_backend.utils import extract_title
from rag_backend.chunking_indexing import chunk_text, index_document
from rag_backend.agent import run_agent, classify_intent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── PPTX builder path ──────────────────────────────────────────────────────────
PPTX_BUILDER = os.path.join(os.path.dirname(__file__), "pptx_builder.js")


def _find_node() -> str:
    """
    Find the node binary — Streamlit on Mac loses the shell PATH so
    we probe common install locations explicitly.
    """
    import shutil
    # 1. Already on PATH (Linux / CI / most setups)
    node = shutil.which("node")
    if node:
        return node
    # 2. Homebrew on Apple Silicon
    for candidate in [
        "/opt/homebrew/bin/node",
        "/usr/local/bin/node",        # Homebrew on Intel Mac
        "/usr/bin/node",
        os.path.expanduser("~/.nvm/versions/node/*/bin/node"),  # nvm (glob)
    ]:
        import glob
        for path in glob.glob(candidate):
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return "node"   # last resort — will fail with a clear error


def _npm_global_root() -> str:
    """
    Find the npm global node_modules directory so NODE_PATH can be set
    for the subprocess — fixes 'Cannot find module pptxgenjs' on Mac/Linux
    when Streamlit strips the shell PATH.
    """
    try:
        result = subprocess.run(
            ["npm", "root", "-g"],
            capture_output=True, text=True, timeout=10
        )
        p = result.stdout.strip()
        if p and os.path.isdir(p):
            return p
    except Exception:
        pass
    # Common fallback locations
    for candidate in [
        os.path.expanduser("~/.npm-global/lib/node_modules"),
        "/usr/local/lib/node_modules",
        "/opt/homebrew/lib/node_modules",
        "/usr/lib/node_modules",
    ]:
        if os.path.isdir(candidate):
            return candidate
    return ""


def build_pptx(slide_data: dict) -> bytes | None:
    """
    Write slide_data to a temp JSON file, call pptx_builder.js via node,
    and return the .pptx bytes (or None on failure).
    """
    node_bin = _find_node()
    logger.info(f"build_pptx: using node at '{node_bin}'")

    # Resolve npm global root and inject into NODE_PATH so pptxgenjs is found
    npm_root = _npm_global_root()
    existing_node_path = os.environ.get("NODE_PATH", "")
    node_path = ":".join(filter(None, [npm_root, existing_node_path]))
    logger.info(f"build_pptx: NODE_PATH='{node_path}'")

    try:
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as jf:
            json.dump(slide_data, jf)
            json_path = jf.name

        out_path = json_path.replace(".json", ".pptx")
        result = subprocess.run(
            [node_bin, PPTX_BUILDER, json_path, out_path],
            capture_output=True, text=True, timeout=90,
            env={
                **os.environ,
                "PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:" + os.environ.get("PATH", ""),
                "NODE_PATH": node_path,
            },
        )
        logger.info(f"build_pptx stdout: {result.stdout.strip()}")
        if result.stderr.strip():
            logger.error(f"build_pptx stderr: {result.stderr.strip()}")

        if result.returncode != 0:
            logger.error(f"pptx_builder.js exited {result.returncode}")
            return None

        if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
            logger.error("pptx_builder.js ran but output file missing/empty")
            return None

        with open(out_path, "rb") as f:
            pptx_bytes = f.read()

        os.unlink(json_path)
        os.unlink(out_path)
        return pptx_bytes
    except Exception as e:
        logger.error(f"build_pptx exception: {e}")
        return None


def slides_to_markdown(slide_data: dict) -> str:
    """
    Render slide_data dict as a readable markdown preview.
    Shows exactly what was generated so the user can verify it matched their request.
    """
    fmt              = slide_data.get("format", "bullets")
    detail           = slide_data.get("detail", "normal")
    slides           = slide_data.get("slides", [])
    points_per_slide = slide_data.get("points_per_slide")
    point_length     = slide_data.get("point_length")

    fmt_label    = {"bullets": "• Bullets", "paragraphs": "¶ Paragraphs", "both": "• + ¶ Both"}.get(fmt, fmt)
    detail_label = {"brief": "Brief", "normal": "Normal", "detailed": "Detailed"}.get(detail, detail)
    len_label    = {"long": "Long points", "short": "Short points"}.get(point_length, "") if point_length else ""

    # Build a spec summary line showing exactly what was produced
    parts = [f"**{len(slides)} slides**", f"Format: {fmt_label}"]
    if points_per_slide is not None:
        word = "paragraphs" if fmt == "paragraphs" else "points"
        parts.append(f"{points_per_slide} {word}/slide")
    else:
        parts.append(f"Depth: {detail_label}")
    if len_label:
        parts.append(len_label)

    lines = ["📊 " + " · ".join(parts) + "\n"]

    for i, sl in enumerate(slides, 1):
        title = sl.get("title", f"Slide {i}")
        lines.append(f"---\n**Slide {i}: {title}**")

        # Show actual count so user can spot if LLM under-delivered
        if fmt in ("bullets", "both"):
            content = sl.get("content") or []
            n = len(content)
            expected = f" *(expected {points_per_slide})*" if points_per_slide and n != points_per_slide else ""
            if content:
                lines.append(f"*{n} points{expected}*")
                for pt in content:
                    lines.append(f"- {pt}")
                lines.append("")

        if fmt in ("paragraphs", "both"):
            para = sl.get("paragraph", "")
            if para:
                # Count paragraphs by blank-line splits
                paras = [p.strip() for p in para.split("\n\n") if p.strip()]
                n = len(paras)
                expected = f" *(expected {points_per_slide})*" if points_per_slide and n != points_per_slide else ""
                lines.append(f"*{n} paragraph(s){expected}*")
                lines.append(para)
                lines.append("")

        if sl.get("speaker_note"):
            lines.append(f"🎤 *{sl['speaker_note']}*\n")

    return "\n".join(lines)



st.set_page_config(
    page_title="Vector Voyagers — Research Paper Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Intent badge colours ───────────────────────────────────────────────────────
INTENT_LABELS = {
    "summarise":        ("📝 Summary",              "#2e7d32"),
    "extract_entities": ("🔍 Entity Extraction",    "#1565c0"),
    "compare":          ("⚖️ Comparison",            "#6a1b9a"),
    "explain":          ("💡 Explanation",           "#e65100"),
    "list_sections":    ("📋 Sections",              "#00695c"),
    "qa":               ("❓ Q&A",                   "#37474f"),
    # New intents
    "simplify":         ("🧒 Simplified",            "#4527a0"),
    "jargon":           ("📖 Jargon Buster",         "#00838f"),
    "unsolved":         ("🔓 Open Problems",         "#ad1457"),
    "method":           ("⚙️ Method Explained",      "#558b2f"),
    "quiz":             ("🧠 Quiz",                  "#f57f17"),
    "build_it":         ("🛠️ Build It",              "#4e342e"),
    "similar_papers":   ("📚 Similar Papers",        "#283593"),
    "slides":           ("🎞️ Slides",                "#6d4c41"),
}

# ── Session state init ─────────────────────────────────────────────────────────
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = None
if "editing_msg_idx" not in st.session_state:
    st.session_state.editing_msg_idx = None
if "pending_pptx_data" not in st.session_state:
    st.session_state.pending_pptx_data = None
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "cancel_generation" not in st.session_state:
    st.session_state.cancel_generation = [False]  # mutable list — shared with thread
if "generating_chat_id" not in st.session_state:
    st.session_state.generating_chat_id = None
if "result_queue" not in st.session_state:
    import queue
    st.session_state.result_queue = queue.Queue()  # thread-safe result channel

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💬 Vector Voyagers")
    if st.button("＋  New Chat", use_container_width=True):
        new_id = str(uuid.uuid4())
        st.session_state.chats[new_id] = {
            "title":       "New Chat",
            "title_ready": False,
            "pdf_name":    None,
            "pdf_hash":    None,
            "ocr_text":    None,
            "file_bytes":  None,
            "messages":    [],
            "status":      "pending",
            "chunks":      0,
        }
        st.session_state.active_chat_id = new_id
        st.rerun()

    st.divider()
    st.caption("Recent Chats")
    for chat_id, chat in st.session_state.chats.items():
        label = chat["title"]
        if not chat.get("title_ready", True):
            label = f"⏳ {label}"
        if st.button(label, key=chat_id, use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()

    # ── Debug panel ───────────────────────────────────────────────────────────
    active = st.session_state.active_chat_id
    if active and active in st.session_state.chats:
        ch = st.session_state.chats[active]
        st.divider()
        st.caption("🔍 Debug Info")
        st.write(f"**Status:** `{ch['status']}`")
        st.write(f"**Chunks indexed:** `{ch['chunks']}`")
        st.write(f"**PDF:** `{ch['pdf_name']}`")

# ── Main area ──────────────────────────────────────────────────────────────────
if st.session_state.active_chat_id is None:
    st.title("📄 Research Paper Assistant")
    st.info("Click **＋ New Chat** in the sidebar to upload a paper and start chatting.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📝 Core Features")
        st.markdown("""
- *"Give me a summary of this paper"*
- *"Who are the authors?"*
- *"What dataset was used?"*
- *"List the sections of this paper"*
- *"What were the main results?"*
- *"Any specific question about the paper"*
        """)

        st.markdown("#### 💡 Understand Better")
        st.markdown("""
- *"Explain [concept] from the paper"*
- *"What does [term] mean?"* — jargon buster
- *"Explain this paper in simple words"* — beginner mode
- *"How does their method work?"*
        """)

        st.markdown("#### ⚖️ Compare & Contrast")
        st.markdown("""
- *"Compare [method A] and [method B]"*
- *"What's the difference between X and Y?"*
        """)

    with col2:
        st.markdown("#### 🔍 Dig Deeper")
        st.markdown("""
- *"What problems are still unsolved after this paper?"*
- *"What are the limitations?"*
- *"Suggest similar papers to read next"*
        """)

        st.markdown("#### 🛠️ Action Features")
        st.markdown("""
- *"How do I build what this paper proposes?"*
- *"Generate quiz questions from this paper"*
- *"Create slides in bullet points"* — short, concise points per slide
- *"Create slides in paragraphs"* — full-sentence narrative per slide
        """)

else:
    chat_id = st.session_state.active_chat_id
    chat    = st.session_state.chats[chat_id]
    msgs    = chat["messages"]
    status  = chat["status"]

    st.title(chat["title"])

    # ── Upload & index ─────────────────────────────────────────────────────────
    if status == "pending":
        uploaded_file = st.file_uploader(
            "Upload a PDF to start chatting",
            type=["pdf"],
            key=f"uploader_{chat_id}"
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            file_hash  = hashlib.md5(file_bytes).hexdigest()

            with st.spinner("Extracting text from document…"):
                try:
                    extracted_text = run_ocr(file_bytes)
                except Exception as e:
                    st.error(f"OCR failed: {e}")
                    st.stop()

            with st.spinner("Chunking and indexing…"):
                try:
                    chunks = chunk_text(extracted_text)
                    if not chunks:
                        st.error("No text could be extracted from this PDF.")
                        st.stop()
                    index_document(chunks, chat_id)
                except Exception as e:
                    st.error(f"Indexing failed: {e}")
                    logger.exception("Indexing error")
                    st.stop()

            st.session_state.chats[chat_id].update({
                "status":      "ready",
                "title":       uploaded_file.name,
                "title_ready": False,
                "pdf_name":    uploaded_file.name,
                "pdf_hash":    file_hash,
                "ocr_text":    extracted_text,
                "file_bytes":  file_bytes,
                "chunks":      len(chunks),
            })
            st.rerun()

    # ── Background title extraction ────────────────────────────────────────────
    if status == "ready" and not chat.get("title_ready", True):
        with st.spinner("Extracting paper title…"):
            try:
                title = extract_title(
                    chat["file_bytes"],
                    chat["ocr_text"],
                    chat["pdf_name"]
                )
            except Exception:
                title = chat["pdf_name"]
        st.session_state.chats[chat_id]["title"]       = title
        st.session_state.chats[chat_id]["title_ready"] = True
        st.session_state.chats[chat_id]["file_bytes"]  = None
        st.rerun()

    # ── Quick action buttons (shown when ready, no messages yet) ──────────────
    if status == "ready" and len(msgs) == 0:
        st.markdown("#### 🚀 Quick Actions")
        st.caption("Click any button to get started:")

        row1 = st.columns(4)
        row2 = st.columns(4)
        row3 = st.columns(2)

        quick_actions = [
            ("📝 Summarise", "Give me a summary of this paper"),
            ("🧒 Simplify", "Explain this paper in simple words for a beginner"),
            ("⚙️ Method", "How does their method work?"),
            ("🔍 Entities", "Who are the authors and what dataset was used?"),
            ("📋 Sections", "List the sections of this paper"),
            ("🔓 Open Problems", "What problems are still unsolved after this paper?"),
            ("📚 Similar Papers", "Suggest similar papers to read next"),
            ("🧠 Quiz Me", "Generate quiz questions from this paper"),
        ]

        rows = [row1, row2]
        for idx, (label, prompt) in enumerate(quick_actions):
            row_idx = idx // 4
            col_idx = idx % 4
            with rows[row_idx][col_idx]:
                if st.button(label, use_container_width=True, key=f"quick_{idx}"):
                    # Inject as if user typed it
                    intent = classify_intent(prompt)
                    msgs.append({"role": "user", "content": prompt})
                    with st.spinner("Thinking…"):
                        try:
                            answer = run_agent(
                                query=prompt,
                                chat_id=chat_id,
                                history=[],
                                ocr_text=chat.get("ocr_text", ""),
                            )
                        except Exception as e:
                            answer = f"❌ Agent error: {e}"
                            logger.exception("quick action run_agent error")
                    msgs.append({"role": "assistant", "content": answer, "intent": intent})
                    st.rerun()

        with row3[0]:
            if st.button("🛠️ How to Build It", use_container_width=True, key="quick_build"):
                prompt = "How do I build what this paper proposes?"
                intent = classify_intent(prompt)
                msgs.append({"role": "user", "content": prompt})
                with st.spinner("Thinking…"):
                    try:
                        answer = run_agent(
                            query=prompt, chat_id=chat_id,
                            history=[], ocr_text=chat.get("ocr_text", ""),
                        )
                    except Exception as e:
                        answer = f"❌ Agent error: {e}"
                msgs.append({"role": "assistant", "content": answer, "intent": intent})
                st.rerun()

        with row3[1]:
            if st.button("🎞️ Generate Slides", use_container_width=True, key="quick_slides"):
                prompt = "Create presentation slides for this paper in bullet points"
                intent = classify_intent(prompt)
                msgs.append({"role": "user", "content": prompt})
                with st.spinner("Thinking…"):
                    try:
                        answer = run_agent(
                            query=prompt, chat_id=chat_id,
                            history=[], ocr_text=chat.get("ocr_text", ""),
                        )
                    except Exception as e:
                        answer = f"❌ Agent error: {e}"
                pptx_bytes = None
                if isinstance(answer, dict) and "slides" in answer:
                    with st.spinner("Building .pptx…"):
                        pptx_bytes = build_pptx(answer)
                msgs.append({"role": "assistant", "content": answer, "intent": intent, "pptx_bytes": pptx_bytes})
                st.rerun()

        st.divider()

    # ── Chat history display ───────────────────────────────────────────────────
    for msg_idx, msg in enumerate(msgs):
        with st.chat_message(msg["role"]):

            # ── USER messages: show text + ✏️ edit button ──────────────────
            if msg["role"] == "user":
                if st.session_state.editing_msg_idx == msg_idx:
                    edited = st.text_area(
                        "Edit your message:",
                        value=msg["content"],
                        key=f"edit_area_{msg_idx}",
                        height=80,
                    )
                    col_save, col_cancel = st.columns([1, 1])
                    with col_save:
                        if st.button("✅ Send", key=f"save_{msg_idx}", use_container_width=True):
                            msgs_before = msgs[:msg_idx]
                            msgs[msg_idx]["content"] = edited
                            while len(msgs) > msg_idx + 1:
                                msgs.pop()
                            st.session_state.editing_msg_idx = None
                            new_intent = classify_intent(edited)
                            with st.spinner("Thinking…"):
                                try:
                                    new_answer = run_agent(
                                        query=edited,
                                        chat_id=chat_id,
                                        history=msgs_before,
                                        ocr_text=chat.get("ocr_text", ""),
                                    )
                                except Exception as e:
                                    new_answer = f"❌ Agent error: {e}"
                            pptx_bytes = None
                            if new_intent == "slides" and isinstance(new_answer, dict) and "slides" in new_answer:
                                with st.spinner("Building .pptx…"):
                                    pptx_bytes = build_pptx(new_answer)
                            msgs.append({"role": "assistant", "content": new_answer, "intent": new_intent, "pptx_bytes": pptx_bytes})
                            st.rerun()
                    with col_cancel:
                        if st.button("✖ Cancel", key=f"cancel_{msg_idx}", use_container_width=True):
                            st.session_state.editing_msg_idx = None
                            st.rerun()
                else:
                    col_text, col_btn = st.columns([11, 1])
                    with col_text:
                        st.write(msg["content"])
                    with col_btn:
                        if st.button("✏️", key=f"edit_btn_{msg_idx}", help="Edit this message"):
                            st.session_state.editing_msg_idx = msg_idx
                            st.rerun()

            # ── ASSISTANT messages ─────────────────────────────────────────
            else:
                if msg.get("intent"):
                    lbl, col = INTENT_LABELS.get(msg["intent"], ("❓ Q&A", "#37474f"))
                    st.markdown(
                        f'<span style="background:{col};color:white;padding:2px 10px;'
                        f'border-radius:12px;font-size:0.75em;">{lbl}</span>',
                        unsafe_allow_html=True
                    )
                content = msg["content"]
                if isinstance(content, dict) and "slides" in content:
                    st.markdown(slides_to_markdown(content))
                    if msg.get("pptx_bytes"):
                        st.download_button(
                            label="⬇️ Download .pptx",
                            data=msg["pptx_bytes"],
                            file_name="presentation.pptx",
                            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
                            key=f"dl_{msg_idx}",
                        )
                    else:
                        st.info("💡 Want this as a PowerPoint file you can download?")
                        if st.button("📥 Yes, generate .pptx", key=f"gen_pptx_{msg_idx}", type="primary"):
                            with st.spinner("Building your .pptx file…"):
                                pptx_bytes = build_pptx(content)
                            if pptx_bytes:
                                msg["pptx_bytes"] = pptx_bytes
                                st.rerun()
                            else:
                                st.error(
                                    "❌ Could not build .pptx — check the terminal logs for details.\n\n"
                                    "If pptxgenjs is not installed, run: `npm install -g pptxgenjs`"
                                )
                else:
                    st.write(content)


    # ── Auto-scroll to bottom ──────────────────────────────────────────────────
    st.components.v1.html(
        """<script>
        window.parent.document.querySelectorAll('section.main')[0]
            .scrollTo(0, 999999);
        </script>""",
        height=0,
    )

    # ── Auto-scroll to bottom after every render ──────────────────────────────
    st.components.v1.html("""
        <script>
            // Scroll the Streamlit main content area to the bottom
            function scrollToBottom() {
                const main = window.parent.document.querySelector('section.main');
                if (main) main.scrollTop = main.scrollHeight;
            }
            scrollToBottom();
            // Also run after a short delay to catch late-rendering content
            setTimeout(scrollToBottom, 300);
        </script>
    """, height=0)

    # ── Chat input ─────────────────────────────────────────────────────────────
    import threading
    import queue as _queue
    import time as _time

    # ── Thread worker — puts result into a thread-safe Queue ─────────────────
    def _run_agent_thread(q, query, chat_id, history, ocr_text, intent, cancel_flag):
        try:
            answer = run_agent(
                query    = query,
                chat_id  = chat_id,
                history  = history,
                ocr_text = ocr_text,
            )
        except Exception as e:
            logger.exception("run_agent thread error")
            answer = f"❌ Agent error: {e}"
        if not cancel_flag[0]:           # cancel_flag is a mutable list so thread can read it
            q.put((answer, intent))

    placeholder = (
        "Ask a question about the paper…"
        if status == "ready"
        else "Upload a PDF first to enable chat"
    )

    # ── Check if a result arrived in the queue ────────────────────────────────
    if st.session_state.is_generating and st.session_state.generating_chat_id == chat_id:
        try:
            answer, intent = st.session_state.result_queue.get_nowait()
            # Got a result — commit it
            st.session_state.is_generating = False
            st.session_state.generating_chat_id = None

            pptx_bytes = None
            if intent == "slides" and isinstance(answer, dict) and "slides" in answer:
                with st.spinner("Building .pptx…"):
                    pptx_bytes = build_pptx(answer)

            msgs.append({"role": "assistant", "content": answer, "intent": intent, "pptx_bytes": pptx_bytes})
            st.rerun()

        except _queue.Empty:
            # Still running — show spinner + Stop button then poll again
            st.markdown("---")
            col_spin, col_stop = st.columns([5, 1])
            with col_spin:
                st.markdown(
                    "⏳ &nbsp;**Thinking…** &nbsp;<span style='color:#888;font-size:0.85em'>"
                    "This may take a few seconds</span>",
                    unsafe_allow_html=True,
                )
            with col_stop:
                if st.button("⏹ Stop", type="secondary", use_container_width=True):
                    st.session_state.cancel_generation[0] = True
                    st.session_state.is_generating = False
                    st.session_state.generating_chat_id = None
                    # drain the queue so stale results don't appear later
                    while not st.session_state.result_queue.empty():
                        try: st.session_state.result_queue.get_nowait()
                        except: break
                    if msgs and msgs[-1]["role"] == "user":
                        msgs.pop()
                    st.rerun()
            _time.sleep(0.25)
            st.rerun()

    # ── Normal input ──────────────────────────────────────────────────────────
    user_input = st.chat_input(
        placeholder,
        disabled=(status != "ready" or st.session_state.is_generating),
    )

    if user_input and status == "ready" and not st.session_state.is_generating:
        intent = classify_intent(user_input)
        msgs.append({"role": "user", "content": user_input})

        # Fresh cancel flag (mutable list so thread shares the reference)
        st.session_state.cancel_generation = [False]

        # Drain any leftover results from a previous run
        while not st.session_state.result_queue.empty():
            try: st.session_state.result_queue.get_nowait()
            except: break

        st.session_state.is_generating = True
        st.session_state.generating_chat_id = chat_id

        thread = threading.Thread(
            target=_run_agent_thread,
            args=(
                st.session_state.result_queue,
                user_input,
                chat_id,
                msgs[:-1],
                chat.get("ocr_text", ""),
                intent,
                st.session_state.cancel_generation,
            ),
            daemon=True,
        )
        thread.start()
        st.rerun()
