import os
from io import BytesIO

import streamlit as st
from groq import Groq
from pypdf import PdfReader
from docx import Document


# -------------------- Page setup --------------------
st.set_page_config(
    page_title="Saad | AI Study Assistant",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Saad")
st.caption("Your AI study partner — ask questions, understand concepts, and learn step by step.")

SYSTEM_PROMPT = """
You are Saad, a friendly, patient, and knowledgeable AI study assistant.
Help the student learn rather than simply giving unexplained answers.
Explain concepts in clear, student-friendly language, step by step.
Use examples, analogies, summaries, and practice questions when useful.
When the student uploads study material, prioritize that material and
say clearly when an answer is not covered by the uploaded material.
If a question is ambiguous, ask a brief clarifying question.
Do not pretend to know something you do not know.
"""


# -------------------- API key --------------------
def get_api_key():
    """Read the key from Streamlit secrets first, then environment variables."""
    try:
        key = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("GROQ_API_KEY", "")


# -------------------- Study file extraction --------------------
def extract_text(uploaded_file):
    """Extract readable text from supported study file formats."""
    filename = uploaded_file.name.lower()
    raw = uploaded_file.getvalue()

    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(BytesIO(raw))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()

        if filename.endswith(".docx"):
            doc = Document(BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs).strip()

        if filename.endswith((".txt", ".md", ".csv")):
            return raw.decode("utf-8", errors="replace").strip()

        return ""
    except Exception as exc:
        return f"[Could not read {uploaded_file.name}: {exc}]"


# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("📚 Your study space")
    st.write("Upload notes or documents. Saad can use their text as context when answering.")

    uploaded_files = st.file_uploader(
        "Upload study material",
        type=["pdf", "docx", "txt", "md", "csv"],
        accept_multiple_files=True,
        help="Supported: PDF, Word (.docx), text, Markdown, and CSV files.",
    )

    st.divider()
    model_name = st.selectbox(
        "Groq model",
        options=[
            "llama-3.3-70b-versatile",
            "openai/gpt-oss-20b",
            "qwen/qwen3-32b",
        ],
        index=0,
        help="If a model is unavailable to your Groq account, choose another available model.",
    )

    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("🔐 Your Groq API key should be stored in Streamlit Secrets or an environment variable — never commit it to GitHub.")


# -------------------- Session state --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "study_context" not in st.session_state:
    st.session_state.study_context = ""

# Rebuild context when uploaded files change.
if uploaded_files:
    context_parts = []
    total_chars = 0
    max_total_chars = 30000

    for file in uploaded_files:
        text = extract_text(file)
        if not text:
            context_parts.append(f"\n--- {file.name} ---\n[No readable text found.]")
            continue

        remaining = max_total_chars - total_chars
        if remaining <= 0:
            context_parts.append("\n[Additional document text omitted because the context limit was reached.]")
            break

        clipped = text[:remaining]
        context_parts.append(f"\n--- FILE: {file.name} ---\n{clipped}")
        total_chars += len(clipped)

    st.session_state.study_context = "\n".join(context_parts)
    st.sidebar.success(f"{len(uploaded_files)} file(s) ready")
else:
    st.session_state.study_context = ""

# Show uploaded file names in the main area.
if uploaded_files:
    with st.expander("📎 Uploaded study material", expanded=False):
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        st.caption("Saad uses extracted text from these files. Scanned PDFs and images may not contain extractable text.")


# -------------------- Chat history --------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -------------------- Chat input and Groq response --------------------
prompt = st.chat_input("Ask Saad anything about your studies...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    api_key = get_api_key()
    if not api_key:
        error_message = (
            "Groq API key not found. Add `GROQ_API_KEY` to your Streamlit Secrets "
            "or set it as an environment variable, then reload the app."
        )
        with st.chat_message("assistant"):
            st.error(error_message)
        st.stop()

    # Add uploaded study material to the system instructions for this request.
    system_message = SYSTEM_PROMPT
    if st.session_state.study_context:
        system_message += (
            "\n\nThe student has uploaded the following study material. "
            "Use it as the primary source when relevant:\n"
            + st.session_state.study_context
        )

    # Send recent conversation history to keep the prompt manageable.
    api_messages = [{"role": "system", "content": system_message}]
    api_messages.extend(st.session_state.messages[-20:])

    try:
        client = Groq(api_key=api_key)

        with st.chat_message("assistant"):
            with st.spinner("Saad is thinking..."):
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=api_messages,
                    temperature=0.5,
                    max_completion_tokens=2048,
                )
                answer = completion.choices[0].message.content or "I couldn't generate a response. Please try again."
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as exc:
        with st.chat_message("assistant"):
            st.error(f"Something went wrong while contacting Groq: {exc}")
            st.info("Check that your API key is valid and that the selected model is available in your Groq account.")
