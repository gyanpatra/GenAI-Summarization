import streamlit as st
import PyPDF2
from PerplexityClient import PerplexityClient

# List of available models (updated from Perplexity docs)
AVAILABLE_MODELS = [
    "sonar-pro",  # Advanced search model with grounding
    "sonar",  # Lightweight, cost-effective search model
    "sonar-deep-research",  # Expert-level research model
    "r1-1776",  # Offline DeepSeek R1 variant
]


def stream_perplexity_response(client, messages, model, stream=True):
    try:
        for chunk in client.chat(
            message=None, messages=messages, model=model, stream=stream
        ):
            yield chunk
    except Exception as e:
        yield f"[Error: {e}]"


st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")
st.title("🤖 Chat with your PDF (Perplexity)")

# Session state
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pdf_content" not in st.session_state:
    st.session_state["pdf_content"] = ""
if "system_prompt" not in st.session_state:
    st.session_state["system_prompt"] = "Be precise and concise."
if "model" not in st.session_state:
    st.session_state["model"] = AVAILABLE_MODELS[0]

# Sidebar: PDF upload, model, and system prompt
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    if uploaded_file:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        st.session_state["pdf_content"] = text
        st.success("PDF loaded!")
        st.write(text[:500] + ("..." if len(text) > 500 else ""))
    st.markdown("---")
    st.session_state["model"] = st.selectbox(
        "Model",
        AVAILABLE_MODELS,
        index=AVAILABLE_MODELS.index(st.session_state["model"]),
    )
    st.session_state["system_prompt"] = st.text_area(
        "System Prompt", st.session_state["system_prompt"]
    )

# Tabs for Chat and PDF Text
tabs = st.tabs(["💬 Chat", "📄 PDF Text"])

# --- Sticky chat input CSS ---
st.markdown(
    """
    <style>
    .sticky-chat-input {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100vw;
        background: #181825;
        z-index: 9999;
        padding: 1rem 0.5rem 0.5rem 0.5rem;
        box-shadow: 0 -2px 8px rgba(0,0,0,0.15);
    }
    .block-container { padding-bottom: 6rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)
# --- End sticky CSS ---

with tabs[0]:
    # Chat UI
    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.chat_message("user").write(msg["content"])
        else:
            st.chat_message("assistant").write(msg["content"])

    # Sticky chat input at bottom
    with st.container():
        st.markdown('<div class="sticky-chat-input">', unsafe_allow_html=True)
        prompt = st.chat_input("Ask something about your PDF...")
        st.markdown("</div>", unsafe_allow_html=True)

    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            client = PerplexityClient()
            # Compose full message history for context
            messages = [
                {"role": "system", "content": st.session_state["system_prompt"]}
            ]
            chat_history = st.session_state["messages"].copy()
            # Prepend PDF context to the first user message only
            if st.session_state["pdf_content"] and chat_history:
                for i, m in enumerate(chat_history):
                    if m["role"] == "user":
                        chat_history[i] = {
                            "role": "user",
                            "content": f"Context from PDF:\n{st.session_state['pdf_content']}\n\n{m['content']}",
                        }
                        break
            messages.extend(chat_history)
            response_placeholder = st.empty()
            response_text = ""
            for chunk in stream_perplexity_response(
                client, messages, st.session_state["model"], stream=True
            ):
                response_text += chunk
                response_placeholder.markdown(response_text)
            st.session_state["messages"].append(
                {"role": "assistant", "content": response_text}
            )

with tabs[1]:
    st.subheader("Extracted PDF Text")
    if st.session_state["pdf_content"]:
        st.text_area(
            "Full PDF Text",
            st.session_state["pdf_content"],
            height=400,
            key="pdf_text_area",
        )
    else:
        st.info("No PDF uploaded yet.")
