# main.py (replace your current file with this)
from dotenv import load_dotenv

load_dotenv()

from typing import Set, Any, Dict, List, Optional
from io import BytesIO
import hashlib

import requests
from PIL import Image
import streamlit as st

from backend.core import run_llm

# Page config (must be before other Streamlit UI code)
st.set_page_config(
    page_title="Langchain RAG",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def create_sources_string(source_urls: Set[str]) -> str:
    if not source_urls:
        return ""
    sources_list = sorted(source_urls)
    sources_string = "sources:\n"
    for i, source in enumerate(sources_list):
        sources_string += f"{i+1}. {source}\n"
    return sources_string


def get_profile_picture(email: str) -> Image.Image:
    """
    Load profile picture from Google Drive (public link).
    Falls back to a simple placeholder if anything goes wrong.
    """
    try:
        # Google Drive file ID for your JPEG
        file_id = "1Ov6OW1gLvom99t4bVXpG5LPMjQEPXmkJ"
        drive_url = f"https://drive.google.com/uc?export=view&id={file_id}"
        resp = requests.get(drive_url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception:
        # fallback: simple placeholder image
        img = Image.new("RGB", (200, 200), color=(45, 45, 48))
        return img


# Custom CSS for dark theme and modern look
st.markdown(
    """
<style>
    .stApp {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stTextInput > div > div > input {
        background-color: #2D2D2D;
        color: #FFFFFF;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: #FFFFFF;
    }
    .stSidebar {
        background-color: #252526;
    }
    .stMessage {
        background-color: #2D2D2D;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar user information
with st.sidebar:
    st.title("User Profile")
    user_name = "Vedant"
    user_email = "robotliker7@gmail.com"
    profile_pic = get_profile_picture(user_email)
    st.image(profile_pic, width=150)
    st.write(f"**Name:** {user_name}")
    st.write(f"**Email:** {user_email}")

st.header("LangChain Documentation 🦜🔗- Helper Bot")

# Initialize session state
if "chat_answers_history" not in st.session_state:
    st.session_state["chat_answers_history"] = []
    st.session_state["user_prompt_history"] = []
    st.session_state["chat_history"] = []

# Create two columns for a modern layout
col1, col2 = st.columns([2, 1])

with col1:
    prompt = st.text_input("Prompt", placeholder="Enter your message here...")

with col2:
    if st.button("Submit", key="submit"):
        prompt = prompt or "Hello"

if prompt:
    with st.spinner("Generating response..."):
        # Call backend
        try:
            generated_response = run_llm(
                query=prompt, chat_history=st.session_state["chat_history"]
            )
        except Exception as e:
            st.error(f"Backend error: {e}")
            generated_response = {
                "answer": f"Error calling backend: {e}",
                "context": [],
            }

        # Accept either a string result or a dict with keys 'answer' and 'context'
        if isinstance(generated_response, str):
            answer_text = generated_response
            retrieved_docs: List[Any] = []
        else:
            # Try to be robust if keys missing
            answer_text = (
                generated_response.get("answer")
                if isinstance(generated_response, dict)
                else str(generated_response)
            )
            if answer_text is None:
                # As last resort, try to normalize raw
                answer_text = str(generated_response)
            retrieved_docs = (
                generated_response.get("context", [])
                if isinstance(generated_response, dict)
                else []
            )

        # Safely extract sources from metadata (try many common keys)
        sources = set()
        sample_metadatas = []
        for doc in retrieved_docs:
            meta = {}
            if hasattr(doc, "metadata"):
                meta = getattr(doc, "metadata") or {}
            elif isinstance(doc, dict):
                # If doc is a raw dict, metadata may be nested
                meta = doc.get("metadata") or doc
            # collect sample for debugging
            if len(sample_metadatas) < 5:
                sample_metadatas.append(meta)
            # check common keys
            src = None
            if isinstance(meta, dict):
                for key in (
                    "source",
                    "url",
                    "source_url",
                    "href",
                    "path",
                    "id",
                    "doc_id",
                ):
                    if meta.get(key):
                        src = meta.get(key)
                        break
            if src:
                sources.add(src)

        if not sources and sample_metadatas:
            # helpful debug information for missing metadata
            st.warning(
                "No `source` found on retrieved docs. Showing sample metadata to help debug ingestion."
            )
            st.json(sample_metadatas)

        formatted_response = f"{answer_text}\n\n{create_sources_string(sources)}"

        # update session state
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answers_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", answer_text))

# Display chat history
if st.session_state["chat_answers_history"]:
    for generated_response, user_query in zip(
        st.session_state["chat_answers_history"],
        st.session_state["user_prompt_history"],
    ):
        st.chat_message("user").write(user_query)
        st.chat_message("assistant").write(generated_response)

# Footer
st.markdown("---")
st.markdown("Powered by LangChain and Streamlit")
