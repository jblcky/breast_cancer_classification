import streamlit as st
from PIL import Image
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Breast Cancer Chatbot", page_icon="🩺", layout="centered")

# --- Custom CSS for styling ---
st.markdown("""
<style>
    /* Page background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
        font-family: 'Segoe UI', sans-serif;
    }

    /* Chat container */
    .chat-container {
        max-width: 700px;
        margin: auto;
        padding: 20px;
        background: white;
        border-radius: 20px;
        box-shadow: 0px 6px 15px rgba(0,0,0,0.1);
    }

    /* Chat bubble - user */
    .user-bubble {
        background: #DCF8C6;
        color: black;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
    }

    /* Chat bubble - assistant */
    .bot-bubble {
        background: #E6E6E6;
        color: black;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        display: inline-block;
        max-width: 80%;
    }

    /* Align bubbles */
    .user-msg { text-align: right; }
    .bot-msg { text-align: left; }
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h2 style='text-align: center;'>🩺 Mammogram Chatbot</h2>", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Chat Container ---
with st.container():
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    for msg in st.session_state["messages"]:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-msg'><div class='user-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='bot-msg'><div class='bot-bubble'>{msg['content']}</div></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input ---
col1, col2 = st.columns([4,1])
with col1:
    prompt = st.text_input("Type your question here...", key="input")
with col2:
    send_btn = st.button("Send")

if send_btn and prompt:
    # User message
    st.session_state["messages"].append({"role": "user", "content": prompt})

    # Placeholder bot reply (replace with your RAG backend)
    reply = f"🤖 Answer for: {prompt}"
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.experimental_rerun()

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Mammogram", use_column_width=True)

    # Placeholder prediction (replace with your model inference)
    prediction = "benign"  # or "malignant"
    reply = f"📷 Prediction: **{prediction}**"
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.experimental_rerun()
