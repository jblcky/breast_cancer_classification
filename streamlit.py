import streamlit as st
import requests
import io

# --- Backend URLs ---
API_URL = "http://localhost:8000"  # change if deployed
PREDICT_URL = f"{API_URL}/predict-image"
ASK_URL = f"{API_URL}/ask-question"

# --- Page Config ---
st.set_page_config(page_title="Breast Cancer Chatbot", page_icon="🩺", layout="wide")

# --- Theme Toggle ---
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

theme_choice = st.sidebar.radio("Theme", ["🌞 Light", "🌙 Dark"])
st.session_state["theme"] = "dark" if "Dark" in theme_choice else "light"

# --- Colors for themes ---
if st.session_state["theme"] == "light":
    bg_color = "#F9FAFB"
    user_color = "#2563EB"     # Blue
    bot_color = "#E5E7EB"      # Light gray
    font_color = "#111827"
else:
    bg_color = "#111827"
    user_color = "#3B82F6"     # Bright blue
    bot_color = "#374151"      # Dark gray
    font_color = "#F9FAFB"

# --- Custom CSS ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {font_color};
        font-family: 'Segoe UI', sans-serif;
    }}
    .chat-container {{
        max-width: 800px;
        margin: auto;
        padding: 20px;
    }}
    .bubble {{
        padding: 12px 16px;
        border-radius: 20px;
        margin: 8px 0;
        display: inline-block;
        max-width: 80%;
        word-wrap: break-word;
        font-size: 15px;
    }}
    .user-bubble {{
        background: {user_color};
        color: white;
        float: right;
        clear: both;
    }}
    .bot-bubble {{
        background: {bot_color};
        color: {font_color};
        float: left;
        clear: both;
    }}
</style>
""", unsafe_allow_html=True)

# --- Title ---
st.markdown("<h2 style='text-align: center;'>🩺 Breast Cancer Chatbot</h2>", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Chat Container ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='bubble user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bubble bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input ---
prompt = st.chat_input("Ask a question or upload an image...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    try:
        # call FastAPI ask-question endpoint
        res = requests.post(ASK_URL, json={"question": prompt})
        if res.status_code == 200:
            answer = res.json().get("Answer", "⚠️ No answer")
        else:
            answer = f"⚠️ Error: {res.status_code}"
    except Exception as e:
        answer = f"⚠️ Could not connect to backend: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

# --- Image Upload ---
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    # Show the uploaded image in chat (optional, for user confirmation)
    st.chat_message("user").image(uploaded_file, caption="Uploaded image")

    # Send raw file directly to backend
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        res = requests.post(PREDICT_URL, files=files)
        if res.status_code == 200:
            data = res.json()
            prediction = data.get("Prediction", "Unknown")
            confidence = data.get("Confidence", "?")
            reply = f"📷 Prediction: **{prediction}** (Confidence: {confidence})"
        else:
            reply = f"⚠️ Error: {res.status_code}"
    except Exception as e:
        reply = f"⚠️ Could not connect to backend: {e}"

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
