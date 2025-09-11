import streamlit as st
import requests

# --- Backend URLs ---
API_URL = "https://mammogram-ragv0-0-236600620437.asia-southeast1.run.app"
PREDICT_URL = f"{API_URL}/predict-image"
ASK_URL = f"{API_URL}/ask-question"

# --- Page Config ---
st.set_page_config(page_title="Breast Cancer Chatbot", page_icon="🩺", layout="centered")

# --- Theme Toggle in Header ---
if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("<h2 style='text-align: center;'>🩺 Breast Cancer Chatbot</h2>", unsafe_allow_html=True)
with col3:
    if st.button("🌞" if st.session_state["theme"] == "light" else "🌙"):
        st.session_state["theme"] = "dark" if st.session_state["theme"] == "light" else "light"

# --- Colors for themes ---
if st.session_state["theme"] == "light":
    bg_color = "#FFFFFF"
    user_color = "#2563EB"
    bot_color = "#F1F5F9"
    font_color = "#111827"
else:
    bg_color = "#0F172A"
    user_color = "#3B82F6"
    bot_color = "#1E293B"
    font_color = "#F9FAFB"

# --- Custom CSS (Gemini style) ---
st.markdown(f"""
<style>
    .stApp {{
        background-color: {bg_color};
        color: {font_color};
        font-family: 'Segoe UI', sans-serif;
    }}
    .chat-container {{
        max-width: 720px;
        margin: auto;
        padding: 20px;
    }}
    .bubble {{
        padding: 14px 18px;
        border-radius: 18px;
        margin: 10px 0;
        display: inline-block;
        max-width: 85%;
        word-wrap: break-word;
        font-size: 16px;
        line-height: 1.5;
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
    .stChatInput textarea {{
        border-radius: 25px !important;
        padding: 14px !important;
        font-size: 16px !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- Backend Helper Functions ---
def get_rag_answer(question: str):
    try:
        res = requests.post(ASK_URL, json={"question": question})
        if res.status_code == 200:
            return res.json().get("Answer", "⚠️ No answer")
        else:
            return f"⚠️ Error: {res.status_code}"
    except Exception as e:
        return f"⚠️ Could not connect to backend: {e}"

def get_image_prediction(uploaded_file):
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
        res = requests.post(PREDICT_URL, files=files)
        if res.status_code == 200:
            data = res.json()
            prediction = data.get("Prediction", "Unknown")
            confidence = data.get("Confidence", "?")
            return f"📷 Prediction: **{prediction}** (Confidence: {confidence})"
        else:
            return f"⚠️ Error: {res.status_code}"
    except Exception as e:
        return f"⚠️ Could not connect to backend: {e}"

# --- Chat Container ---
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.markdown(f"<div class='bubble user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bubble bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# --- Chat Input ---
prompt = st.chat_input("Ask about breast cancer or upload an image...")

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    answer = get_rag_answer(prompt)
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

# --- Image Upload ---
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    st.chat_message("user").image(uploaded_file, caption="Uploaded image")
    reply = get_image_prediction(uploaded_file)
    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
