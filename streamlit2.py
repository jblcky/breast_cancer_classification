import streamlit as st
import requests
from io import BytesIO

# --- Backend URLs ---
API_URL = "https://mammogram-rag-v0-1-236600620437.asia-southeast2.run.app"
PREDICT_URL = f"{API_URL}/predict-image"
ASK_URL = f"{API_URL}/ask-question"

# --- Helper Functions ---

# <-- 1. THIS IS THE NEW CSS FUNCTION WITH GEMINI STYLING -->
def load_css(theme: str):
    """
    Injects custom CSS to style the app closer to the Gemini aesthetic.
    """
    # Import Google's Roboto font
    st.markdown("""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

    # Define colors based on the selected theme
    if theme == "dark":
        colors = {
            "bg_color": "#131314", "font_color": "#E3E3E3",
            "user_color": "#89B4F8", "user_font_color": "#131314",
            "bot_color": "#333537",
        }
    else: # Light theme
        colors = {
            "bg_color": "#FFFFFF", "font_color": "#212121",
            "user_color": "#1A73E8", "user_font_color": "#FFFFFF",
            "bot_color": "#F1F3F4",
        }

    # The CSS string
    css = f"""
<style>
    /* --- Base App Style --- */
    .stApp {{
        background-color: {colors['bg_color']};
        font-family: 'Roboto', 'Google Sans', sans-serif;
    }}

    /* --- Chat Bubbles --- */
    /* This targets the container of a message to remove default padding */
    .st-emotion-cache-1c7y2kd {{
        padding: 0;
    }}

    .chat-bubble {{
        padding: 16px 20px; border-radius: 22px; margin: 8px 0;
        display: inline-block; max-width: 85%; word-wrap: break-word;
        font-size: 16px; line-height: 1.6; clear: both;
    }}
    .user-bubble {{
        background: {colors['user_color']}; color: {colors['user_font_color']}; float: right;
    }}
    .bot-bubble {{
        background: {colors['bot_color']}; color: {colors['font_color']}; float: left;
    }}

    /* --- Chat Input --- */
    .stChatInput textarea {{
        background-color: {colors['bot_color']}; color: {colors['font_color']};
        border-radius: 25px; border: 1px solid transparent;
    }}
    .stChatInput textarea:focus {{
        border-color: {colors['user_color']};
        box-shadow: 0 0 0 1px {colors['user_color']};
    }}
    h2 {{ color: {colors['font_color']}; }}
</style>
"""
    st.markdown(css, unsafe_allow_html=True)

def get_rag_answer(question: str):
    """Fetches an answer from the RAG backend."""
    try:
        res = requests.get(ASK_URL, params={"question": question}, timeout=30)
        res.raise_for_status()
        return res.json().get("Answer", "⚠️ Sorry, I couldn't find an answer.")
    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not connect to the analysis service: {e}"

def get_image_prediction(image_bytes: bytes, filename: str):
    """Sends an image to the backend for prediction."""
    try:
        files = {"file": (filename, image_bytes, "image/jpeg")}
        res = requests.post(PREDICT_URL, files=files, timeout=30)
        res.raise_for_status()
        data = res.json()
        prediction = data.get("Prediction", "Unknown")
        confidence = data.get("Confidence", "N/A")
        return f"**Prediction:** {prediction}\n\n**Confidence:** {confidence}"
    except requests.exceptions.RequestException as e:
        return f"⚠️ Could not connect to the prediction service: {e}"


# --- Page Config and Session State ---
st.set_page_config(page_title="Breast Cancer Chatbot", page_icon="🩺", layout="centered")

if "theme" not in st.session_state:
    st.session_state.theme = "light"
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I'm here to answer your questions about breast cancer. You can also upload a mammogram image for analysis."}]

# --- UI Rendering ---
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.markdown("<h2 style='text-align: center;'>🩺 Breast Cancer Chatbot</h2>", unsafe_allow_html=True)
with col3:
    if st.button("🌙" if st.session_state.theme == "light" else "☀️", key="theme_toggle"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"
        st.rerun()

# <-- 2. CALL THE CSS FUNCTION HERE, IN THE MAIN BODY OF THE SCRIPT -->
# This applies the styles right after the theme is set or changed.
load_css(st.session_state.theme)

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "image" in message:
            st.image(message["image"], caption="Uploaded for analysis", width=250)
        if "content" in message:
            bubble_class = "user-bubble" if message["role"] == "user" else "bot-bubble"
            st.markdown(f'<div class="chat-bubble {bubble_class}">{message["content"]}</div>', unsafe_allow_html=True)

# --- Input Handling ---
prompt = st.chat_input("Ask a question or upload an image...")
uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    img_bytes = uploaded_file.getvalue()
    st.session_state.messages.append({"role": "user", "image": BytesIO(img_bytes)})
    with st.spinner("Analyzing image..."):
        prediction_result = get_image_prediction(img_bytes, uploaded_file.name)
        st.session_state.messages.append({"role": "assistant", "content": prediction_result})
    st.rerun()

elif prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner("Finding an answer..."):
        answer = get_rag_answer(prompt)
        st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()
