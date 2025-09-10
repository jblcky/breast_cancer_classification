import streamlit as st
from PIL import Image

st.set_page_config(page_title="Mammo Chatbot", layout="wide")

st.title("💬 Mammogram Chatbot")

# --- session state to store conversation ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- render chat history ---
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- chat input ---
prompt = st.chat_input("Ask a question or upload an image...")

if prompt:
    # user sends text
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # placeholder bot reply (replace with your RAG answer fn)
    answer = f"🤖 Answer for: {prompt}"
    st.session_state["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)

# --- image upload inside chat ---
uploaded_file = st.file_uploader("Upload mammogram image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.chat_message("user").image(img, caption="Uploaded image")

    # placeholder prediction (replace with your model inference)
    prediction = "benign"  # or "malignant"
    reply = f"📷 Prediction: **{prediction}**"

    st.session_state["messages"].append({"role": "assistant", "content": reply})
    st.chat_message("assistant").write(reply)
