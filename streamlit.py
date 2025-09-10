import streamlit as st

st.set_page_config(page_title="Mammogram Classifier & Chatbot", layout="wide")

st.title("📸 Mammogram Classifier + Breast Cancer Chatbot")

st.write(
    "Upload one mammogram image to get a benign / malignant prediction, and ask breast cancer questions using the chatbot."
)

col1, col2 = st.columns([1, 1])

# Image upload and prediction
with col1:
    st.subheader("1) Upload mammogram")
    uploaded_image = st.file_uploader(
        "Upload one image (.png, .jpg, .jpeg)", type=["png", "jpg", "jpeg"]
    )

    if uploaded_image is not None:
        img = Image.open(uploaded_image).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)
        if st.button("Run prediction"):
            st.success("Prediction result will be displayed here.")
    else:
        st.info("No image uploaded yet.")

# Chatbot section
with col2:
    st.subheader("2) Chatbot — Ask breast cancer questions")
    user_q = st.text_input(
        "Ask a question (e.g. 'What does a BI-RADS 4 mean?')"
    )
    if st.button("Ask") and user_q.strip() != "":
        st.success("Answer will be displayed here.")

    st.markdown("---")
    st.info("Chat history will appear here.")
