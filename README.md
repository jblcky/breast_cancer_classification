# 🩺 Breast Cancer Chatbot with ResNet50 & RAG

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-frontend-orange.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-brightgreen.svg)](LICENSE)

A **web-based breast cancer assistant** that combines **mammogram image classification** using a **ResNet50 model** and **retrieval-augmented generation (RAG)** using **OpenAI embeddings** for question answering.

---

## Features

- **Mammogram Image Classification**
  - Upload images (`png`, `jpg`, `jpeg`) to get prediction: *Benign* or *Malignant*.
  - Returns prediction confidence.

- **RAG Question Answering**
  - Ask natural language questions about breast cancer.
  - Uses FAISS vectorstore and OpenAI embeddings to retrieve and answer queries.

- **Frontend**
  - Streamlit-based Gemini-style chat interface.
  - Light/dark theme toggle.
  - Displays chat history and image upload responses.

- **Backend**
  - FastAPI served on Google Cloud Run.
  - TensorFlow for ResNet50 inference.
  - Vectorstore loaded for RAG.
