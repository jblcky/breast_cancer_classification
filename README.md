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


## 🏗️ Project Architecture

The system is split into **two main components**:

1. **Frontend (Streamlit)**
   - User-facing interface for:
     - Uploading mammogram images
     - Chatting with the breast cancer assistant
   - Communicates with backend via REST API.

2. **Backend (FastAPI)**
   - Hosts machine learning model (ResNet50).
   - Provides RAG-powered Q&A via vectorstore + OpenAI embeddings.
   - Deployed as a container on **Google Cloud Run**.

### 🔄 Workflow

1. **User uploads an image** → Streamlit sends request to FastAPI `/predict-image`.
2. **ResNet50 model** classifies as *Benign* or *Malignant* → response returned to frontend.
3. **User asks a question** → Streamlit sends request to FastAPI `/ask-question`.
4. **RAG pipeline** retrieves context from vectorstore and generates an answer.
5. **Answer returned** to frontend and displayed in chat interface.


## 🛠️ Tech Stack

- **Frontend**
  - [Streamlit](https://streamlit.io/) → Interactive chat UI & image upload
  - Custom CSS for Gemini-style light/dark theme

- **Backend**
  - [FastAPI](https://fastapi.tiangolo.com/) → REST API for predictions & Q&A
  - [Uvicorn](https://www.uvicorn.org/) → ASGI server

- **Machine Learning**
  - [TensorFlow / Keras](https://www.tensorflow.org/) → ResNet50 model for binary classification
  - [OpenAI Embeddings](https://platform.openai.com/docs/guides/embeddings) → Convert text to vectors
  - [FAISS](https://faiss.ai/) → Vector similarity search for RAG

- **Infrastructure**
  - [Docker](https://www.docker.com/) → Containerization
  - [Google Cloud Run](https://cloud.google.com/run) → Serverless deployment of backend

- **Other Libraries**
  - `requests` → API calls from Streamlit to FastAPI
  - `Pydantic` → Data validation in FastAPI


## ⚙️ Setup & Installation

### 1. Clone Repository
```bash
git clone https://github.com/your-username/breast-cancer-chatbot.git
cd breast-cancer-chatbot


### 2. Create Virtual Environment

It is recommended to use a virtual environment to isolate project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate it (Linux/Mac)
source venv/bin/activate

# Activate it (Windows)
venv\Scripts\activate


### 3. Install Dependencies

Install all required packages listed in `requirements.txt`.

```bash
pip install --upgrade pip
pip install -r requirements.txt


### 📦 Key Dependencies

- **TensorFlow** → Deep learning framework used to train and run the ResNet50 model for mammogram classification.
- **FastAPI** → High-performance Python web framework for serving backend APIs.
- **Uvicorn** → ASGI server to run FastAPI in production.
- **Streamlit** → Frontend web application framework for building the chatbot interface.
- **LangChain** → Framework for Retrieval-Augmented Generation (RAG) pipelines.
- **FAISS** → Facebook AI Similarity Search, used as the vector database for efficient document retrieval.
- **OpenAI** → Provides embeddings and LLMs for question answering.
- **Python-dotenv** → For managing environment variables (e.g., API keys).


### 4. Setup Environment Variables

Create a `.env` file in the project root directory to store your secret keys.
This file should **not** be committed to GitHub for security reasons.

Example `.env`:

```env
OPENAI_API_KEY=your_openai_api_key_here



### 5. Project Structure

Organized layout of the project:

breast-cancer-chatbot/
│
├── app/ # Backend FastAPI service
│ ├── main.py # API entry point (/predict-image, /ask-question)
│ ├── model/ # Saved ResNet50 model weights
│ ├── vectorstore/ # FAISS vector database for RAG
│ └── utils/ # Helper functions (preprocessing, inference, etc.)
│
├── streamlit_app/ # Frontend Streamlit application
│ └── app.py # Streamlit chatbot UI
│
├── requirements.txt # Python dependencies
├── Dockerfile # Docker image for backend
├── .env # Environment variables (OpenAI API key)
└── README.md # Project documentation


**Notes:**
- `app/` contains all backend logic and model/vectorstore resources.
- `streamlit_app/` contains the frontend UI code.
- `.env` stores sensitive information and should **not** be committed to Git.
- `Dockerfile` is used to containerize the backend for deployment on Google Cloud Run.


### 6. Running Locally

Follow these steps to run the backend and frontend on your local machine:

---

#### 6.1 Start the Backend (FastAPI)

```bash
# Go to backend folder
cd app

# Start FastAPI server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload


#### 6.2 Start the Frontend (Streamlit)

```bash
# Navigate to the Streamlit app directory
cd streamlit_app

# Run the Streamlit UI
streamlit run app.py


#### 6.3 Test the Application

Once both backend and frontend are running locally:

1. **Image Prediction**
   - Upload a mammogram image (`.png`, `.jpg`, `.jpeg`) via the Streamlit UI.
   - The app will display:
     - Prediction: **Benign** or **Malignant**
     - Confidence score (%)

2. **RAG Question Answering**
   - Type a natural language question about breast cancer in the chat input.
   - The app sends the question to the `/ask-question` endpoint.
   - The response is retrieved from the FAISS vectorstore and OpenAI embeddings, and displayed in the chat.

3. **Verify**
   - Check that predictions match expected results for test images.
   - Confirm that questions receive context-aware answers.


### 7. Docker & Cloud Run Deployment

You can containerize the backend and deploy it to Google Cloud Run for a scalable, serverless deployment.

---

#### 7.1 Build Docker Image

```bash
# From project root
docker build -t mammogram-chatbot .


#### 7.2 Test Docker Image Locally

After building the Docker image, you can run it locally to ensure everything works:

```bash
# Run the Docker container
docker run -p 8000:8000 mammogram-chatbot


#### 7.3 Push Docker Image to Google Artifact Registry

Once your Docker image works locally, push it to Google Artifact Registry for deployment:

```bash
# Tag the Docker image for your Artifact Registry
docker tag mammogram-chatbot \
asia-southeast1-docker.pkg.dev/PROJECT_ID/breast-cancer-classification/mammogram-chatbot

# Push the Docker image
docker push asia-southeast1-docker.pkg.dev/PROJECT_ID/breast-cancer-classification/mammogram-chatbot


#### 7.4 Deploy to Cloud Run

Deploy the Docker image to Google Cloud Run for a scalable, serverless backend:

```bash
gcloud run deploy mammogram-chatbot \
  --image asia-southeast1-docker.pkg.dev/PROJECT_ID/breast-cancer-classification/mammogram-chatbot \
  --platform managed \
  --region asia-southeast1 \
  --memory 2Gi \
  --allow-unauthenticated


#### 7.5 Connect Streamlit Frontend

After deploying the backend to Cloud Run, update your Streamlit frontend to use the live API:

1. Open `streamlit_app/app.py`.
2. Update the `API_URL` to point to the Cloud Run endpoint:

```python
API_URL = "https://mammogram-chatbot-xxxxxx-uc.a.run.app"



### 8. Usage Examples

Once both backend and frontend are running (locally or deployed), you can interact with the chatbot as follows:

---

#### 8.1 Image Classification

1. Open the Streamlit UI (`http://localhost:8501` or deployed URL).
2. Upload a mammogram image (`.png`, `.jpg`, `.jpeg`).
3. The app will display:
   - Prediction: **Benign** or **Malignant**
   - Confidence score (%)

Example response: 📷 Prediction: Malignant (Confidence: 92%)


---

#### 8.2 RAG Question Answering

1. Type a natural language question in the chat input, e.g.: What are the early signs of breast cancer?
2. The question is sent to the `/ask-question` endpoint.
3. The backend retrieves relevant context from the FAISS vectorstore and OpenAI embeddings, then generates an answer.

Example response: Early signs of breast cancer may include a lump in the breast, change in breast shape, dimpling of the skin, or nipple discharge. Consult a healthcare professional for evaluation.


#### 8.3 Combining Features

- Users can upload images and ask questions in the same session.
- Chat history and responses are displayed persistently in the Streamlit interface.
- Light/dark theme toggle is available in the header for a better user experience.


### 9. Notes & Best Practices

#### 9.1 Environment Variables
- Keep sensitive keys like `OPENAI_API_KEY` in a `.env` file or Cloud Run environment variables.
- **Do not commit `.env` to version control.**

#### 9.2 Model & Vectorstore
- Store your ResNet50 model weights in `app/model/`.
- Store FAISS vectorstore files in `app/vectorstore/`.
- Ensure the backend has read access to both directories.

#### 9.3 Docker & Deployment
- Use Docker to isolate dependencies and ensure reproducibility.
- Allocate sufficient memory in Cloud Run (at least `2Gi`) for image inference.
- Use `--allow-unauthenticated` only if public access is intended.

#### 9.4 Frontend
- Update `API_URL` in `streamlit_app/app.py` to point to the deployed backend.
- Streamlit can be deployed on [Streamlit Cloud](https://streamlit.io/cloud) for a public interface.

#### 9.5 Testing & Maintenance
- Test both endpoints (`/predict-image` and `/ask-question`) after deployment.
- Monitor Cloud Run logs for errors and performance issues.
- Retrain or update the model/vectorstore periodically as new data becomes available.


### 10. License & References

#### 10.1 License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

#### 10.2 References
- [TensorFlow](https://www.tensorflow.org/) – Deep learning framework used for ResNet50.
- [FastAPI](https://fastapi.tiangolo.com/) – Backend API framework.
- [Streamlit](https://streamlit.io/) – Frontend web application framework.
- [LangChain](https://www.langchain.com/) – RAG pipelines and embeddings management.
- [FAISS](https://github.com/facebookresearch/faiss) – Vector similarity search for retrieval.
- [OpenAI](https://openai.com/) – LLMs and embeddings for question answering.

#### 10.3 Acknowledgements
- Inspired by modern AI-driven healthcare assistants combining image analysis and retrieval-augmented generation.
