from fastapi import FastAPI, UploadFile, File
from app.predict import predict_image
from app.model import model
from app.preprocessor import preprocess_images_resnet50_bytes
from app.rag import load_vectorstore, qa_chain
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to Breast Cancer Classification & Chatbot API"}


@app.post("/predict-image")
async def predict_image_endpoint(file: UploadFile = File(...)):

    # Preprocess uploaded image
    image_tensor = preprocess_images_resnet50_bytes(file.file)

    # Run prediction (synchronous)
    class_labels, confidence_percent = predict_image(model, image_tensor)

    return {
        "Prediction": class_labels,
        "Confidence": f"{confidence_percent:.0f}%"
    }

VECTORSTORE_PATH = 'app/vectorstore'
vectorstore = load_vectorstore(VECTORSTORE_PATH)
question_answer = qa_chain(vectorstore)

@app.post("/ask-question")
async def ask_question_endpoint(question: str):
    # pass input as dict
    result = question_answer.invoke({"input": question})

    # Get the final answer from the 'output' key
    return {"Answer": result['answer']}
