from fastapi import FastAPI, UploadFile, File
from app.predict import predict_image
from app.model import model
from app.preprocessor import preprocess_images_resnet50_bytes

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

@app.post("/ask-question")
async def ask_question_endpoint(question: str):
    answer = answer_question(question)
    return {"answer": answer}
