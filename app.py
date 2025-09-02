from fastapi import FastAPI
from pydantic import BaseModel
from optimum.pipelines import pipeline as ort_pipe

# Modeli yükle
model = "cardiffnlp/twitter-roberta-base-sentiment-latest"
pipe = ort_pipe("text-classification", model=model)

# FastAPI app
app = FastAPI(title="Sentiment Analysis API")

# Request body
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    result = pipe(request.text)
    return result

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}
