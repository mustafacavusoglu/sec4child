from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging

# Logging ekle
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable
pipe = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global pipe
    logger.info("Loading model...")
    try:
        from optimum.pipelines import pipeline as ort_pipe
        pipe = ort_pipe("text-classification", "cardiffnlp/twitter-roberta-base-sentiment-latest")
        logger.info("Model loaded successfully!")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    yield  # Bu satır startup ve shutdown arasındaki ayrım
    
    # Shutdown (isteğe bağlı)
    logger.info("Shutting down...")

# FastAPI app'i lifespan ile oluştur
app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)

# Request body
class TextRequest(BaseModel):
    text: str

@app.post("/predict")
def predict(request: TextRequest):
    if pipe is None:
        return {"error": "Model not loaded"}
    result = pipe(request.text)
    return result

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": pipe is not None}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Sentiment Analysis API"}
