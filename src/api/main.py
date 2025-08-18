from fastapi import FastAPI

from .config import MODEL_BACKEND
from .models import (
    CommentRequest,
    PredictionResponse,
    BatchCommentRequest,
    BatchPredictionResponse,
    BatchPredictionItem,
)
from .inference import model_instance

app = FastAPI(
    title="Comment Moderation API",
    description="API for real-time comment moderation with MLOps support",
    version="0.1.0",
)


@app.get("/health")
def health():
    """
    Health check endpoint to verify API is running.
    """
    return {
        "status": "ok",
        "message": "API is healthy",
        "backend": MODEL_BACKEND,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(req: CommentRequest):
    """
    Accepts a comment and returns a moderation prediction.
    """
    moderation_label, probs = model_instance.predict(req.text)
    return PredictionResponse(moderation_label=moderation_label, toxic_probs=probs)


@app.get("/")
def root():
    """
    Root endpoint (useful for quick browser checks).
    """
    return {"message": "Welcome to the Comment Moderation API!"}


def _predict_batch_impl(texts: list[str]) -> BatchPredictionResponse:
    results = model_instance.predict_batch(texts)
    items = [
        BatchPredictionItem(moderation_label=ml, toxic_probs=tp) for ml, tp in results
    ]
    return BatchPredictionResponse(predictions=items)


@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_batch(req: BatchCommentRequest):
    return _predict_batch_impl(req.texts)
