from fastapi import FastAPI
from .models import CommentRequest, PredictionResponse
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
    return {"status": "ok", "message": "API is healthy"}


@app.post("/predict")
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
