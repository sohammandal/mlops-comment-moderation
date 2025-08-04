from fastapi import FastAPI
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Comment Moderation API",
    description="API for real-time comment moderation with MLOps support",
    version="0.1.0",
)


# Health check endpoint
@app.get("/health")
def health():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "ok", "message": "API is healthy"}


# Request schema for predictions
class CommentRequest(BaseModel):
    text: str


# Dummy predict endpoint
@app.post("/predict")
def predict(req: CommentRequest):
    """
    Accepts a comment and returns a dummy moderation prediction.
    Replace with real model inference later.
    """
    dummy_label = 0  # 0 = clean, 1 = flagged
    dummy_confidence = 0.95

    return {
        "comment": req.text,
        "moderation_label": dummy_label,
        "confidence": dummy_confidence,
    }


# Root endpoint
@app.get("/")
def root():
    """
    Root endpoint (useful for quick browser checks).
    """
    return {"message": "Welcome to the Comment Moderation API!"}
