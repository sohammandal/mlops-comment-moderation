from pydantic import BaseModel
from typing import Dict


class CommentRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    moderation_label: int
    toxic_probs: Dict[str, float]
