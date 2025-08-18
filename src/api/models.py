from pydantic import BaseModel
from typing import Dict, List


class CommentRequest(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    moderation_label: int
    toxic_probs: Dict[str, float]


class BatchCommentRequest(BaseModel):
    texts: List[str]


class BatchPredictionItem(BaseModel):
    moderation_label: int
    toxic_probs: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]
