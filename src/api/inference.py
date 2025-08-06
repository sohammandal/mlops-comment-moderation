import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import MODEL_NAME, THRESHOLD


class ToxicCommentModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()  # Inference mode
        self.labels = self.model.config.id2label

    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).squeeze().tolist()

        # Map labels to probabilities
        label_probs = {self.labels[i]: float(probs[i]) for i in range(len(probs))}
        moderation_label = 1 if any(p > THRESHOLD for p in probs) else 0

        return moderation_label, label_probs


# Instantiate singleton model at import
model_instance = ToxicCommentModel()
