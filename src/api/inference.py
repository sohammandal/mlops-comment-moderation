# inference.py
import os
import torch
import joblib
import boto3
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .config import MODEL_BACKEND, MODEL_NAME, S3_MODEL_PATH, THRESHOLD


class HFToxicCommentModel:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.labels = self.model.config.id2label

    def _postprocess(self, logits):
        probs = torch.sigmoid(logits).tolist()
        if isinstance(probs[0], list):
            out = []
            for row in probs:
                label_probs = {self.labels[i]: float(row[i]) for i in range(len(row))}
                moderation_label = 1 if any(p > THRESHOLD for p in row) else 0
                out.append((moderation_label, label_probs))
            return out
        else:
            label_probs = {self.labels[i]: float(probs[i]) for i in range(len(probs))}
            moderation_label = 1 if any(p > THRESHOLD for p in probs) else 0
            return moderation_label, label_probs

    def predict(self, text: str):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._postprocess(outputs.logits.squeeze(0))

    def predict_batch(self, texts: list[str]):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._postprocess(outputs.logits)


class SklearnToxicCommentModel:
    def __init__(self):
        local_path = "/tmp/latest.joblib"
        if not os.path.exists(local_path):
            s3 = boto3.client("s3")
            bucket, key = S3_MODEL_PATH.replace("s3://", "").split("/", 1)
            s3.download_file(bucket, key, local_path)
        self.model = joblib.load(local_path)
        # Map sklearn classes to human-readable labels
        self.label_map = {0: "clean", 1: "toxic"}

    def predict(self, text: str):
        probs = self.model.predict_proba([text])[0]
        label_probs = {
            self.label_map[self.model.classes_[i]]: float(probs[i])
            for i in range(len(self.model.classes_))
        }
        moderation_label = 1 if label_probs["toxic"] > THRESHOLD else 0
        return moderation_label, label_probs

    def predict_batch(self, texts: list[str]):
        prob_matrix = self.model.predict_proba(texts)
        out = []
        for row in prob_matrix:
            label_probs = {
                self.label_map[self.model.classes_[i]]: float(row[i])
                for i in range(len(self.model.classes_))
            }
            moderation_label = 1 if label_probs["toxic"] > THRESHOLD else 0
            out.append((moderation_label, label_probs))
        return out


def get_model():
    if MODEL_BACKEND == "hf":
        return HFToxicCommentModel()
    elif MODEL_BACKEND == "s3":
        return SklearnToxicCommentModel()
    else:
        raise ValueError(f"Unknown MODEL_BACKEND={MODEL_BACKEND}")


# Singleton
model_instance = get_model()
