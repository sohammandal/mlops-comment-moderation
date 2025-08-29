# config.py
MODEL_BACKEND = "s3"  # options: "hf" or "s3"

MODEL_NAME = "unitary/toxic-bert"  # used if backend == "hf"
THRESHOLD = 0.5

S3_MODEL_PATH = (
    "s3://mlops-comment-artifacts/models/latest.joblib"  # used if backend == "s3"
)
POS_LABEL_NAME = "toxic"  # only relevant for sklearn model
