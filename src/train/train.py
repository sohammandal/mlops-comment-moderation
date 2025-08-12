import os
import argparse
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from datetime import datetime
from flaml.automl.automl import AutoML
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from src.train.s3_utils import upload_file, copy_to_latest

# Load .env for AWS_PROFILE etc
from dotenv import load_dotenv

load_dotenv()


def parse_args():
    p = argparse.ArgumentParser(
        description="Train toxic comment model with FLAML and upload to S3"
    )
    p.add_argument("--train-csv", default="data/comments_train.csv")
    p.add_argument("--test-csv", default="data/comments_test.csv")
    p.add_argument("--experiment", default="toxic_comment_flaml")
    p.add_argument("--time-budget", type=int, default=120)  # seconds
    p.add_argument("--max-features", type=int, default=300_000)
    p.add_argument(
        "--mlflow-uri",
        default=None,
        help="Optional MLflow tracking URI. If not set, uses local ./mlruns",
    )
    p.add_argument("--s3-bucket", required=True)
    p.add_argument(
        "--s3-key-prefix",
        default="models",
        help="Prefix in S3 to store model artifacts",
    )
    p.add_argument(
        "--upload-latest", action="store_true", help="Also copy to models/latest.joblib"
    )
    p.add_argument(
        "--upload-only",
        help="Path to an existing .joblib model to upload instead of training",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Upload-only mode
    if args.upload_only:
        local_model_path = args.upload_only
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model file not found: {local_model_path}")

        key = f"{args.s3_key_prefix}/{os.path.basename(local_model_path)}"
        s3_uri = upload_file(local_model_path, args.s3_bucket, key)
        print(f"Uploaded: {s3_uri}")

        latest_uri = None
        if args.upload_latest:
            latest_uri = copy_to_latest(args.s3_bucket, key)
            print(f"Also copied to: {latest_uri}")

        print("Upload-only mode complete.")
        return

    # MLflow setup
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    else:
        mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
    mlflow.set_experiment(args.experiment)

    # Load data
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)

    X_train = train_df["comment_text"]
    y_train = train_df["moderation_label"].astype(int)

    X_test = test_df["comment_text"]
    y_test = test_df["moderation_label"].astype(int)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        lowercase=True,
        max_features=args.max_features,
    )

    # FLAML on vectorized features
    automl = AutoML()
    settings = {
        "time_budget": args.time_budget,
        "metric": "f1",
        "task": "classification",
        "estimator_list": ["lrl2", "lgbm", "rf"],
        "eval_method": "cv",
        "n_splits": 3,
        "seed": 42,
        "log_file_name": "flaml_train.log",
    }

    print("Starting FLAML search...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    automl.fit(X_train_tfidf, y_train, **settings)
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")

    # Refit a clean sklearn Pipeline on raw text
    # Use the best estimator from AutoML inside a Pipeline with the vectorizer
    best_est = automl.model
    pipeline = Pipeline(
        [
            ("vectorizer", vectorizer),
            ("model", best_est),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    f1_w = f1_score(y_test, preds, average="weighted")
    f1_m = f1_score(y_test, preds, average="macro")
    print(f"F1 weighted: {f1_w:.4f} | F1 macro: {f1_m:.4f}")
    print(classification_report(y_test, preds))

    # Log to MLflow and save artifact
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    local_model_path = f"model_{automl.best_estimator}_{ts}.joblib"
    with mlflow.start_run(run_name=f"flaml_{automl.best_estimator}_{ts}"):
        mlflow.log_param("vectorizer", "TfidfVectorizer(char 3-5)")
        mlflow.log_param("best_estimator", automl.best_estimator)
        best_config = automl.best_config if automl.best_config is not None else {}
        mlflow.log_params({f"best_{k}": v for k, v in best_config.items()})
        mlflow.log_metric("f1_weighted", float(f1_w))
        mlflow.log_metric("f1_macro", float(f1_m))

        joblib.dump(pipeline, local_model_path)
        mlflow.log_artifact(local_model_path)

        # Upload to S3
        key = f"{args.s3_key_prefix}/{local_model_path}"
        s3_uri = upload_file(local_model_path, args.s3_bucket, key)
        print(f"Uploaded: {s3_uri}")

        latest_uri = None
        if args.upload_latest:
            latest_uri = copy_to_latest(args.s3_bucket, key)
            print(f"Also copied to: {latest_uri}")

        # Add useful artifacts as run tags
        mlflow.set_tag("s3_uri", s3_uri)
        if latest_uri:
            mlflow.set_tag("s3_latest_uri", latest_uri)

    print("Training complete.")


if __name__ == "__main__":
    main()
