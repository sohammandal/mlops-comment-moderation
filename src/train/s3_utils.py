import os
import boto3
from dotenv import load_dotenv

# Load AWS_PROFILE and other vars from .env
load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE", "default")


def _get_s3_client():
    session = boto3.Session(profile_name=AWS_PROFILE)
    return session.client("s3")


def upload_file(local_path: str, bucket: str, key: str) -> str:
    s3 = _get_s3_client()
    s3.upload_file(local_path, bucket, key)
    return f"s3://{bucket}/{key}"


def copy_to_latest(
    bucket: str, key: str, latest_key: str = "models/latest.joblib"
) -> str:
    s3 = _get_s3_client()
    src = {"Bucket": bucket, "Key": key}
    s3.copy_object(Bucket=bucket, CopySource=src, Key=latest_key)
    return f"s3://{bucket}/{latest_key}"
