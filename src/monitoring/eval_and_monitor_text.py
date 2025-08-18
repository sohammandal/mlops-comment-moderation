import os
import time
import warnings
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.metrics import f1_score, classification_report

# Evidently v0.6+ API
from evidently import Report, Dataset, DataDefinition, BinaryClassification
from evidently.presets import DataDriftPreset, ClassificationPreset

# Optional Evidently Cloud
try:
    from evidently.ui.workspace import CloudWorkspace

    EVIDENTLY_CLOUD_AVAILABLE = True
except Exception:
    EVIDENTLY_CLOUD_AVAILABLE = False

warnings.filterwarnings("ignore")


# ----------------------------
# Config
# ----------------------------
load_dotenv()

# You can set API_BASE_URL instead of API_URL if you prefer.
API_URL = os.getenv("API_URL", "http://localhost:8000/predict").strip()
if API_URL.endswith("/predict"):
    API_BASE = API_URL.rsplit("/", 1)[0]
else:
    API_BASE = API_URL  # assume it's already a base

API_SINGLE_URL = os.getenv("API_SINGLE_URL", f"{API_BASE}/predict").strip()
API_BATCH_URL = os.getenv("API_BATCH_URL", f"{API_BASE}/predict_batch").strip()

ASSETS_DIR = os.getenv("ASSETS_DIR", "assets").strip()
ORIG_CSV = os.getenv(
    "ORIG_TEST_CSV", os.path.join(ASSETS_DIR, "comments_test.csv")
).strip()
CHGD_CSV = os.getenv(
    "CHGD_TEST_CSV", os.path.join(ASSETS_DIR, "comments_test_v2.csv")
).strip()

# Limit calls for demo. Set SAMPLE_N=None to use full data.
# SAMPLE_N = int(os.getenv("SAMPLE_N", "300")) if os.getenv("SAMPLE_N", "300") != "None" else None
SAMPLE_N = None
TIMEOUT = float(os.getenv("API_TIMEOUT", "30"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# Evidently Cloud (optional)
EVIDENTLY_API_KEY = os.getenv("EVIDENTLY_API_KEY")
EVIDENTLY_PROJECT_ID = os.getenv("EVIDENTLY_PROJECT_ID")

SAVE_DIR = ASSETS_DIR
os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------
# Helpers
# ----------------------------
def load_test(path: str, sample_n: int | None = SAMPLE_N) -> pd.DataFrame:
    df = pd.read_csv(path)
    needed = {"id", "comment_text", "moderation_label"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"CSV {path} missing columns: {missing}")
    if sample_n and len(df) > sample_n:
        df = df.sample(sample_n, random_state=42).reset_index(drop=True)
    return df[["id", "comment_text", "moderation_label"]].copy()


def _post_single(text: str) -> Dict[str, Any]:
    r = requests.post(API_SINGLE_URL, json={"text": str(text)}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _post_batch(texts: List[str]) -> List[Dict[str, Any]]:
    r = requests.post(
        API_BATCH_URL, json={"texts": [str(t) for t in texts]}, timeout=TIMEOUT
    )
    r.raise_for_status()
    payload = r.json()
    # Expect either a list directly or {"predictions": [...]}
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "predictions" in payload:
        return payload["predictions"]
    raise ValueError("Unexpected batch response format")


def call_api_with_fallback(texts: List[str]) -> List[Dict[str, Any]]:
    """Try batch; if not available, fall back to single calls."""
    try:
        out: List[Dict[str, Any]] = []
        for i in range(0, len(texts), BATCH_SIZE):
            chunk = texts[i : i + BATCH_SIZE]
            out.extend(_post_batch(chunk))
            if (i // BATCH_SIZE + 1) % 5 == 0:
                print(f"  scored {i + len(chunk)}...", flush=True)
        return out
    except Exception as e:
        print(f"Batch endpoint failed, falling back to single calls: {e}")
        out: List[Dict[str, Any]] = []
        for i, t in enumerate(texts, 1):
            tries = 0
            while True:
                try:
                    out.append(_post_single(t))
                    break
                except Exception:
                    tries += 1
                    if tries >= 2:
                        raise
                    time.sleep(0.4)
            if i % 50 == 0:
                print(f"  scored {i}...", flush=True)
        return out


def evaluate(
    df: pd.DataFrame, preds: List[Dict[str, Any]]
) -> Tuple[float, float, pd.DataFrame]:
    if len(df) != len(preds):
        raise ValueError("Length mismatch between dataframe and predictions")

    y_pred = [int(p.get("moderation_label", 0)) for p in preds]
    max_prob = []
    for p in preds:
        probs = p.get("toxic_probs", {})
        max_prob.append(
            float(max(probs.values()))
            if isinstance(probs, dict) and probs
            else float("nan")
        )

    out = df.copy()
    out["prediction"] = y_pred
    out["score_max_prob"] = max_prob

    y_true = out["moderation_label"].astype(int).values
    y_hat = out["prediction"].astype(int).values

    f1_w = f1_score(np.asarray(y_true), np.asarray(y_hat), average="weighted")
    f1_m = f1_score(np.asarray(y_true), np.asarray(y_hat), average="macro")
    print(classification_report(np.asarray(y_true), np.asarray(y_hat), digits=4))
    return float(f1_w), float(f1_m), out


def build_and_save_reports(
    ref_df: pd.DataFrame,
    cur_df: pd.DataFrame,
    report_name_slug: str,
    push_cloud: bool = True,
):
    required = {"id", "comment_text", "moderation_label", "prediction"}
    miss_ref = required - set(ref_df.columns)
    miss_cur = required - set(cur_df.columns)
    if miss_ref or miss_cur:
        raise ValueError(
            f"Missing columns. Ref missing: {miss_ref}, Cur missing: {miss_cur}"
        )

    def _run_report(_ref: pd.DataFrame, _cur: pd.DataFrame, use_strings: bool):
        ref = _ref.copy()
        cur = _cur.copy()

        if use_strings:
            ref["moderation_label"] = ref["moderation_label"].astype(str)
            ref["prediction"] = ref["prediction"].astype(str)
            cur["moderation_label"] = cur["moderation_label"].astype(str)
            cur["prediction"] = cur["prediction"].astype(str)

            data_def = DataDefinition(
                id_column="id",
                text_columns=["comment_text"],
                classification=[
                    BinaryClassification(
                        target="moderation_label",
                        prediction_labels="prediction",
                        pos_label="1",
                        labels=["0", "1"],
                    )
                ],
            )
        else:
            ref["moderation_label"] = ref["moderation_label"].astype(int)
            ref["prediction"] = ref["prediction"].astype(int)
            cur["moderation_label"] = cur["moderation_label"].astype(int)
            cur["prediction"] = cur["prediction"].astype(int)

            data_def = DataDefinition(
                id_column="id",
                text_columns=["comment_text"],
                classification=[
                    BinaryClassification(
                        target="moderation_label",
                        prediction_labels="prediction",
                        pos_label=1,
                        # omit labels to let Evidently infer ints cleanly
                    )
                ],
            )

        cols = ["id", "comment_text", "moderation_label", "prediction"]
        ref_ds = Dataset.from_pandas(ref[cols], data_definition=data_def)
        cur_ds = Dataset.from_pandas(cur[cols], data_definition=data_def)

        report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])
        return report, report.run(reference_data=ref_ds, current_data=cur_ds)

    # Try numeric first, then fallback to string labels if Evidently complains
    try:
        report, eval_result = _run_report(ref_df, cur_df, use_strings=False)
    except KeyError:
        # mismatch of pos_label vs matrix labels in this build; retry with strings
        report, eval_result = _run_report(ref_df, cur_df, use_strings=True)

    # Save the report
    html_path = os.path.join(SAVE_DIR, f"{report_name_slug}.html")
    eval_result.save_html(html_path)
    print(f"✓ Saved HTML report to: {html_path}")

    # Cloud upload
    if (
        push_cloud
        and EVIDENTLY_CLOUD_AVAILABLE
        and EVIDENTLY_API_KEY
        and EVIDENTLY_PROJECT_ID
    ):
        try:
            ws = CloudWorkspace(
                token=EVIDENTLY_API_KEY, url="https://app.evidently.cloud"
            )
            project = ws.get_project(EVIDENTLY_PROJECT_ID)
            ws.add_run(project.id, eval_result, include_data=False)
            print("✓ Uploaded run to Evidently Cloud.")
        except Exception as e:
            print(f"✗ Could not upload to Evidently Cloud: {e}")

    return eval_result


def main():
    print("Loading test sets...")
    df_ref = load_test(ORIG_CSV, SAMPLE_N)
    df_cur = load_test(CHGD_CSV, SAMPLE_N)

    print(f"Reference rows: {len(df_ref)}")
    print(f"Changed rows:   {len(df_cur)}")

    # Align on common ids so we compare like for like
    common = sorted(set(df_ref["id"]).intersection(set(df_cur["id"])))
    if len(common) >= 5:
        df_ref = (
            df_ref[df_ref["id"].isin(common)].sort_values("id").reset_index(drop=True)
        )
        df_cur = (
            df_cur[df_cur["id"].isin(common)].sort_values("id").reset_index(drop=True)
        )
        print(f"Aligned on {len(common)} common ids for apples-to-apples comparison.")

    print("\nScoring Reference set...")
    ref_preds = call_api_with_fallback(df_ref["comment_text"].tolist())
    print("Scoring Changed set...")
    cur_preds = call_api_with_fallback(df_cur["comment_text"].tolist())

    print("\nMetrics on Reference set")
    f1w_ref, f1m_ref, ref_eval = evaluate(df_ref, ref_preds)
    print(f"F1 weighted: {f1w_ref:.4f} | F1 macro: {f1m_ref:.4f}")

    print("\nMetrics on Changed set")
    f1w_cur, f1m_cur, cur_eval = evaluate(df_cur, cur_preds)
    print(f"F1 weighted: {f1w_cur:.4f} | F1 macro: {f1m_cur:.4f}")

    # Save per-row predictions
    ref_csv = os.path.join(SAVE_DIR, "preds_reference.csv")
    cur_csv = os.path.join(SAVE_DIR, "preds_changed.csv")
    ref_eval.to_csv(ref_csv, index=False)
    cur_eval.to_csv(cur_csv, index=False)
    print(f"Saved per-row predictions:\n  {ref_csv}\n  {cur_csv}")

    # Evidently report
    build_and_save_reports(
        ref_eval,
        cur_eval,
        report_name_slug="evidently_text_moderation_ref_vs_changed",
        push_cloud=True,
    )

    # Slide-friendly summary
    delta_w = f1w_cur - f1w_ref
    delta_m = f1m_cur - f1m_ref
    print("\n=== Summary ===")
    print(
        f"Reference F1 weighted: {f1w_ref:.4f} | Changed: {f1w_cur:.4f} | Delta: {delta_w:+.4f}"
    )
    print(
        f"Reference F1 macro:    {f1m_ref:.4f} | Changed: {f1m_cur:.4f} | Delta: {delta_m:+.4f}"
    )
    print("Open the HTML report to point at drift and classification sections.")


if __name__ == "__main__":
    main()
