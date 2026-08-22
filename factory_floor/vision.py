import base64
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import torch
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from torchvision.models import ResNet18_Weights, resnet18

from factory_floor.config import PROJECT_ROOT, VISION_MODEL_DIR
from factory_floor.defect_dataset import COARSE_LABELS
from factory_floor.rag import get_llm

CLASSIFIER_PATH = VISION_MODEL_DIR / "classifier.joblib"
FEATURE_CACHE_PATH = VISION_MODEL_DIR / "feature_cache.joblib"

VISION_SYSTEM_PROMPT = """You are the defect-classification component of an industrial
maintenance copilot for electric motors and variable-frequency drives (VFDs).

You will be shown a close-up photograph of a small component (a cable, a metal nut, a
screw, or a transistor) that might be found on or near this equipment. Classify the
visible condition of the component into exactly one of these categories:
- good: no visible defect
- scratch: a scratch or scrape mark on the surface
- deformation: bent, warped, or flipped out of shape
- structural_damage: cut, cracked, broken, or otherwise structurally compromised
- contamination: discoloration or foreign material on the surface
- other_defect: a visible defect that does not fit the categories above (e.g. a missing
  or misplaced part)

Only use what is visibly in the photograph. Do not guess a specific fault code or model
number from this image — that is not what this photo is for."""


class DefectPrediction(BaseModel):
    label: str = Field(description=f"One of: {', '.join(COARSE_LABELS)}")
    confidence: float = Field(description="Self-reported confidence, 0-1 (not a calibrated probability)")
    reasoning: str = Field(description="One-sentence visual justification")


# --- Trained classifier path (main approach) -------------------------------------

@lru_cache(maxsize=1)
def get_feature_extractor():
    """Frozen, pretrained ResNet18 with its classification head removed — used only to
    turn an image into a 512-d feature vector, never fine-tuned (the bootcamp spec
    explicitly discourages fine-tuning)."""
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    return model, weights.transforms()


def extract_features(image_path, extractor=None) -> np.ndarray:
    from PIL import Image

    model, preprocess = extractor or get_feature_extractor()
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor)
    return features.squeeze(0).numpy()


def extract_features_batch(image_paths, extractor=None, cache_path=FEATURE_CACHE_PATH) -> np.ndarray:
    """Same as extract_features, but with an on-disk cache keyed by filepath — an image
    already seen in a previous run is never re-processed, same spirit as the persistent
    Chroma store never re-embedding the same manual chunk twice."""
    extractor = extractor or get_feature_extractor()
    cache = {}
    if cache_path and Path(cache_path).exists():
        cache = joblib.load(cache_path)

    updated = False
    features = []
    for image_path in image_paths:
        key = str(image_path)
        if key not in cache:
            cache[key] = extract_features(image_path, extractor)
            updated = True
        features.append(cache[key])

    if cache_path and updated:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(cache, cache_path)
    return np.array(features)


def train_classifier(features: np.ndarray, labels: list) -> LogisticRegression:
    """class_weight='balanced' matters here: 'good' images outnumber any single defect
    class roughly 10-to-1 in the curated dataset, so an unweighted fit would just learn
    to predict 'good' most of the time and still score a deceptively high accuracy."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(features, labels)
    return clf


def save_classifier(clf: LogisticRegression, path=CLASSIFIER_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"classifier": clf, "label_list": list(clf.classes_)}, path)


def load_classifier(path=CLASSIFIER_PATH):
    data = joblib.load(path)
    return data["classifier"], data["label_list"]


def classify_defect_trained(image_path, clf: LogisticRegression, label_list=None, extractor=None) -> dict:
    features = extract_features(image_path, extractor).reshape(1, -1)
    probs = clf.predict_proba(features)[0]
    classes = list(clf.classes_)
    best_idx = int(np.argmax(probs))
    predicted_label = classes[best_idx]
    return {
        "image_path": str(image_path),
        "predicted_label": predicted_label,
        "confidence": float(probs[best_idx]),
        "probabilities": {label: float(p) for label, p in zip(classes, probs)},
        "is_defective": predicted_label != "good",
        "method": "trained_classifier",
    }


# --- Zero-shot LLM path (baseline) ------------------------------------------------

def _encode_image_base64(image) -> str:
    """Accepts a filesystem path (str/Path, used by evaluate_classifier over the
    manifest) or an in-memory file-like object / raw bytes (used by app.py for a
    Streamlit-uploaded photo, which never touches disk)."""
    if hasattr(image, "read"):
        return base64.b64encode(image.read()).decode("utf-8")
    if isinstance(image, (bytes, bytearray)):
        return base64.b64encode(image).decode("utf-8")
    with open(image, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_defect_zero_shot(image_path, llm=None, labels=None) -> dict:
    """Baseline for the trained classifier above — no training data needed, but the
    'confidence' field is the model's own self-reported estimate, not a calibrated
    probability like classify_defect_trained's, so the two should never be compared
    directly as if they were the same kind of number."""
    llm = llm or get_llm()
    labels = labels or COARSE_LABELS
    structured_llm = llm.with_structured_output(DefectPrediction)
    b64 = _encode_image_base64(image_path)
    messages = [
        SystemMessage(VISION_SYSTEM_PROMPT),
        HumanMessage(
            content=[
                {"type": "text", "text": "Classify the component's condition in this photo."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        ),
    ]
    prediction = structured_llm.invoke(messages)
    predicted_label = prediction.label if prediction.label in labels else "other_defect"
    return {
        "image_path": str(image_path),
        "predicted_label": predicted_label,
        "confidence": prediction.confidence,
        "probabilities": None,
        "is_defective": predicted_label != "good",
        "method": "zero_shot_llm",
        "reasoning": prediction.reasoning,
    }


# --- Recommended actions (turns a label into something the operator can act on) --

ACTION_SYSTEM_PROMPT = """You are the recommended-actions component of an industrial
maintenance copilot for electric motors and variable-frequency drives (VFDs).

You will be shown a close-up photo of a component (a cable, metal nut, screw, or
transistor) found on or near this equipment. A defect classifier has already identified
its condition as: {predicted_label}.

Give the maintenance operator concrete next steps: what to inspect further, whether the
part should be cleaned, tightened, re-torqued, repaired, or replaced, and any safety
precaution that applies before touching the equipment (e.g. lockout/tagout, power
isolation) — safety precautions first, before any repair step.

This recommendation is based only on this photo and the defect type above — you were
not given this equipment's manuals or safety data sheets to cite, so do not invent a
specific fault code, parameter number, or manual reference. Say explicitly that this is
general guidance, not sourced from the equipment's documentation, and recommend
cross-checking the manual or escalating to a qualified technician if the defect looks
severe or this guidance is insufficient."""


def recommend_actions(image, predicted_label: str, llm=None, language: str = "English") -> dict:
    """Turns a bare defect label into something the operator can actually act on —
    without this, classify_defect_* only tells the operator *what* is wrong, never
    *what to do* about it. Deliberately not grounded in the manuals (no retrieval here,
    same scope boundary as the rest of the vision path) — that fusion is the future
    Orchestrator Agent's job, not this function's."""
    llm = llm or get_llm()
    b64 = _encode_image_base64(image)
    messages = [
        SystemMessage(ACTION_SYSTEM_PROMPT.format(predicted_label=predicted_label)),
        HumanMessage(
            content=[
                {"type": "text", "text": f"Respond in {language}. What should the operator do next?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        ),
    ]
    response = llm.invoke(messages)
    return {
        "predicted_label": predicted_label,
        "actions": response.content,
        "grounded_in_manual": False,
    }


# --- Single entry point, in the spirit of rag.ask() -------------------------------

def classify_defect(image_path, method="trained_classifier", clf=None, label_list=None, llm=None) -> dict:
    if method == "trained_classifier":
        if clf is None:
            clf, label_list = load_classifier()
        return classify_defect_trained(image_path, clf, label_list)
    if method == "zero_shot_llm":
        return classify_defect_zero_shot(image_path, llm=llm)
    raise ValueError(f"Unknown method: {method!r} (expected 'trained_classifier' or 'zero_shot_llm')")


# --- Evaluation (used by notebook 06 for the required accuracy + benchmarking) ----

def evaluate_classifier(predict_fn, manifest_rows: list) -> dict:
    """Runs predict_fn (either classify_defect_trained or classify_defect_zero_shot,
    partially applied) over every row of a manifest split and scores it against the
    real coarse_label — works for either method since both return the same dict shape."""
    y_true = [row["coarse_label"] for row in manifest_rows]
    predictions = []
    for row in manifest_rows:
        image_path = PROJECT_ROOT / row["filepath"]
        predictions.append(predict_fn(image_path))
    y_pred = [p["predicted_label"] for p in predictions]
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "n_samples": len(manifest_rows),
        "predictions": predictions,
    }
