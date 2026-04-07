"""
CNN Inference Module — ChestX-ray14 Multi-Label Prediction
==========================================================
Loads the trained EfficientNet-B3 checkpoint and runs inference on uploaded X-rays.
Falls back to a placeholder if the model isn't trained yet.

Features:
- Per-class optimized thresholds (loaded from optimal_thresholds.json)
- Confidence check: if no condition > MIN_CONFIDENCE, flags for manual selection
- torch.load with weights_only=False for PyTorch 2.6+ compatibility

Usage:
    from api.cnn_inference import predict_conditions, is_model_loaded
    results = predict_conditions(image_bytes)
    # Returns: [{"condition": "Pneumonia", "probability": 0.87, "detected": True}, ...]
"""

import io
import json
import logging
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ── Conditions (display names use spaces) ───────────────────────────────────

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax",
]

# ── Paths ───────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "models/checkpoints/efficientnet_b3_multilabel_best.pth"
THRESHOLD_PATH = "models/checkpoints/optimal_thresholds.json"
IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.5
MIN_CONFIDENCE = 0.15  # Global floor — catches non-X-ray images (dogs, selfies, etc.)


# ── Module state ────────────────────────────────────────────────────────────

_model = None
_device = None
_model_loaded = False
_thresholds = {}  # Per-class optimized thresholds

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Model building ─────────────────────────────────────────────────────────

def _build_model():
    """Build EfficientNet-B3 with 14-class multi-label head."""
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(CONDITIONS)),
    )
    return model


def _load_thresholds():
    """Load per-class optimized thresholds, fall back to defaults."""
    global _thresholds

    if os.path.exists(THRESHOLD_PATH):
        try:
            with open(THRESHOLD_PATH, "r") as f:
                data = json.load(f)
            raw_thresholds = data.get("thresholds", {})
            # Map from underscore names (training) to space names (inference)
            for condition in CONDITIONS:
                underscore_name = condition.replace(" ", "_")
                if condition in raw_thresholds:
                    _thresholds[condition] = raw_thresholds[condition]
                elif underscore_name in raw_thresholds:
                    _thresholds[condition] = raw_thresholds[underscore_name]
                else:
                    _thresholds[condition] = DEFAULT_THRESHOLD
            logger.info(f"Loaded per-class thresholds from {THRESHOLD_PATH}")
            logger.info(f"  Method: {data.get('method', 'unknown')}")
            logger.info(f"  Mean F1 improvement: {data.get('mean_f1_default', 0):.3f} -> {data.get('mean_f1_optimized', 0):.3f}")
        except Exception as e:
            logger.warning(f"Failed to load thresholds: {e}, using defaults")
            _thresholds = {c: DEFAULT_THRESHOLD for c in CONDITIONS}
    else:
        logger.info(f"No threshold file at {THRESHOLD_PATH}, using default {DEFAULT_THRESHOLD}")
        _thresholds = {c: DEFAULT_THRESHOLD for c in CONDITIONS}


# ── Startup ─────────────────────────────────────────────────────────────────

def load_model():
    """Load the trained model checkpoint. Call once at startup."""
    global _model, _device, _model_loaded

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load thresholds first (works even without model)
    _load_thresholds()

    if os.path.exists(CHECKPOINT_PATH):
        try:
            _model = _build_model()
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location=_device,
                weights_only=False  # PyTorch 2.6+ compatibility
            )
            _model.load_state_dict(checkpoint["model_state_dict"])
            _model.to(_device)
            _model.eval()
            _model_loaded = True

            best_auc = checkpoint.get("best_auc", "unknown")
            epoch = checkpoint.get("epoch", "unknown")
            logger.info(f"CNN model loaded: {CHECKPOINT_PATH} (epoch {epoch}, AUC {best_auc})")
        except Exception as e:
            logger.error(f"Failed to load CNN model: {e}")
            _model_loaded = False
    else:
        logger.warning(f"CNN model not found at {CHECKPOINT_PATH} — using placeholder predictions")
        _model_loaded = False


def is_model_loaded() -> bool:
    """Check if the real CNN model is available."""
    return _model_loaded


# ── Prediction ──────────────────────────────────────────────────────────────

def predict_conditions(image_bytes: bytes) -> dict:
    """
    Predict conditions from a chest X-ray image.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, etc.)

    Returns:
        dict with keys:
            - predictions: list of dicts sorted by probability (highest first)
            - model_loaded: bool
            - needs_manual_selection: bool (True if no condition > MIN_CONFIDENCE)
            - top_condition: str or None
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if _model_loaded and _model is not None:
        predictions = _predict_real(image)
    else:
        predictions = _predict_placeholder(image)

    # Check if any condition exceeds minimum confidence
    # 0.15 global floor catches non-X-ray images (dogs, selfies)
    # Per-class thresholds (0.12-0.37) handle individual condition detection
    max_prob = max(p["probability"] for p in predictions) if predictions else 0
    any_detected = any(p["detected"] for p in predictions)
    needs_manual = (max_prob < MIN_CONFIDENCE or not any_detected) and _model_loaded
    top_condition = predictions[0]["condition"] if predictions and predictions[0]["detected"] else None

    return {
        "predictions": predictions,
        "model_loaded": _model_loaded,
        "needs_manual_selection": needs_manual,
        "top_condition": top_condition,
        "confidence_threshold": MIN_CONFIDENCE,
    }


def _predict_real(image: Image.Image) -> list[dict]:
    """Run real model inference with per-class optimized thresholds."""
    input_tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(input_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    results = []
    for i, condition in enumerate(CONDITIONS):
        threshold = _thresholds.get(condition, DEFAULT_THRESHOLD)
        prob = float(probabilities[i])
        results.append({
            "condition": condition,
            "probability": round(prob, 4),
            "detected": prob >= threshold,
            "threshold": round(threshold, 3),
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def _predict_placeholder(image: Image.Image) -> list[dict]:
    """Placeholder predictions when model isn't trained yet."""
    results = []
    for condition in CONDITIONS:
        results.append({
            "condition": condition,
            "probability": 0.0,
            "detected": False,
            "threshold": DEFAULT_THRESHOLD,
        })
    return results
