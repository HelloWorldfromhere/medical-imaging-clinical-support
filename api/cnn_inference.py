"""
CNN Inference Module — ChestX-ray14 Multi-Label Prediction
Loads the trained EfficientNet-B3 checkpoint and runs inference on uploaded X-rays.
Falls back to a placeholder if the model isn't trained yet.

Usage:
    from api.cnn_inference import predict_conditions
    results = predict_conditions(image_bytes)
    # Returns: [{"condition": "Pneumonia", "probability": 0.87}, ...]
"""

import io
import os
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

logger = logging.getLogger(__name__)

CONDITIONS = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema",
    "Effusion", "Emphysema", "Fibrosis", "Hernia",
    "Infiltration", "Mass", "Nodule", "Pleural Thickening",
    "Pneumonia", "Pneumothorax",
]

CHECKPOINT_PATH = "models/checkpoints/efficientnet_b3_multilabel_best.pth"
IMAGE_SIZE = 224
THRESHOLD = 0.5

_model = None
_device = None
_model_loaded = False

_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _build_model():
    """Build EfficientNet-B3 with 14-class multi-label head."""
    model = models.efficientnet_b3(weights=None)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(num_features, len(CONDITIONS)),
    )
    return model


def load_model():
    """Load the trained model checkpoint. Call once at startup."""
    global _model, _device, _model_loaded

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(CHECKPOINT_PATH):
        try:
            _model = _build_model()
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=_device)
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


def is_model_loaded():
    """Check if the real CNN model is available."""
    return _model_loaded


def predict_conditions(image_bytes: bytes) -> list[dict]:
    """
    Predict conditions from a chest X-ray image.

    Args:
        image_bytes: Raw image bytes (PNG, JPEG, etc.)

    Returns:
        List of dicts: [{"condition": str, "probability": float, "detected": bool}]
        Sorted by probability (highest first).
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if _model_loaded and _model is not None:
        return _predict_real(image)
    else:
        return _predict_placeholder(image)


def _predict_real(image: Image.Image) -> list[dict]:
    """Run real model inference."""
    input_tensor = _transform(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        logits = _model(input_tensor)
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    results = []
    for i, condition in enumerate(CONDITIONS):
        results.append({
            "condition": condition,
            "probability": round(float(probabilities[i]), 4),
            "detected": bool(probabilities[i] >= THRESHOLD),
        })

    results.sort(key=lambda x: x["probability"], reverse=True)
    return results


def _predict_placeholder(image: Image.Image) -> list[dict]:
    """
    Placeholder predictions when model isn't trained yet.
    Returns all conditions with 0.0 probability and a flag.
    """
    results = []
    for condition in CONDITIONS:
        results.append({
            "condition": condition,
            "probability": 0.0,
            "detected": False,
        })
    return results
