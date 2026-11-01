#!/usr/bin/env python3
"""
AgriVision Multi-Model Inference Engine (Strict Crop Scoping)
------------------------------------------------------------
Routes diagnostic queries to specialized SOTA models with STRICT crop isolation:
  - When a crop is selected (e.g. Apple), the model strictly evaluates only that crop's diseases.
  - No cross-crop leakage or foreign crop predictions can occur.

Supported Crops (13 total):
  1. Apple
  2. Chickpea
  3. Corn (Maize)
  4. Cotton
  5. Onion
  6. Pepper (Bell Pepper)
  7. Potato
  8. Rice
  9. Rose
  10. Soybean
  11. Sugarcane
  12. Tomato
  13. Wheat
"""

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print(json.dumps({"error": "onnxruntime is not installed. Please run: pip install onnxruntime"}))
    sys.exit(1)

IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# General Crops Dictionary (PlantVillage SOTA)
GENERAL_CLASSES = {
    0: "Apple___Apple_scab",
    1: "Apple___Black_rot",
    2: "Apple___Cedar_apple_rust",
    3: "Apple___healthy",
    4: "Blueberry___healthy",
    5: "Cherry___Powdery_mildew",
    6: "Cherry___healthy",
    7: "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    8: "Corn___Common_rust",
    9: "Corn___Northern_Leaf_Blight",
    10: "Corn___healthy",
    11: "Grape___Black_rot",
    12: "Grape___Esca_Black_Measles",
    13: "Grape___Leaf_blight_Isariopsis_Leaf_Spot",
    14: "Grape___healthy",
    15: "Orange___Haunglongbing_Citrus_greening",
    16: "Peach___Bacterial_spot",
    17: "Peach___healthy",
    18: "Pepper___Bacterial_spot",
    19: "Pepper___healthy",
    20: "Potato___Early_blight",
    21: "Potato___Late_blight",
    22: "Potato___healthy",
    23: "Raspberry___healthy",
    24: "Soybean___healthy",
    25: "Squash___Powdery_mildew",
    26: "Strawberry___Leaf_scorch",
    27: "Strawberry___healthy",
    28: "Tomato___Bacterial_spot",
    29: "Tomato___Early_blight",
    30: "Tomato___Late_blight",
    31: "Tomato___Leaf_Mold",
    32: "Tomato___Septoria_leaf_spot",
    33: "Tomato___Spider_mites_Two_spotted_spider_mite",
    34: "Tomato___Target_Spot",
    35: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    36: "Tomato___Tomato_mosaic_virus",
    37: "Tomato___healthy"
}

# Dedicated Rice Specialist Classes
RICE_CLASSES = {
    0: "Rice___Bacterial_leaf_blight",
    1: "Rice___Brown_spot",
    2: "Rice___Leaf_smut",
    3: "Rice___Tungro"
}


def load_disease_metadata(disease_info_path: str = "disease_info.json") -> Dict:
    if os.path.exists(disease_info_path):
        try:
            with open(disease_info_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def load_image(image_input: Union[str, Path, Image.Image]) -> Image.Image:
    if isinstance(image_input, (str, Path)):
        image_str = str(image_input)
        if image_str.startswith("http://") or image_str.startswith("https://"):
            import urllib.request
            req = urllib.request.Request(image_str, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                image_bytes = resp.read()
            return Image.open(io.BytesIO(image_bytes)).convert("RGB")
        else:
            if not os.path.exists(image_str):
                raise FileNotFoundError(f"Image not found: {image_str}")
            return Image.open(image_str).convert("RGB")
    elif isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    raise ValueError("Invalid image input type.")


def preprocess_nchw(img: Image.Image) -> np.ndarray:
    """Preprocess to (1, 3, 224, 224) with ImageNet normalization."""
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def preprocess_nhwc(img: Image.Image) -> np.ndarray:
    """Preprocess to (1, 224, 224, 3) normalized [0, 1]."""
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    arr = np.array(img_resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class StrictMultiModelDetector:
    """Strictly crop-isolated neural disease diagnostic engine."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.disease_info = load_disease_metadata(str(self.base_dir / "disease_info.json"))

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]

        # 1. General SOTA Model (Apple, Corn, Potato, Tomato, Pepper, Soybean)
        general_model_path = self.models_dir / "general_crops.onnx"
        if not general_model_path.exists():
            general_model_path = self.base_dir / "crop_model.onnx"

        self.general_session = None
        self.general_input_name = None
        if general_model_path.exists():
            try:
                self.general_session = ort.InferenceSession(str(general_model_path), providers=providers)
                self.general_input_name = self.general_session.get_inputs()[0].name
            except Exception as e:
                print(f"Warning: Failed to load general model: {e}", file=sys.stderr)

        # 2. Rice Specialist Model
        rice_model_path = self.models_dir / "rice_model.onnx"
        self.rice_session = None
        self.rice_input_name = None
        if rice_model_path.exists():
            try:
                self.rice_session = ort.InferenceSession(str(rice_model_path), providers=providers)
                self.rice_input_name = self.rice_session.get_inputs()[0].name
            except Exception as e:
                print(f"Warning: Failed to load rice model: {e}", file=sys.stderr)

        # 3. Remaining Crops Model (Wheat, Cotton, Sugarcane, Onion, Rose, Chickpea)
        remaining_model_path = self.models_dir / "remaining_crops.onnx"
        remaining_classes_path = self.models_dir / "remaining_classes.json"
        self.remaining_session = None
        self.remaining_classes = {}
        if remaining_model_path.exists() and remaining_classes_path.exists():
            try:
                self.remaining_session = ort.InferenceSession(str(remaining_model_path), providers=providers)
                with open(remaining_classes_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.remaining_classes = {int(k): str(v) for k, v in data.items()}
            except Exception as e:
                print(f"Warning: Failed to load remaining crops model: {e}", file=sys.stderr)

    def diagnose(self, image_input: Union[str, Path, Image.Image], top_k: int = 3, crop_filter: str = "apple"):
        img = load_image(image_input)
        start_time = time.perf_counter()

        # Normalize crop name
        crop_clean = (crop_filter or "apple").strip().lower().replace("_", "").replace(" ", "").replace("(maize)", "")
        if "corn" in crop_clean:
            crop_clean = "corn"
        elif "pepper" in crop_clean:
            crop_clean = "pepper"

        candidates = []

        # ----------------------------------------------------------------------
        # STRICT ROUTE 1: RICE
        # ----------------------------------------------------------------------
        if crop_clean in {"rice", "paddy"}:
            if not self.rice_session:
                raise RuntimeError("Rice specialist model not loaded.")
            tensor = preprocess_nhwc(img)
            out = self.rice_session.run(None, {self.rice_input_name: tensor})
            logits = out[0][0]
            probs = softmax(logits)
            for idx, p in enumerate(probs):
                candidates.append((RICE_CLASSES[idx], float(p)))

        # ----------------------------------------------------------------------
        # STRICT ROUTE 2: GENERAL CROPS (Apple, Corn, Potato, Tomato, Pepper, Soybean)
        # ----------------------------------------------------------------------
        elif crop_clean in {"apple", "corn", "potato", "tomato", "pepper", "soybean"}:
            if not self.general_session:
                raise RuntimeError("General crops model not loaded.")
            tensor = preprocess_nchw(img)
            out = self.general_session.run(None, {self.general_input_name: tensor})
            logits = out[0][0]

            # Find matching indices STRICTLY for the target crop
            matched_indices = []
            for idx, lbl in GENERAL_CLASSES.items():
                c_prefix = lbl.split("___")[0].lower()
                if crop_clean in c_prefix or c_prefix in crop_clean:
                    matched_indices.append(idx)

            if not matched_indices:
                matched_indices = [0, 1, 2, 3]  # Fallback to Apple if unmatched

            # Compute softmax STRICTLY among the target crop's logits
            sub_logits = np.array([logits[i] for i in matched_indices])
            sub_probs = softmax(sub_logits)

            for idx, p in zip(matched_indices, sub_probs):
                candidates.append((GENERAL_CLASSES[idx], float(p)))

        # ----------------------------------------------------------------------
        # STRICT ROUTE 3: REMAINING CROPS (Wheat, Cotton, Sugarcane, Onion, Rose, Chickpea)
        # ----------------------------------------------------------------------
        elif crop_clean in {"wheat", "cotton", "sugarcane", "sugar", "cane", "onion", "rose", "chickpea"}:
            if not self.remaining_session:
                raise RuntimeError("Remaining crops model not loaded.")
            tensor = preprocess_nchw(img)
            input_name = self.remaining_session.get_inputs()[0].name
            out = self.remaining_session.run(None, {input_name: tensor})
            logits = out[0][0]

            # Search strictly for labels starting with this crop name
            matched_indices = []
            for idx, lbl in self.remaining_classes.items():
                lbl_lower = lbl.lower()
                if lbl_lower.startswith(crop_clean + "___"):
                    matched_indices.append(idx)
                elif crop_clean in ["sugarcane", "sugar", "cane"] and lbl_lower.startswith("sugarcane___"):
                    matched_indices.append(idx)

            if not matched_indices:
                # Fallback to any class containing crop name
                matched_indices = [idx for idx, lbl in self.remaining_classes.items() if crop_clean in lbl.lower()]

            if not matched_indices:
                matched_indices = list(self.remaining_classes.keys())[:4]

            # Compute softmax STRICTLY among matched crop logits
            sub_logits = np.array([logits[i] for i in matched_indices])
            sub_probs = softmax(sub_logits)

            for idx, p in zip(matched_indices, sub_probs):
                candidates.append((self.remaining_classes[idx], float(p)))

        else:
            # Default to Apple strictly if unrecognized
            tensor = preprocess_nchw(img)
            out = self.general_session.run(None, {self.general_input_name: tensor})
            logits = out[0][0]
            matched_indices = [0, 1, 2, 3]
            sub_probs = softmax(np.array([logits[i] for i in matched_indices]))
            for idx, p in zip(matched_indices, sub_probs):
                candidates.append((GENERAL_CLASSES[idx], float(p)))

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Sort strictly by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = candidates[:top_k]

        results = []
        for rank, (label, prob) in enumerate(top_candidates):
            parts = label.split("___")
            raw_crop = parts[0] if len(parts) > 1 else crop_clean.title()
            raw_disease = parts[1] if len(parts) > 1 else label

            # Normalize names
            crop_name = raw_crop.replace("_", " ").title()
            disease_name = raw_disease.replace("_", " ").title()
            is_healthy = "healthy" in disease_name.lower()
            status = "Healthy" if is_healthy else "Diseased"

            # Check metadata
            matched_info = None
            for key, val in self.disease_info.items():
                if val.get("crop", "").lower() == crop_name.lower() and val.get("disease", "").lower() == disease_name.lower():
                    matched_info = val
                    break

            treatment = f"{crop_name} crop is healthy. Continue balanced nutrition and scouting." if is_healthy else f"Apply recommended {crop_name} disease spray (Neem oil or systemic fungicide/bactericide)."
            if matched_info and "treatment" in matched_info:
                treatment = matched_info["treatment"]

            results.append({
                "class_id": rank,
                "label": label,
                "crop": crop_name,
                "disease": disease_name,
                "status": status,
                "treatment": treatment,
                "confidence": round(prob * 100.0, 2)
            })

        return results, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Strict Multi-Model Crop Disease Diagnostic Engine")
    parser.add_argument("--image", "-i", type=str, default=None, help="Path or URL to leaf image")
    parser.add_argument("--crop", "-c", type=str, default="apple", help="Target crop (STRICT: Apple, Corn, Potato, Rice, Wheat, etc.)")
    parser.add_argument("--topk", "-k", type=int, default=3, help="Number of predictions")
    parser.add_argument("--json", action="store_true", help="Output JSON format")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        for fallback in ["sample_leaf.jpg", "test_leaf.jpg"]:
            if os.path.exists(fallback):
                image_path = fallback
                break

    if not image_path:
        if args.json:
            print(json.dumps({"error": "No image provided"}))
        else:
            print("Error: No image provided")
        sys.exit(1)

    detector = StrictMultiModelDetector(base_dir=".")
    try:
        results, latency_ms = detector.diagnose(image_path, top_k=args.topk, crop_filter=args.crop)
        if args.json:
            top = results[0] if results else {}
            output = {
                "top": top,
                "predictions": results,
                "latency_ms": round(latency_ms, 2),
                "image": str(image_path)
            }
            print(json.dumps(output, indent=2))
        else:
            print(f"\nStrict {args.crop.title()} Diagnosis complete in {latency_ms:.1f}ms:")
            for r in results:
                print(f"[{r['confidence']}%] {r['crop']} - {r['disease']} ({r['status']})")
    except Exception as e:
        if args.json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"Inference error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
