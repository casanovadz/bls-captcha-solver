import os
import io
import base64
import re
from typing import Tuple

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps

import torch
from transformers import VisionEncoderDecoderModel, TrOCRProcessor

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-printed")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load once at startup
processor = TrOCRProcessor.from_pretrained(MODEL_NAME)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()

app = Flask(__name__)
CORS(app)

digit_re = re.compile(r"\d+")


def _load_image_from_b64(b64_str: str) -> Image.Image:
    """Accepts base64 with or without data URI prefix and returns a PIL Image (RGB).
    Handles GIF by taking the first frame.
    """
    if "," in b64_str and b64_str.strip().startswith("data:"):
        # strip data URI prefix
        b64_str = b64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(img_bytes))
    try:
        img.seek(0)  # in case of GIF, go to first frame
    except Exception:
        pass
    if img.mode not in ("L", "RGB"):
        img = img.convert("RGB")
    return img


def _preprocess(img: Image.Image) -> Image.Image:
    """Light denoise/contrast suitable for BLS numeric tiles."""
    # Convert to grayscale, increase contrast, then back to RGB
    g = ImageOps.grayscale(img)
    # Auto contrast helps with faint digits
    g = ImageOps.autocontrast(g)
    # Optional: simple threshold if very noisy; keep soft to avoid losing thin strokes
    # Commented out by default; enable if needed
    # g = g.point(lambda p: 255 if p > 140 else 0)
    return g


def trocr_predict_text(pil_img: Image.Image) -> Tuple[str, float]:
    """Run TrOCR and return (pred_text, confidence in [0,1])."""
    with torch.no_grad():
        pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(DEVICE)
        outputs = model.generate(pixel_values, max_new_tokens=8)
        text = processor.batch_decode(outputs, skip_special_tokens=True)[0]

    # crude confidence from logits if available (average max softmax per token)
    try:
        # regenerate with output scores
        with torch.no_grad():
            outputs = model.generate(
                pixel_values, max_new_tokens=8, output_scores=True, return_dict_in_generate=True
            )
            # outputs.scores is a list[tensor] of logits per step
            import torch.nn.functional as F

            scores = outputs.scores
            if scores:
                max_probs = []
                for step in scores:
                    probs = F.softmax(step, dim=-1)
                    max_probs.append(probs.max(dim=-1).values.mean().item())
                conf = float(sum(max_probs) / len(max_probs))
            else:
                conf = 0.5
    except Exception:
        conf = 0.5

    return text.strip(), conf


def only_digits(s: str) -> str:
    return "".join(ch for ch in s if ch.isdigit())


@app.route("/health", methods=["GET"])  # simple health check
def health():
    return jsonify({"status": "ok", "device": DEVICE, "model": MODEL_NAME})


@app.route("/", methods=["GET"])  # compatible with the userscript
@app.route("/solve", methods=["GET"])  # alternative path
def solve():
    # Query params expected by the userscript
    model_name = request.args.get("a", "trocr").lower()
    image_b64 = request.args.get("b")
    number_to_find = request.args.get("n")

    if not image_b64 or number_to_find is None:
        return jsonify({"status": "error", "msg": "Missing parameters b or n"}), 400

    try:
        pil_img = _load_image_from_b64(image_b64)
        pil_img = _preprocess(pil_img)
    except Exception as e:
        return jsonify({"status": "error", "msg": f"Invalid image: {e}"}), 400

    # For now we support TrOCR; if 'a=vitstr' comes in, we still run TrOCR for compatibility
    pred_text, conf = trocr_predict_text(pil_img)
    pred_digits = only_digits(pred_text)
    target_digits = only_digits(str(number_to_find))

    ok = pred_digits == target_digits and len(target_digits) > 0

    return jsonify({
        "status": "ok" if ok else "fail",
        "pred": pred_text,
        "pred_digits": pred_digits,
        "target": target_digits,
        "conf": round(conf, 4),
        "used_model": "trocr"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "3000"))
    app.run(host="0.0.0.0", port=port)
