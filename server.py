from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from flask import Flask, request, jsonify
from PIL import Image
import torch
import os

# المسار المحلي للموديل اللي هنحفظه فيه
MODEL_PATH = "./model"

# تحميل الـ Processor والـ Model من المجلد المحلي
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = VisionEncoderDecoderModel.from_pretrained(MODEL_PATH)

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image = Image.open(request.files["image"]).convert("RGB")
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return jsonify({"captcha": generated_text})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

