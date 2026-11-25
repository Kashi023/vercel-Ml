
from pathlib import Path
import io

import joblib
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from flask import Flask, request, jsonify
from flask_cors import CORS


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

SYMPTOM_MODEL_PATH = BASE_DIR / "symptom_model.joblib"
IMAGE_MODEL_PATH = BASE_DIR / "image_model.keras"

IMAGE_SIZE = (224, 224)
MAX_IMAGE_BYTES = 10 * 1024 * 1024

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = MAX_IMAGE_BYTES


# --------------------------------------------------
# 2. LOAD SYMPTOM MODEL
# --------------------------------------------------

def load_symptom_model():
    if not SYMPTOM_MODEL_PATH.exists():
        raise RuntimeError(
            f"Symptom model not found: {SYMPTOM_MODEL_PATH}. "
            "Run your symptom training script first."
        )

    bundle = joblib.load(SYMPTOM_MODEL_PATH)

    if not isinstance(bundle, dict):
        raise RuntimeError("Invalid symptom model bundle.")

    if "model" not in bundle or "encoder" not in bundle:
        raise RuntimeError(
            "Symptom model must contain 'model' and 'encoder'."
        )

    return bundle


bundle = load_symptom_model()
symptom_model = bundle["model"]
encoder = bundle["encoder"]

FEATURES = [
    str(s).strip().lower()
    for s in encoder.classes_
]
FEATURE_SET = set(FEATURES)


# --------------------------------------------------
# 3. LOAD IMAGE MODEL
# --------------------------------------------------

def load_image_model():
    if not IMAGE_MODEL_PATH.exists():
        print(
            f"WARNING: Image model not found: {IMAGE_MODEL_PATH}"
        )
        return None

    try:
        # The updated training script uses a built-in Rescaling
        # layer, so it no longer needs a Lambda custom function.
        model = tf.keras.models.load_model(
            IMAGE_MODEL_PATH,
            compile=False,
        )

        print("Image model loaded successfully.")
        return model

    except Exception:
        app.logger.exception("Unable to load image model")
        return None


image_model = load_image_model()


# --------------------------------------------------
# 4. COMMON HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():
    return jsonify({
        "success": True,
        "service": "HealthAI Python ML service",
        "symptom_model_loaded": True,
        "image_model_loaded": image_model is not None,
        "supported_symptom_count": len(FEATURES),
    })


@app.get("/symptoms")
def get_supported_symptoms():
    return jsonify({
        "success": True,
        "symptoms": FEATURES,
    })


# --------------------------------------------------
# 5. EXISTING SYMPTOM PREDICTION
# --------------------------------------------------

def normalize_symptom(value):
    return value.strip().lower().replace(" ", "_")


@app.post("/predict")
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "success": False,
            "message": "Request body must be a JSON object.",
        }), 400

    symptoms = data.get("symptoms")

    if not isinstance(symptoms, list) or not symptoms:
        return jsonify({
            "success": False,
            "message": "Please provide a non-empty symptoms list.",
        }), 400

    if not all(isinstance(s, str) for s in symptoms):
        return jsonify({
            "success": False,
            "message": "Every symptom must be a string.",
        }), 400

    normalized = {
        normalize_symptom(s)
        for s in symptoms
        if s.strip()
    }

    if not normalized:
        return jsonify({
            "success": False,
            "message": "Please select at least one valid symptom.",
        }), 400

    matched = sorted(normalized & FEATURE_SET)
    unknown = sorted(normalized - FEATURE_SET)

    print("Received symptoms:", sorted(normalized))
    print("Matched symptoms:", matched)
    print("Unknown symptoms:", unknown)

    if not matched:
        return jsonify({
            "success": False,
            "message": (
                "None of the selected symptoms match "
                "the training dataset."
            ),
            "unknownSymptoms": unknown,
            "supportedSymptoms": FEATURES,
        }), 400

    feature_row = encoder.transform([matched])
    prediction = str(symptom_model.predict(feature_row)[0])

    return jsonify({
        "success": True,
        "prediction": prediction,
        "matchedSymptoms": matched,
        "unmatchedSymptoms": unknown,
        "demo": True,
        "message": (
            "Educational demonstration only. "
            "Not validated for medical diagnosis."
        ),
    })


# --------------------------------------------------
# 6. CHEST X-RAY IMAGE ANALYSIS
# --------------------------------------------------

@app.post("/predict-image")
def predict_image():
    if image_model is None:
        return jsonify({
            "success": False,
            "message": (
                "Image model is not loaded. Check image_model.keras "
                "and the Flask terminal for model-loading errors."
            ),
            "educational_only": True,
        }), 503

    uploaded_file = request.files.get("image")

    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({
            "success": False,
            "message": (
                "Please upload a chest X-ray using "
                "the form field 'image'."
            ),
        }), 400

    try:
        raw_bytes = uploaded_file.stream.read(
            MAX_IMAGE_BYTES + 1
        )

        if not raw_bytes:
            return jsonify({
                "success": False,
                "message": "The uploaded image is empty.",
            }), 400

        if len(raw_bytes) > MAX_IMAGE_BYTES:
            return jsonify({
                "success": False,
                "message": "Image must be 10 MB or smaller.",
            }), 413

        # Verify the real image format.
        with Image.open(io.BytesIO(raw_bytes)) as img:
            if img.format not in ("JPEG", "PNG"):
                return jsonify({
                    "success": False,
                    "message": "Upload a JPG or PNG image.",
                }), 400

            img.verify()

        # Resize and convert to RGB.
        # The model performs MobileNetV2 rescaling internally.
        with Image.open(io.BytesIO(raw_bytes)) as img:
            img = img.convert("RGB")
            img = img.resize(IMAGE_SIZE)

            image_array = np.asarray(
                img,
                dtype=np.float32,
            )

        batch = np.expand_dims(image_array, axis=0)

        # Run model inference.
        output = image_model.predict(
            batch,
            verbose=0,
        )

        score = float(
            np.asarray(output).reshape(-1)[0]
        )

        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(
                "The model returned an invalid score."
            )

        # Training class mapping:
        # NORMAL = 0
        # PNEUMONIA = 1
        prediction = (
            "PNEUMONIA"
            if score >= 0.5
            else "NORMAL"
        )

        # This is the model's score for the predicted class,
        # not a clinically calibrated probability.
        class_score = (
            score
            if prediction == "PNEUMONIA"
            else 1.0 - score
        )

        if prediction == "PNEUMONIA":
            result_title = "Pneumonia class predicted"

            explanation = (
                "The model assigned this image to its "
                "pneumonia category. This result alone "
                "cannot establish whether pneumonia is present."
            )
        else:
            result_title = "Normal class predicted"

            explanation = (
                "The model assigned this image to its normal "
                "category. This result does not rule out "
                "pneumonia or another medical condition."
            )

        recommendation = (
            "Have the X-ray reviewed by a qualified "
            "healthcare professional. Do not use this "
            "experimental output to make treatment decisions."
        )

        disclaimer = (
            "Educational prototype only. This model has not "
            "been clinically validated and is not a diagnostic tool."
        )

        print(
            f"Image prediction: {prediction}; "
            f"model score: {class_score:.4f}"
        )

        return jsonify({
            "success": True,
            "prediction": prediction,
            "result_title": result_title,
            "model_score": round(class_score * 100, 2),
            "explanation": explanation,
            "recommendation": recommendation,
            "message": explanation,
            "educational_only": True,
            "disclaimer": disclaimer,
        }), 200

    except (UnidentifiedImageError, OSError, ValueError) as exc:
        return jsonify({
            "success": False,
            "message": f"Unable to analyze image: {exc}",
            "educational_only": True,
        }), 400

    except Exception:
        app.logger.exception("Image prediction failed")

        return jsonify({
            "success": False,
            "message": (
                "Image analysis failed on the ML service. "
                "Check the Flask terminal."
            ),
            "educational_only": True,
        }), 500


# --------------------------------------------------
# 7. HANDLE OVERSIZED REQUESTS
# --------------------------------------------------

@app.errorhandler(413)
def request_too_large(error):
    return jsonify({
        "success": False,
        "message": "Image must be 10 MB or smaller.",
    }), 413


# --------------------------------------------------
# 8. START FLASK
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,

    )
    
#venv\Scripts\activate
