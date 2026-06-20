from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__, static_folder="static")
CORS(app)

# Load model and preprocessors at startup
model = load_model("obesity_ann_model.keras")
scaler = joblib.load("obesity_scaler.joblib")
encoders = joblib.load("obesity_encoders.joblib")
target_encoder = joblib.load("obesity_target_encoder.joblib")

CATEGORICAL_COLUMNS = [
    "Gender", "family_history_with_overweight", "FAVC",
    "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"
]

FEATURE_ORDER = [
    "Gender", "Age", "Height", "Weight",
    "family_history_with_overweight", "FAVC", "FCVC", "NCP",
    "CAEC", "SMOKE", "CH2O", "SCC", "FAF", "TUE", "CALC", "MTRANS"
]

OBESITY_INFO = {
    "Insufficient_Weight": {
        "label": "Insufficient Weight",
        "color": "#60a5fa",
        "advice": "You may be underweight. Consider a balanced diet with adequate calories and consult a nutritionist.",
        "risk": "Low"
    },
    "Normal_Weight": {
        "label": "Normal Weight",
        "color": "#34d399",
        "advice": "Great! You are at a healthy weight. Maintain your current lifestyle with regular exercise and balanced nutrition.",
        "risk": "Minimal"
    },
    "Overweight_Level_I": {
        "label": "Overweight — Level I",
        "color": "#fbbf24",
        "advice": "Slightly overweight. Moderate dietary adjustments and regular physical activity can help.",
        "risk": "Low–Moderate"
    },
    "Overweight_Level_II": {
        "label": "Overweight — Level II",
        "color": "#f97316",
        "advice": "Consider consulting a healthcare professional for a structured diet and fitness plan.",
        "risk": "Moderate"
    },
    "Obesity_Type_I": {
        "label": "Obesity — Type I",
        "color": "#ef4444",
        "advice": "Medical consultation is recommended. Lifestyle changes including diet and exercise are important.",
        "risk": "High"
    },
    "Obesity_Type_II": {
        "label": "Obesity — Type II",
        "color": "#dc2626",
        "advice": "Please seek medical advice. This level of obesity carries significant health risks.",
        "risk": "Very High"
    },
    "Obesity_Type_III": {
        "label": "Obesity — Type III (Severe)",
        "color": "#991b1b",
        "advice": "Immediate medical attention is strongly advised. Comprehensive treatment plan needed.",
        "risk": "Severe"
    },
}


@app.route("/")
def index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Encode categorical features
        encoded = {}
        for col in CATEGORICAL_COLUMNS:
            if col not in data:
                return jsonify({"error": f"Missing field: {col}"}), 400
            val = data[col]
            if col in encoders:
                try:
                    encoded[col] = int(encoders[col].transform([val])[0])
                except ValueError:
                    return jsonify({"error": f"Invalid value '{val}' for field '{col}'"}), 400
            else:
                encoded[col] = val

        # Build feature vector in correct order
        features = []
        for col in FEATURE_ORDER:
            if col in CATEGORICAL_COLUMNS:
                features.append(encoded[col])
            else:
                if col not in data:
                    return jsonify({"error": f"Missing field: {col}"}), 400
                features.append(float(data[col]))

        patient = np.array([features])
        patient_scaled = scaler.transform(patient)

        prediction = model.predict(patient_scaled, verbose=0)
        probabilities = prediction[0].tolist()
        result_idx = int(np.argmax(prediction))
        obesity_level = target_encoder.inverse_transform([result_idx])[0]

        classes = target_encoder.classes_.tolist()
        prob_map = {cls: round(float(p) * 100, 2) for cls, p in zip(classes, probabilities)}

        info = OBESITY_INFO.get(obesity_level, {
            "label": obesity_level,
            "color": "#6b7280",
            "advice": "Please consult a healthcare professional.",
            "risk": "Unknown"
        })

        bmi = round(float(data["Weight"]) / (float(data["Height"]) ** 2), 2)

        return jsonify({
            "success": True,
            "obesity_level": obesity_level,
            "label": info["label"],
            "color": info["color"],
            "advice": info["advice"],
            "risk_level": info["risk"],
            "confidence": round(float(probabilities[result_idx]) * 100, 2),
            "bmi": bmi,
            "probabilities": prob_map
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
