from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, TypedDict

from flask import Flask, jsonify, render_template, request

from model import MedicalChatbot, PredictionResult



class Message(TypedDict):
    role: str
    text: str


BASE_DIR = Path(__file__).resolve().parent

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)



chatbot = MedicalChatbot(dataset_path=str(BASE_DIR / "symptoms_data.csv"))


def sanitize_history(history: object) -> List[Message]:
    if not isinstance(history, list):
        raise ValueError("History must be a list.")

    cleaned: List[Message] = []
    for item in history:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip()
        text = str(item.get("text", "")).strip()
        if role and text:
            cleaned.append({"role": role, "text": text})

    return cleaned


def prediction_to_payload(prediction: PredictionResult) -> Dict[str, Any]:
    return {
        "message": prediction.message,
        "top_disease": prediction.top_disease,
        "matched_symptoms": prediction.matched_symptoms,
        "precautions": prediction.precautions,
        "advice": prediction.advice,
        "recommended_doctor": prediction.recommended_doctor,
        "all_predictions": [
            {
                "disease": item["disease"],
                "matched_symptoms": item["matched_symptoms"],
            }
            for item in prediction.all_predictions
        ],
    }


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health() -> Any:
    return jsonify(
        {
            "status": "ok",
            "app": "MetaMaid",
            "bot": "Medical Symptom Chatbot",
            "dataset_loaded": chatbot.has_data,
            "disease_count": chatbot.disease_count,
            "model_ready": chatbot.model_ready,
            "predict_api_url": "http://127.0.0.1:5005/api/metamaid-symptom-checker",
        }
    )


@app.post("/api/chat")
def chat() -> Any:
    payload = request.get_json(silent=True) or {}
    message = str(payload.get("message", "")).strip()

    try:
        history = sanitize_history(payload.get("history", []))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not message:
        return jsonify({"error": "Message is required."}), 400

    prediction = chatbot.chat(message, history)
    return jsonify(prediction_to_payload(prediction))


@app.post("/api/metamaid-symptom-checker")
def predict() -> Any:
    payload = request.get_json(silent=True) or {}
    symptoms = payload.get("symptoms", "")

    if isinstance(symptoms, list):
        text = ", ".join(str(item).strip() for item in symptoms if str(item).strip())
    else:
        text = str(symptoms).strip()

    if not text:
        return jsonify({"error": "Symptoms are required."}), 400

    prediction = chatbot.predict_from_text(text)
    return jsonify(prediction_to_payload(prediction))


import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="127.0.0.1", port=port, debug=False)
