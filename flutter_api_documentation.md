# MetaMaid Chatbot API Documentation for Flutter Integration

This document outlines the REST API endpoints available for integrating the MetaMaid Medical Symptom Chatbot into a Flutter application.

## Base URL
If running locally, the base URL will be:
`http://127.0.0.1:5005` *(Note: For a physical Android device or iOS simulator, you may need to use your machine's local IP address, e.g., `http://192.168.1.x:5005` or `http://10.0.2.2:5005` for Android Emulator).*

---

## 1. Chat Interaction Endpoint
This is the primary endpoint for conversational interactions. It accepts a user message and an optional chat history, returning the bot's response and disease predictions.

* **Endpoint:** `/api/chat`
* **Method:** `POST`
* **Headers:** 
  * `Content-Type: application/json`

### Request Body (JSON)
```json
{
  "message": "I have a severe headache and fever",
  "history": [
    {"role": "user", "text": "Hi"},
    {"role": "bot", "text": "Hello! I am MetaMaid, your medical symptom chatbot."}
  ]
}
```
*Note: `history` is optional but helpful if you want to maintain context.*

### Success Response (200 OK)
```json
{
  "message": "MetaMaid predicts that your symptoms match Common Cold.",
  "top_disease": "Common Cold",
  "matched_symptoms": ["headache", "fever"],
  "precautions": ["Drink plenty of fluids", "Rest", "Take vitamin C"],
  "advice": "Please consult a doctor for proper guidance.",
  "recommended_doctor": "General Physician",
  "all_predictions": [
    {
      "disease": "Common Cold",
      "matched_symptoms": ["headache", "fever"]
    },
    {
      "disease": "Flu",
      "matched_symptoms": ["fever"]
    }
  ]
}
```

---

## 2. Direct Symptom Checker Endpoint
Use this endpoint if your Flutter app has a specific UI (like checkboxes or a list of text fields) where users submit symptoms directly without a conversational interface.

* **Endpoint:** `/api/metamaid-symptom-checker`
* **Method:** `POST`
* **Headers:** 
  * `Content-Type: application/json`

### Request Body (JSON)
You can send symptoms as a single string or an array of strings.
```json
{
  "symptoms": ["fever", "cough", "headache"]
}
```
*OR*
```json
{
  "symptoms": "fever, cough, headache"
}
```

### Success Response (200 OK)
*(The response structure is exactly the same as the `/api/chat` endpoint).*

---

## 3. Health Check Endpoint
Useful for checking if the backend is online and the ML model is successfully loaded before the user tries to send a message.

* **Endpoint:** `/api/health`
* **Method:** `GET`

### Success Response (200 OK)
```json
{
  "status": "ok",
  "app": "MetaMaid",
  "bot": "Medical Symptom Chatbot",
  "dataset_loaded": true,
  "disease_count": 41,
  "model_ready": true,
  "predict_api_url": "http://127.0.0.1:5005/api/metamaid-symptom-checker"
}
```

---

## Flutter Integration Tips
For your Flutter developer:
1. Use the [`http`](https://pub.dev/packages/http) or [`dio`](https://pub.dev/packages/dio) package to make network requests.
2. Create Dart model classes for the Request and Response to easily serialize/deserialize the JSON using `json_serializable` or `freezed`.
3. If hitting `localhost` from the Android Emulator, use `10.0.2.2` instead of `127.0.0.1`.
