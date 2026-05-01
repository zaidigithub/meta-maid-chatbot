from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, TypedDict

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import PhraseMatcher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class Message(TypedDict):
    role: str
    text: str


class RankedPrediction(TypedDict):
    disease: str
    score: float
    matched_symptoms: List[str]


@dataclass
class PredictionResult:
    message: str
    top_disease: str
    matched_symptoms: List[str]
    precautions: List[str]
    advice: str
    recommended_doctor: str
    all_predictions: List[RankedPrediction]


@dataclass
class DiseaseRecord:
    disease: str
    symptoms: List[str]
    precautions: List[str]
    advice: str
    recommended_doctor: str


@dataclass
class MedicalChatbot:
    dataset_path: str = "symptoms_data.csv"
    records: List[DiseaseRecord] = field(init=False, default_factory=list)
    model: Pipeline | None = field(init=False, default=None)
    label_order: List[str] = field(init=False, default_factory=list)
    nlp: spacy.language.Language = field(init=False)
    symptom_matcher: PhraseMatcher = field(init=False)

    def __post_init__(self) -> None:
        self.records = self._load_dataset(Path(self.dataset_path))
        self.nlp = spacy.blank("en")
        self.symptom_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self._build_symptom_matcher()
        self.model, self.label_order = self._train_model(self.records)

    @property
    def has_data(self) -> bool:
        return bool(self.records)

    @property
    def disease_count(self) -> int:
        return len(self.records)

    @property
    def model_ready(self) -> bool:
        return self.model is not None and bool(self.label_order)

    def _load_dataset(self, path: Path) -> List[DiseaseRecord]:
        if not path.exists():
            return []

        try:
            dataframe = pd.read_csv(path, encoding="utf-8-sig")
        except (OSError, pd.errors.EmptyDataError):
            return []

        dataframe.columns = [str(column).strip().lower() for column in dataframe.columns]
        records: List[DiseaseRecord] = []

        for row in dataframe.to_dict(orient="records"):
            record = self._row_to_record(row)
            if record:
                records.append(record)

        return records

    def _row_to_record(self, row: Dict[str, str]) -> DiseaseRecord | None:
        normalized_row = {str(k).strip().lower(): str(v or "").strip() for k, v in row.items()}

        disease = (
            normalized_row.get("disease")
            or normalized_row.get("prognosis")
            or normalized_row.get("diagnosis")
            or normalized_row.get("name")
            or ""
        ).strip()
        if not disease:
            return None

        symptoms = self._split_values(normalized_row.get("symptoms", ""))
        if not symptoms:
            symptom_columns = [
                value
                for key, value in normalized_row.items()
                if key.startswith("symptom") and value.strip()
            ]
            symptoms = [self._normalize_token(value) for value in symptom_columns if value.strip()]

        symptoms = [item for item in symptoms if item]
        if not symptoms:
            return None

        precautions = self._split_values(normalized_row.get("precautions", ""))
        advice = normalized_row.get("advice", "").strip()
        recommended_doctor = normalized_row.get("recommended_doctor", "").strip() or "General Physician"

        return DiseaseRecord(
            disease=disease,
            symptoms=symptoms,
            precautions=precautions,
            advice=advice,
            recommended_doctor=recommended_doctor,
        )

    def _train_model(self, records: Sequence[DiseaseRecord]) -> tuple[Pipeline | None, List[str]]:
        if len(records) < 2:
            return None, []

        training_rows: List[Dict[str, str]] = []
        for record in records:
            symptoms_text = ", ".join(record.symptoms)
            training_rows.append({"text": symptoms_text, "label": record.disease})
            training_rows.append({"text": f"i have {symptoms_text}", "label": record.disease})
            training_rows.append({"text": f"symptoms include {symptoms_text}", "label": record.disease})

        training_frame = pd.DataFrame(training_rows)
        if training_frame.empty or training_frame["label"].nunique() < 2:
            return None, []

        pipeline = Pipeline(
            steps=[
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
                ("classifier", LogisticRegression(max_iter=1000)),
            ]
        )
        pipeline.fit(training_frame["text"], training_frame["label"])

        classifier = pipeline.named_steps["classifier"]
        label_order = [str(label) for label in classifier.classes_]
        return pipeline, label_order

    def _build_symptom_matcher(self) -> None:
        symptom_terms = sorted(
            {
                self._normalize_token(symptom)
                for record in self.records
                for symptom in record.symptoms
                if self._normalize_token(symptom)
            }
        )
        if not symptom_terms:
            return

        patterns = [self.nlp.make_doc(term) for term in symptom_terms]
        self.symptom_matcher.add("SYMPTOM", patterns)

    def _split_values(self, raw: str) -> List[str]:
        if not raw:
            return []
        parts = re.split(r"[|,;/]", raw)
        return [self._normalize_token(part) for part in parts if self._normalize_token(part)]

    def _normalize_token(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text).replace("_", " ").strip().lower())

    def _extract_symptoms(self, text: str) -> List[str]:
        cleaned = self._normalize_token(text)
        doc = self.nlp(cleaned)
        chunks = re.split(r"(?:,|\.| and | with | also | plus |\n)", cleaned)

        stop_phrases = {
            "i have",
            "i am having",
            "i am feeling",
            "i feel",
            "my",
            "me",
            "the",
            "a",
            "an",
            "symptoms",
            "symptom",
        }

        extracted: List[str] = []
        for chunk in chunks:
            phrase = chunk.strip()
            if not phrase:
                continue

            phrase = re.sub(
                r"^(i have|i am having|i am feeling|i feel|having|suffering from)\s+",
                "",
                phrase,
            ).strip()

            if not phrase or phrase in stop_phrases:
                continue

            extracted.append(phrase)

        matched_spans = self.symptom_matcher(doc)
        for _, start, end in matched_spans:
            phrase = self._normalize_token(doc[start:end].text)
            if phrase:
                extracted.append(phrase)

        unique: List[str] = []
        for symptom in extracted:
            if symptom not in unique:
                unique.append(symptom)
        return unique

    def _lookup_record(self, disease: str) -> DiseaseRecord | None:
        for record in self.records:
            if record.disease == disease:
                return record
        return None

    def _match_symptoms(self, user_symptoms: Sequence[str], known_symptoms: Sequence[str]) -> List[str]:
        known_list = [self._normalize_token(item) for item in known_symptoms]
        matched: List[str] = []

        for user_symptom in user_symptoms:
            normalized_user = self._normalize_token(user_symptom)
            for known_symptom in known_list:
                if normalized_user == known_symptom:
                    matched.append(known_symptom)
                    break
                if normalized_user in known_symptom or known_symptom in normalized_user:
                    matched.append(known_symptom)
                    break

        unique: List[str] = []
        for item in matched:
            if item not in unique:
                unique.append(item)
        return unique

    def predict_from_text(self, text: str) -> PredictionResult:
        user_symptoms = self._extract_symptoms(text)

        if not user_symptoms:
            return PredictionResult(
                message="Please describe your symptoms, for example: fever, cough, headache.",
                top_disease="Unknown",
                matched_symptoms=[],
                precautions=[],
                advice="Try listing 2 or 3 symptoms separated by commas.",
                recommended_doctor="General Physician",
                all_predictions=[],
            )

        if not self.records:
            return PredictionResult(
                message="The symptoms dataset is missing or empty, so I cannot predict diseases yet.",
                top_disease="Unknown",
                matched_symptoms=user_symptoms,
                precautions=[],
                advice="Add rows to your symptoms dataset with disease names and symptoms.",
                recommended_doctor="General Physician",
                all_predictions=[],
            )

        if not self.model_ready or self.model is None:
            return PredictionResult(
                message="The model could not be trained from the current dataset.",
                top_disease="Unknown",
                matched_symptoms=user_symptoms,
                precautions=[],
                advice="Add more disease rows and make sure each one has clear symptom text.",
                recommended_doctor="General Physician",
                all_predictions=[],
            )

        normalized_input = ", ".join(user_symptoms)
        probabilities = self.model.predict_proba([normalized_input])[0]
        ranked_indices = np.argsort(probabilities)[::-1]

        ranked: List[RankedPrediction] = []
        for index in ranked_indices[:3]:
            disease = self.label_order[int(index)]
            probability = float(probabilities[int(index)])
            record = self._lookup_record(disease)
            matched = self._match_symptoms(user_symptoms, record.symptoms if record else [])
            ranked.append(
                {
                    "disease": disease,
                    "score": probability,
                    "matched_symptoms": matched,
                }
            )

        top = ranked[0]
        best_record = self._lookup_record(top["disease"])

        precautions = best_record.precautions if best_record else []
        advice = best_record.advice if best_record else "Please consult a doctor for proper guidance."
        recommended_doctor = best_record.recommended_doctor if best_record else "General Physician"

        message = f"MetaMaid predicts that your symptoms match {top['disease']}."

        return PredictionResult(
            message=message,
            top_disease=top["disease"],
            matched_symptoms=top["matched_symptoms"],
            precautions=precautions,
            advice=advice,
            recommended_doctor=recommended_doctor,
            all_predictions=ranked,
        )

    def chat(self, message: str, history: Sequence[Message]) -> PredictionResult:
        del history
        normalized = self._normalize_token(message)

        if normalized in {"hi", "hello", "hey", "assalamualaikum", "salam"}:
            return PredictionResult(
                message=(
                    "Hello! I am MetaMaid, your medical symptom chatbot. "
                    "Tell me your symptoms, for example: fever, cough, sore throat."
                ),
                top_disease="Greeting",
                matched_symptoms=[],
                precautions=[],
                advice="Describe your symptoms separated by commas.",
                recommended_doctor="General Physician",
                all_predictions=[],
            )

        if "help" in normalized:
            return PredictionResult(
                message=(
                    "Send symptoms like 'fever, cough, body pain' and I will predict the most likely disease, "
                    "then show precautions, advice, and recommended doctor."
                ),
                top_disease="Help",
                matched_symptoms=[],
                precautions=[],
                advice="Example: headache, nausea, sensitivity to light",
                recommended_doctor="General Physician",
                all_predictions=[],
            )

        return self.predict_from_text(message)
