import json
import os
import re
import pickle
import tempfile
import zipfile
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from keras.models import load_model
from keras.utils import pad_sequences


from keras.layers import Embedding, Dense, GRU

"""
1. Constants
"""
MODEL_PATH = "Artifacts/GloVe_BiGRU_Model.keras"
TOKENIZER_PATH = "Artifacts/glove_tokenizer.pkl"
MAX_SEQUENCE_LENGTH = 50
CONFIDENCE_THRESHOLD = 0.60

EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

EMOTION_EMOJIS = {
    "sadness": "😢",
    "joy": "😄",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}

"""
2. Text Preprocessing
Matches dataset formatting (e.g., "I'm" -> "im", punctuation -> space).
"""
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"'", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

"""
3. Request and Response Schemas
"""
class TextInput(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The sentence to analyze",
        json_schema_extra={"example": "I feel so happy and excited!"},
    )

class PredictionResponse(BaseModel):
    text: str
    cleaned_text: str
    predicted_emotion: str
    emoji: str
    confidence: float
    is_confident: bool
    all_probabilities: dict[str, float]

class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_loaded: bool

"""
4. Model Loading & Lifespan Management
Strips newer Keras 3.15+ quantization_config keys so the model loads cleanly on any Keras 3 server.
"""
def load_keras_compat(model_path: str):
    def strip_quantization_config(obj):
        if isinstance(obj, dict):
            obj.pop("quantization_config", None)
            for value in obj.values():
                strip_quantization_config(value)
        elif isinstance(obj, list):
            for item in obj:
                strip_quantization_config(item)

    compat_path = os.path.join(tempfile.gettempdir(), f"compat_{os.path.basename(model_path)}")
    with zipfile.ZipFile(model_path, "r") as zin, zipfile.ZipFile(compat_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data.decode("utf-8"))
                strip_quantization_config(cfg)
                data = json.dumps(cfg).encode("utf-8")
            zout.writestr(item, data)

    return load_model(compat_path)


dl_model = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading GloVe-BiGRU model and tokenizer...")
    dl_model["model"] = load_keras_compat(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as file:
        dl_model["tokenizer"] = pickle.load(file)
    print("GloVe-BiGRU model and tokenizer loaded successfully. API ready!")

    yield

    dl_model.clear()
    print("Server shutdown: Cleared model from memory.")

"""
5. FastAPI App, CORS & Static Files Setup
"""
app = FastAPI(
    title="Emotion Classification API",
    description="6-Class Emotion Detection powered by BiLSTM",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

"""
6. API Endpoints
"""
@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse("static/index.html")

@app.get("/health", response_model=HealthResponse)
def health_check():
    is_ready = "model" in dl_model and "tokenizer" in dl_model
    return HealthResponse(
        status="Server is running" if is_ready else "Initializing",
        model_name="BiLSTM (300D, Bidirectional)",
        model_loaded=is_ready,
    )

@app.post("/predict", response_model=PredictionResponse)
def predict_emotion(text_input: TextInput):
    bilstm_model = dl_model.get("model")
    tokenizer_model = dl_model.get("tokenizer")

    if bilstm_model is None or tokenizer_model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded yet. Please try again later.",
        )

    # 1. Clean text and ensure valid words remain
    cleaned_text = preprocess_text(text_input.text)
    if not cleaned_text:
        raise HTTPException(
            status_code=400,
            detail="Input must contain at least one alphanumeric word.",
        )

    # 2. Tokenize and pad sequence to length 50
    tokenized_text = tokenizer_model.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(
        tokenized_text,
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )

    # 3. Predict probabilities (verbose=0 keeps server logs clean)
    probabilities = bilstm_model.predict(padded_sequence, verbose=0)[0]

    top_idx = int(np.argmax(probabilities))
    predicted_label = EMOTION_LABELS[top_idx]
    top_confidence = round(float(probabilities[top_idx]), 4)

    all_probabilities = {
        label: round(float(prob), 4)
        for prob, label in zip(probabilities, EMOTION_LABELS)
    }

    return PredictionResponse(
        text=text_input.text,
        cleaned_text=cleaned_text,
        predicted_emotion=predicted_label,
        emoji=EMOTION_EMOJIS[predicted_label],
        confidence=top_confidence,
        is_confident=top_confidence >= CONFIDENCE_THRESHOLD,
        all_probabilities=all_probabilities,
    )