import numpy as np
import gradio as gr
import uvicorn
from keras.utils import pad_sequences

from main import (
    app,
    dl_model,
    preprocess_text,
    EMOTION_LABELS,
    EMOTION_EMOJIS,
    MAX_SEQUENCE_LENGTH,
)


def gradio_predict(text: str):
    if not text or not text.strip():
        return "Please enter a sentence.", {}

    cleaned = preprocess_text(text)
    if not cleaned:
        return "Please enter at least one valid word.", {}

    seq = pad_sequences(
        dl_model["tokenizer"].texts_to_sequences([cleaned]),
        maxlen=MAX_SEQUENCE_LENGTH,
        padding="post",
        truncating="post",
    )
    probs = dl_model["model"].predict(seq, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    label = EMOTION_LABELS[top_idx]

    scores = {lbl.capitalize(): float(p) for lbl, p in zip(EMOTION_LABELS, probs)}
    return f"{EMOTION_EMOJIS[label]} {label.upper()} ({probs[top_idx] * 100:.1f}%)", scores


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Textbox(lines=3, placeholder="Type a sentence to analyze...", label="Input Sentence"),
    outputs=[
        gr.Textbox(label="Predicted Emotion"),
        gr.Label(num_top_classes=6, label="Confidence Breakdown"),
    ],
    title="6-Class Emotion Detection (GloVe-BiGRU)",
)

app = gr.mount_gradio_app(app, demo, path="/gradio")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)