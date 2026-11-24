---
# 🎭 Multi-Class Emotion Detection (`GloVe-BiGRU`)

An end-to-end Natural Language Processing (NLP) web application that classifies English text into **six emotions** (*Sadness, Joy, Love, Anger, Fear, and Surprise*) with real-time probability scores.

🔗 **Live Interactive App:** [https://nishak14-emotion-detection.hf.space](https://nishak14-emotion-detection.hf.space)

---

## 📌 Project Highlights
* **Leak-Free Data Pipeline:** Trained on the `dair-ai/emotion` dataset (20,000 utterances) using a strict `16,000 / 2,000 / 2,000` train/validation/test split, fitting the 10,000-word Keras Tokenizer exclusively on training data.
* **Class Imbalance Handling:** Resolved a 9.4:1 skew between majority *Joy* and minority *Surprise* samples using `compute_class_weight('balanced')`, lifting *Surprise* recall to **87.9%**.
* **Transfer Learning & Architecture Selection:** Benchmarked five recurrent architectures and paired pre-trained **300D GloVe embeddings** with a **Bidirectional GRU**, achieving **92.95% accuracy** on the strictly unseen test set.
* **Production Web Service:** Built a FastAPI REST backend (`/predict`) with startup lifespan model caching, Keras cross-version config patching, and a responsive HTML/CSS/JavaScript frontend hosted via Gradio on Hugging Face Spaces.

---

## 📊 Model Progression & Benchmark Results

All models were trained using Sparse Categorical Crossentropy, the Adam optimizer, spatial/standard dropout regularization, and `EarlyStopping` (`patience=4`) on validation loss.

| Model Architecture | Validation Accuracy | Unseen Test Accuracy | Key Takeaway |
| :--- | :---: | :---: | :--- |
| SimpleRNN (Baseline) | 42.65% | — | Struggled with long-range sentence context |
| GRU | 78.55% | — | Gated memory captured longer dependencies |
| LSTM | 89.35% | — | Strong single-direction context retention |
| BiLSTM | 90.80% | — | Bidirectional context; slightly overfit on 16k rows |
| BiGRU | 92.15% | 91.05% | Streamlined 2-gate design generalized better |
| **300D GloVe + BiGRU (Final)** | **—** | **92.95%** | **Pre-trained semantic vectors improved rare-word accuracy** |

---

## 🗂️ Repository Structure

```text
├── Artifacts/                  # Saved Keras model (104 MB, tracked via Git LFS) & tokenizer
├── static/
│   └── index.html              # Responsive frontend UI with live confidence bars
├── app.py                      # Hugging Face Spaces / Gradio entrypoint (Port 7860)
├── main.py                     # FastAPI REST API, lifespan model loader & prediction logic
├── requirements.txt            # Production Python dependencies
├── .gitattributes              # Git LFS tracking rules for *.keras and *.pkl files
└── README.md                   # Project documentation & Space metadata
