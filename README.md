# 🇲🇦 Darija Sentiment Analysis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-UI-ff7c00.svg)](https://gradio.app/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-yellow.svg)](https://huggingface.co/transformers/)

A powerful and lightweight Moroccan Darija (Moroccan Arabic dialect) Sentiment Analysis model, built using **PEFT/LoRA** on top of the `SI2M-Lab/DarijaBERT` foundation model.

This project classifies text into three categories:
- 🔴 **Negative** (سلبية)
- ⚪ **Neutral** (محايدة)
- 🟢 **Positive** (إيجابية)

## ✨ Features

- **High Accuracy**: Fine-tuned on over 56,000 Moroccan Darija examples (including Jumia reviews and curated datasets).
- **Efficient & Fast**: Utilizes Low-Rank Adaptation (LoRA), keeping the model footprint incredibly small (~6 MB adapter) while preserving the massive base model knowledge.
- **Interactive UI**: Comes with a clean, easy-to-use Web Interface built with Gradio.
- **Ready-to-use**: Fully configured for local inference on CPU or GPU.

## 🚀 Quick Start

### Prerequisites

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running the Web App

Launch the Gradio interface directly from your terminal:

```bash
python app.py
```

The app will open automatically in your browser at `http://127.0.0.1:7860`.

## 🧠 Model Architecture

- **Base Model**: [`SI2M-Lab/DarijaBERT`](https://huggingface.co/SI2M-Lab/DarijaBERT)
- **Fine-Tuning Method**: PEFT / LoRA (Rank = 16, Alpha = 32)
- **Target Modules**: `query`, `key`, `value`, `classifier`, `pooler`
- **Dataset Size**: ~56,000 balanced sentences (Train: 44.8k, Val: 5.6k, Test: 5.6k)
- **Performance**: ~81% Accuracy across 3 classes

## 📂 Project Structure

```text
darija_sentiment_analysis/
├── app.py                     # Main Gradio application script
├── requirements.txt           # Python dependencies
├── model_v5_final/            # Saved LoRA adapter and tokenizer weights
├── notebooks/                 # Jupyter notebooks for data processing and training
└── README.md                  # Project documentation
```

## 🛠️ Usage Example

You can use the model directly via code without the UI:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

# Load Tokenizer & Base Model
tokenizer = AutoTokenizer.from_pretrained("./model_v5_final")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "SI2M-Lab/DarijaBERT", 
    num_labels=3, 
    ignore_mismatched_sizes=True
)

# Merge LoRA Adapter
model = PeftModel.from_pretrained(base_model, "./model_v5_final")
model = model.merge_and_unload()
model.eval()

# Predict
text = "المنتوج مزيان بزاف، وصل بسرعة"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    logits = model(**inputs).logits
    predicted_class = torch.argmax(logits, dim=-1).item()

print(predicted_class) # Output: 2 (Positive)
```

## 📝 License

This project is open-source and available under the [MIT License](LICENSE).
