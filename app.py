import gradio as gr
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel

# Configuration
BASE_MODEL_NAME = "SI2M-Lab/DarijaBERT"
ADAPTER_PATH = "./model_v5_final/"

# Labels mapping
id2label = {0: 'negative', 1: 'neutral', 2: 'positive'}
label2id = {'negative': 0, 'neutral': 1, 'positive': 2}
emojis = {'negative': '😡 Négatif', 'neutral': '😐 Neutre', 'positive': '😊 Positif'}

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading models on {device}...")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)

# Load base model
base_model = AutoModelForSequenceClassification.from_pretrained(
    BASE_MODEL_NAME,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()
model.to(device)
model.eval()
print("Model loaded successfully - LoRA merged")

# Note: if merge_and_unload() still gives random results, the classifier head needs
# to be re-saved from the original training session using:
# lora_config = LoraConfig(..., modules_to_save=["classifier", "pooler"])
# trainer.save_model(SAVE_PATH)

def predict_sentiment(text):
    if not text.strip():
        return "", {}, 0.0

    # Tokenize
    inputs = tokenizer(
        text,
        max_length=128,
        truncation=True,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
        
    # Get predicted class
    pred_id = int(np.argmax(probabilities))
    pred_label_str = id2label[pred_id]
    
    # Format output
    result_emoji = emojis[pred_label_str]
    confidence = probabilities[pred_id] * 100
    
    # Format scores dictionary for gr.Label
    scores_dict = {
        emojis['negative']: float(probabilities[0]),
        emojis['neutral']: float(probabilities[1]),
        emojis['positive']: float(probabilities[2])
    }
    
    return result_emoji, scores_dict, confidence

# Gradio Interface
custom_css = """
#rtl-container textarea {
    direction: rtl;
    text-align: right;
}
"""

with gr.Blocks(css=custom_css, title="تحليل المشاعر بالدارجة المغربية 🇲🇦") as demo:
    gr.Markdown("<h1 style='text-align: center;'>تحليل المشاعر بالدارجة المغربية 🇲🇦</h1>")
    gr.Markdown("<h3 style='text-align: center;'>DarijaBERT + LoRA Fine-tuned — Analyse de sentiment en Darija marocain</h3>")
    
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                lines=5, 
                placeholder="اكتب شي حاجة بالدارجة هنا...", 
                label="النص (Text)", 
                elem_id="rtl-container"
            )
            analyze_btn = gr.Button("🔍 تحليل المشاعر", variant="primary")
            
        with gr.Column():
            result_label = gr.Textbox(label="النتيجة (Result)")
            confidence_label = gr.Number(label="نسبة الثقة (Confidence %)", precision=2)
            scores_bar = gr.Label(label="التفاصيل (Class Scores)")
            
    analyze_btn.click(
        fn=predict_sentiment,
        inputs=input_text,
        outputs=[result_label, scores_bar, confidence_label]
    )
    
    gr.Examples(
        examples=[
            ["المنتج زوين بزاف وعجبني، التوصيل كان سريع."],
            ["هادشي خايب ومكيخدمش مزيان، ضيعت فلوسي."],
            ["المنتج عادي، ماشي شي حاجة واو ولكن كيقضي الغرض."],
            ["شكرا بزاف، الجودة ممتازة حسن من داكشي لي توقعت!"],
            ["تعطلتو عليا بزاف فالتوصيل والتعامل ديالكم ناقص."],
            ["مزيان"]
        ],
        inputs=input_text
    )

if __name__ == "__main__":
    demo.launch()
