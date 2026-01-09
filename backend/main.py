from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer (using a pre-trained model for Vietnamese text classification)
MODEL_NAME = "../model/"  # PhoBERT for Vietnamese

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

print("Model and tokenizer loaded.")

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict_fake_news(news: NewsInput):
    inputs = tokenizer(news.text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
        probs = outputs.logits.softmax(dim=-1).squeeze(0)
    label = predictions.item()
    result = "Fake News" if label == 1 else "Real News"
    confidence_percent = (probs * 100).tolist()
    return {"prediction": result, "confidence": confidence_percent}

@app.get("/")
def read_root():
    return {"message": "Vietnamese Fake News Detector API"}
