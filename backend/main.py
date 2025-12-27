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
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer (using a pre-trained model for Vietnamese text classification)
MODEL_NAME = "vinai/phobert-base"  # PhoBERT for Vietnamese
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
try:
    model = AutoModelForSequenceClassification.from_pretrained("../model", num_labels=2)
    print("Loaded trained model from ../model")
except:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    print("Using pre-trained model")

class NewsInput(BaseModel):
    text: str

@app.post("/predict")
def predict_fake_news(news: NewsInput):
    inputs = tokenizer(news.text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=-1)
    label = predictions.item()
    result = "Fake News" if label == 1 else "Real News"
    return {"prediction": result, "confidence": outputs.logits.softmax(dim=-1).tolist()}

@app.get("/")
def read_root():
    return {"message": "Vietnamese Fake News Detector API"}