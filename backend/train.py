import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from pyvi import ViTokenizer
from sklearn.metrics import accuracy_score, f1_score

# =========================
# 1. CHECK DEVICE (CPU / GPU)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# 2. LOAD DATA
# =========================
train_df = pd.read_csv("../dataset/train.csv")
val_df = pd.read_csv("../dataset/val.csv")

train_df = train_df[["post_message", "label"]].dropna()
val_df = val_df[["post_message", "label"]].dropna()

# =========================
# 3. WORD SEGMENTATION (BẮT BUỘC CHO PHOBERT)
# =========================
def preprocess_texts(texts):
    return [ViTokenizer.tokenize(str(t)) for t in texts]

train_texts = preprocess_texts(train_df["post_message"].tolist())
val_texts = preprocess_texts(val_df["post_message"].tolist())

train_labels = train_df["label"].tolist()
val_labels = val_df["label"].tolist()

# =========================
# 4. TOKENIZER
# =========================
tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

train_encodings = tokenizer(
    train_texts,
    truncation=True,
    padding="max_length",
    max_length=256,
)

val_encodings = tokenizer(
    val_texts,
    truncation=True,
    padding="max_length",
    max_length=256,
)

# =========================
# 5. DATASET CLASS
# =========================
class FakeNewsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = FakeNewsDataset(train_encodings, train_labels)
val_dataset = FakeNewsDataset(val_encodings, val_labels)

# =========================
# 6. LOAD MODEL
# =========================
model = AutoModelForSequenceClassification.from_pretrained(
    "vinai/phobert-base",
    num_labels=2,
)
model.to(device)

# =========================
# 7. METRICS
# =========================
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {
        "accuracy": acc,
        "f1": f1,
    }

# =========================
# 8. TRAINING ARGUMENTS
# =========================
training_args = TrainingArguments(
    output_dir="../model/results",
    num_train_epochs=3,
    per_device_train_batch_size=8 if torch.cuda.is_available() else 4,
    per_device_eval_batch_size=8 if torch.cuda.is_available() else 4,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="../model/logs",
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),   # ✅ dùng FP16 nếu có GPU
    report_to="none",
)

# =========================
# 9. TRAINER
# =========================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# =========================
# 10. TRAIN
# =========================
print("🚀 Starting training...")
trainer.train()

# =========================
# 11. SAVE MODEL
# =========================
print("💾 Saving model...")
trainer.save_model("../model")
tokenizer.save_pretrained("../model")

print("✅ Training completed. Model saved to ../model")
