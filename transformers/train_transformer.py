import os
import argparse
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

# -----------------------------
# Dataset
# -----------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding=False,
            max_length=self.max_len
        )
        enc["labels"] = int(self.labels[idx])
        return enc


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# -----------------------------
# Main
# -----------------------------
def main(args):
    df = pd.read_csv(args.data)

    X = df["content_clean"].astype(str).tolist()
    y = df["label"].tolist()

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    train_ds = NewsDataset(X_train, y_train, tokenizer)
    val_ds = NewsDataset(X_val, y_val, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="/content/tmp",
        do_eval=True,
        learning_rate=2e-5,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        report_to="none"
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()
    metrics = trainer.evaluate()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    pd.DataFrame([{
        "language": args.lang,
        "model": args.model,
        "accuracy": metrics["eval_accuracy"],
        "precision": metrics["eval_precision"],
        "recall": metrics["eval_recall"],
        "f1": metrics["eval_f1"]
    }]).to_csv(args.out, index=False)

    print(f"\n[OK] Results saved to {args.out}")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="Path to *_clean.csv")
    parser.add_argument("--model", required=True, help="HF model name")
    parser.add_argument("--lang", required=True, help="Language name")
    parser.add_argument("--out", required=True, help="Output CSV path")

    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)

    args = parser.parse_args()
    main(args)
