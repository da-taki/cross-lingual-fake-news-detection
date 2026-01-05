import argparse
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def run_logreg(data_path, output_path, language):
    df = pd.read_csv(data_path)

    X = df["content_clean"]
    y = df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    tfidf = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2)
    )

    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)

    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_val_tfidf)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="binary"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pd.DataFrame([{
        "language": language,
        "model": "TF-IDF + Logistic Regression",
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }]).to_csv(output_path, index=False)

    print(f"[OK] Saved results to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to *_clean.csv")
    parser.add_argument("--out", required=True, help="Path to output CSV")
    parser.add_argument("--lang", required=True, help="Language name")

    args = parser.parse_args()
    run_logreg(args.data, args.out, args.lang)
