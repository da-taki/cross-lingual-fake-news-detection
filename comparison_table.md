# Cross-Lingual Fake News Detection – Results

## TF-IDF Baselines

| Language | Model | Accuracy | Precision | Recall | F1 |
|--------|------|----------|-----------|--------|----|
| English | Logistic Regression | ~0.99 | ~0.99 | ~0.99 | ~0.99 |
| English | Naive Bayes | ~0.95 | ~0.94 | ~0.95 | ~0.94 |
| Hindi | Logistic Regression | 0.795 | 0.792 | 0.773 | 0.782 |
| Hindi | Naive Bayes | 0.795 | 0.790 | 0.776 | 0.783 |
| Korean | Logistic Regression | 0.641 | 0.857 | 0.308 | 0.454 |
| Korean | Naive Bayes | 0.652 | 0.841 | 0.346 | 0.490 |

---

## Transformer Models

### mBERT (`bert-base-multilingual-cased`)

| Language | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|
| Hindi | 0.867 | 0.863 | 0.857 | 0.860 |
| Korean | 0.668 | 0.675 | 0.603 | 0.637 |

---

### XLM-RoBERTa (`xlm-roberta-base`)

| Language | Accuracy | Precision | Recall | F1 |
|--------|----------|-----------|--------|----|
| Hindi | 0.853 | 0.830 | 0.871 | 0.850 |
| Korean | 0.639 | 0.787 | 0.346 | 0.481 |

---

## Observations

- Classical models struggle under heavy morphological variation
- Korean is significantly harder than Hindi
- Transformers recover semantic signal lost by TF-IDF
- mBERT is consistently more robust than XLM-R in cross-lingual settings