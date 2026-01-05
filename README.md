# Cross-Lingual Fake News Detection

This project investigates how fake news detection models behave when applied
across languages with very different linguistic structure.

Instead of focusing on a single-language classifier, the goal is to study
**cross-lingual robustness**: how performance changes when models trained on
English news are evaluated on translated Hindi and Korean text.

---

## Languages Studied
- **English** (original dataset)
- **Hindi** (machine-translated)
- **Korean** (machine-translated)

Hindi and Korean were chosen due to their structural and morphological distance
from English.

---

## Models Evaluated

### Classical NLP Baselines
- TF-IDF + Logistic Regression
- TF-IDF + Multinomial Naive Bayes

### Multilingual Transformers
- **mBERT** (`bert-base-multilingual-cased`)
- **XLM-RoBERTa** (`xlm-roberta-base`)

Transformer models were fine-tuned for binary fake/real classification.

---

## Experimental Setup

- Balanced subsets used for all languages
- Labels preserved across translation
- Evaluation performed independently per language
- Metrics reported: Accuracy, Precision, Recall, F1-score

Large datasets are excluded from version control due to size constraints.
All reported results are stored as CSV files for transparency.

---

## Key Results (High-Level)

- TF-IDF baselines perform well on English but degrade sharply on Korean
- Hindi retains more signal after translation than Korean
- Transformer models significantly outperform classical baselines
- mBERT shows stronger robustness than XLM-R on Korean
- Cross-lingual fake news detection is **not symmetric across languages**

Full metrics are available in `comparison_table.md`.

---

## Why This Matters

Most fake news detectors are evaluated in monolingual settings.
These results highlight how linguistic distance and translation noise
can severely impact model reliability, even when using large multilingual models.

---

## Status

Baseline and transformer experiments completed.
Future work may explore zero-shot transfer and native non-English datasets.
