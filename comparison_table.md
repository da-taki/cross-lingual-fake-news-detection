# Baseline Model Comparison

## Overview
This table summarizes baseline performance across languages using classical
TF-IDF–based models. English experiments use the full dataset, while Hindi and
Korean experiments use a translated stratified subset.

## Results

| Language | Model | Accuracy | F1 Score |
|--------|------|----------|----------|
| English | TF-IDF + Logistic Regression | ~0.99 | ~0.99 |
| English | TF-IDF + Naive Bayes | ~0.95 | ~0.94 |
| Hindi | TF-IDF + Logistic Regression | ~0.80 | ~0.78 |
| Hindi | TF-IDF + Naive Bayes | ~0.80 | ~0.78 |
| Korean | TF-IDF + Logistic Regression | ~0.64 | ~0.45 |
| Korean | TF-IDF + Naive Bayes | ~0.65 | ~0.49 |

## Key Observations
- English results are inflated due to source-specific stylistic cues.
- Hindi shows moderate degradation under translation noise.
- Korean exhibits severe recall collapse, highlighting the limitations of
  bag-of-words models in morphologically complex languages.

These results motivate the use of transformer-based multilingual models.
