# Multilingual Fake News Detection

## Motivation
Fake news detection models often report strong results on English datasets, but
their robustness across languages is rarely examined. This project investigates
how classical machine learning models degrade under cross-lingual translation
and morphological variation.

## Dataset
- English fake/real news dataset
- Hindi and Korean versions created via neural machine translation
- Multilingual experiments conducted on a stratified subset due to computational constraints

## Methods
### Baselines
- TF-IDF + Logistic Regression
- TF-IDF + Naive Bayes

All models use identical preprocessing and hyperparameters across languages to
ensure fair comparison.

### Preprocessing
- English: standard text cleaning
- Hindi/Korean: minimal, language-agnostic cleaning to preserve script integrity

## Key Findings
- Classical models perform extremely well in English but rely heavily on
  stylistic cues.
- Performance degrades in Hindi due to translation noise.
- Performance collapses in Korean, especially recall, due to morphological
  complexity and tokenization limitations.

These results demonstrate the representational limits of bag-of-words models and
motivate transformer-based multilingual approaches.

## Execution Environment
Translation and transformer experiments were run in Google Colab due to GPU
requirements. This repository contains all scripts and result files needed to
reproduce the reported metrics.

## Next Steps
- Multilingual transformers (mBERT, XLM-R)
- Comparison against classical baselines
- Analysis of recall recovery in morphologically complex languages
