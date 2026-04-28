#   <h1 align="center">Fake News Detection with Classical NLP</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue" />
  <img src="https://img.shields.io/badge/ML-Scikit--Learn-orange" />
  <img src="https://img.shields.io/badge/Task-Text%20Classification-green" />
  <img src="https://img.shields.io/badge/Status-Complete-success" />
</p>

---

## Overview

This project builds a text classification system that distinguishes fake news from real news using linguistic patterns and writing style.

It is important to be clear about what this system does and does not do. It does not verify factual correctness. Instead, it learns statistical differences in how fake and real news are written. Those differences turn out to be strong enough to produce very high accuracy, which makes the results interesting but also potentially misleading without proper context.

The full pipeline covers data preparation, feature engineering, model comparison, evaluation, and deployment-ready inference.

---

## Dataset

**Source:** Kaggle — Fake and Real News Dataset  
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

The dataset consists of:

- `Fake.csv` — fake news articles  
- `True.csv` — real news articles  

Each article combines title and body into a single `content` field.

### Key Observation

A large portion of real news originates from Reuters:


WASHINGTON (Reuters) —


This introduces a strong dataset bias. A simple feature like the presence of “reuters” can already separate classes with high accuracy.

---

## Methodology

### Data Preparation
- Merge and label datasets  
- Shuffle and deduplicate  
- Combine title + text  
- Stratified train/validation/test split  

---

### Preprocessing
- Lowercasing  
- Remove Reuters patterns  
- Remove URLs  
- Normalize whitespace  

---

### Feature Engineering

Two TF-IDF representations are combined:

**Word-Level TF-IDF**
- N-grams up to trigrams  
- 50,000 features  

**Character-Level TF-IDF**
- Character n-grams (3–5)  
- 30,000 features  

Combined feature space: ~80,000 dimensions

---

## Models

Models evaluated:

- Logistic Regression  
- Linear SVM  
- Naive Bayes  
- Random Forest  
- XGBoost  
- LightGBM  
- Voting and Stacking  

Linear models dominate due to the nature of sparse TF-IDF features.

---

## Results

| Model | Test Accuracy | Test F1 | ROC-AUC |
|------|--------------|--------|--------|
| **Linear SVM** | **0.9980** | **0.9980** | **0.9998** |
| Logistic Regression | 0.9957 | 0.9957 | 0.9997 |
| Stacking | 0.9959 | 0.9959 | 0.9997 |
| LightGBM | 0.9901 | 0.9901 | 0.9995 |
| XGBoost | 0.9898 | 0.9898 | 0.9994 |
| Random Forest | 0.9824 | 0.9824 | 0.9984 |
| Naive Bayes | 0.9652 | 0.9652 | 0.9935 |

### Summary

- Linear SVM performs best  
- Tree models are slower and less effective  
- Ensemble methods provide limited improvement  

---

## Important Limitation

The model does not detect factual truth.

It distinguishes:

- structured, formal journalism  
- vs  
- informal or sensational writing  

This distinction is driven largely by dataset bias, not real-world verification capability.

---

## Running the Project

### Install dependencies

```bash
pip install numpy pandas scikit-learn scipy joblib xgboost lightgbm matplotlib seaborn
Run notebook
jupyter notebook code.ipynb
Inference
import joblib
from scipy import sparse

model = joblib.load("model.pkl")
tfidf_word = joblib.load("tfidf_word.pkl")
tfidf_char = joblib.load("tfidf_char.pkl")

result = predict("Your article text")
print(result)
Project Structure
fake-news-detection/
├── code.ipynb
├── app.py
├── Fake.csv
├── True.csv
├── model.pkl
├── tfidf_word.pkl
├── tfidf_char.pkl
└── README.md
Future Improvements
Transformer-based models (BERT, RoBERTa)
Reduce dataset bias
Add fact-checking APIs
Evaluate across multiple datasets
Conclusion

The system performs extremely well on this dataset, but the performance is largely driven by structural patterns rather than genuine fact verification.

Understanding this distinction is essential. The value of the project lies not only in the model performance, but in recognizing the limits of what the model has actually learned.
