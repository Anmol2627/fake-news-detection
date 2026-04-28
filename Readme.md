# Fake News Detection with Classical NLP

## Overview

This project builds a text classification pipeline that distinguishes fake news from real news using writing style and linguistic patterns. It does not verify facts — and that distinction matters. The system learns that real and fake news are written differently: different vocabulary, different sentence structure, different emotional register. That signal alone turns out to be surprisingly powerful, which also raises important questions about what "detection" actually means here.

The pipeline covers everything from raw data loading to a serialized model ready for inference, with a structured comparison of seven classifiers evaluated on a held-out test set.

---

## Dataset

**Source:** [Kaggle — Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

The dataset consists of two CSV files:

- `Fake.csv` — articles labeled as fake news (label `0`)
- `True.csv` — articles labeled as real news (label `1`)

Both files contain `title` and `text` columns. The title and body are concatenated into a single `content` field for modeling.

**Critical observation:** The real news corpus is heavily sourced from Reuters wire reporting. A large proportion of real articles begin with a Reuters dateline (e.g., `WASHINGTON (Reuters) —`). This is not a neutral stylistic quirk — it is a dataset-level bias that makes the classification problem considerably easier than it would be in production. A single binary feature for "contains reuters" achieves high AUC on its own, which is a red flag worth taking seriously.

---

## Methodology

### Data Preparation

The two CSVs are loaded, labeled, concatenated, and shuffled. Missing titles or body text are filled with empty strings before concatenation. Duplicates are dropped on the combined `content` field.

The data is split into three stratified partitions:

| Split      | Size |
|------------|------|
| Train      | ~70% |
| Validation | ~15% |
| Test       | ~15% |

Stratification ensures balanced class ratios across all three splits. Label alignment between `X` and `y` is enforced explicitly by keeping them in a shared DataFrame through preprocessing — a small but important safeguard against index misalignment bugs.

### Preprocessing

The cleaning function applies the following steps in order:

1. Lowercase the text
2. Strip Reuters byline patterns — `(reuters)` and standalone `reuters` — to reduce source-based leakage from propagating directly into the model
3. Remove URLs
4. Normalize whitespace

Removing the Reuters token is a conscious attempt to push the model toward stylistic features rather than simple source identification. As the bias analysis section shows, this only partially succeeds — the structural differences between wire-service and non-wire-service writing remain.

### Feature Engineering

Two TF-IDF representations are built and concatenated into a single sparse feature matrix:

**Word-level TF-IDF**
- Unigrams, bigrams, and trigrams
- 50,000 maximum features
- `min_df=3`, `max_df=0.95` to suppress very rare and near-universal terms
- Sublinear TF scaling

**Character-level TF-IDF**
- Character n-grams, lengths 3 to 5
- 30,000 maximum features
- Captures punctuation patterns, morphology, and writing style at a sub-word level

The two feature matrices are horizontally stacked with `scipy.sparse.hstack`, giving a combined feature space of 80,000 dimensions. Both vectorizers are fit exclusively on the training set, then applied to validation and test.

Note: Naive Bayes and Random Forest are trained on word features only, as they either require non-negative input or degrade on very high-dimensional sparse data.

### Evaluation Pipeline

Each model is evaluated on the validation set during training, then re-evaluated on the held-out test set in a separate final pass. Metrics reported:

- **Accuracy** — proportion of correct predictions
- **F1 (weighted)** — F1 averaged by class support; appropriate when classes are near-balanced
- **F1 (macro)** — unweighted average F1 across classes; more sensitive to per-class performance
- **ROC-AUC** — area under the receiver operating characteristic curve; measures ranking quality independent of threshold

---

## Models

Seven classifiers are trained and compared. Multiple models are included not to stack them all in production, but to understand where the performance ceiling lies and which approaches generalize best to this type of sparse, high-dimensional text data.

| Model | Notes |
|---|---|
| **Logistic Regression** | Primary baseline. `saga` solver, `C=2.0`, balanced class weights. Consistently strong on sparse TF-IDF. |
| **Linear SVM** | SGD-trained hinge-loss classifier, probability-calibrated via 3-fold cross-validation. Competitive with LR on linear problems. |
| **Naive Bayes** | Multinomial NB with `alpha=0.1`. Fast and interpretable; works on word features only due to non-negativity requirement. |
| **Random Forest** | 150 trees with balanced class weights. Trained on word features to keep dimensionality manageable. Generally underperforms linear models on TF-IDF. |
| **XGBoost** | Gradient-boosted trees with 200 estimators. Included for completeness; tends to be slower and no better than LR here. |
| **LightGBM** | More memory-efficient than XGBoost for sparse input. Better practical performance among tree-based options. |
| **Voting / Stacking** | Hard voting, soft voting, and a stacking ensemble (LR + SGD, meta-learner: LR). Stacking uses 3-fold CV to generate meta-features. |

---

## Results

All models are ranked by test F1 (weighted). The full comparison is below.

| # | Model | Val Accuracy | Val F1 (weighted) | Test Accuracy | Test F1 (weighted) | Test F1 (macro) | ROC-AUC | Train Time |
|---|---|---|---|---|---|---|---|---|
| 1 | **Linear SVM** | 0.9969 | 0.9969 | **0.9980** | **0.9980** | **0.9979** | **0.9998** | 5.5s |
| 2 | Stacking | 0.9949 | 0.9949 | 0.9959 | 0.9959 | 0.9959 | 0.9997 | 117.0s |
| 3 | Logistic Regression | 0.9952 | 0.9952 | 0.9957 | 0.9957 | 0.9957 | 0.9997 | 39.0s |
| 4 | Soft Voting | 0.9937 | 0.9937 | 0.9944 | 0.9944 | 0.9943 | 0.9995 | 83.0s |
| 5 | Hard Voting | 0.9939 | 0.9939 | 0.9937 | 0.9937 | 0.9936 | 0.9937 | 64.6s |
| 6 | LightGBM | 0.9903 | 0.9903 | 0.9901 | 0.9901 | 0.9900 | 0.9995 | 176.7s |
| 7 | XGBoost | 0.9884 | 0.9884 | 0.9898 | 0.9898 | 0.9897 | 0.9994 | 377.7s |
| 8 | Random Forest | 0.9811 | 0.9811 | 0.9824 | 0.9824 | 0.9823 | 0.9984 | 36.6s |
| 9 | Naive Bayes | 0.9652 | 0.9652 | 0.9652 | 0.9652 | 0.9650 | 0.9935 | 0.2s |

**Winner: Linear SVM** — 99.80% test accuracy, 0.9998 ROC-AUC, trained in 5.5 seconds. It beats every other model including the stacking ensemble and Logistic Regression, while being by far the fastest non-trivial classifier in the comparison.

A few patterns worth noting from the table:

- Linear models (SVM, LR) dominate tree-based methods across every metric, which is the expected outcome on sparse high-dimensional TF-IDF features. Gradient-boosted trees (XGBoost, LightGBM) are slower by an order of magnitude and still fall 0.8–0.9 points short on F1.
- The Stacking ensemble edges out Logistic Regression by a small margin (0.9959 vs 0.9957 F1) but costs 3× the training time. Not a meaningful tradeoff in practice.
- Naive Bayes, despite being the fastest model by a wide margin (0.2 seconds), still achieves 96.5% accuracy — a reasonable result for something that makes fully independent feature assumptions.
- Hard Voting drops noticeably on ROC-AUC (0.9937) compared to every other model. This is expected: hard voting discards probability estimates, which ROC-AUC depends on.
- XGBoost's 377-second training time is the worst in the comparison, with no corresponding performance gain over LR or SVM.

Feature inspection via Logistic Regression coefficients confirms the expected pattern: terms associated with neutral, dateline-style reporting dominate the REAL class indicators, while emotionally charged and informal vocabulary drives the FAKE class predictions.

---

## Limitations

This section is the most important one.

**The model does not detect factual falsehoods.** It detects writing patterns. A well-written fabrication that mimics wire-service prose would likely be classified as real. A genuine Reuters article with no byline, or one written more casually, could be flagged as fake.

**Dataset bias is the primary driver of performance.** The real news corpus is dominated by Reuters wire stories. The fake news corpus contains writing from a different stylistic tradition entirely — more emotional, less formal, different topic distribution. The model is largely learning to distinguish Reuters-style professional journalism from informal or sensationalist writing. That is a useful signal in some contexts, but it is not the same as fact-checking.

**Removing the Reuters token is not enough.** Even after stripping explicit Reuters mentions, the structural fingerprint of wire-service reporting — sentence rhythm, vocabulary, inverted pyramid structure — remains in the data. The model picks up on this.

**Topic distribution is not balanced.** Real and fake news in this dataset cover different topic clusters. The model partially learns topic boundaries in addition to style, which would not generalize to fake news written about the same topics as real news.

High benchmark accuracy on this dataset should not be interpreted as evidence that the system would work reliably in production.

---

## Running the Project

### Prerequisites

```bash
pip install numpy pandas scikit-learn scipy joblib xgboost lightgbm matplotlib seaborn
```

### Running the Notebook

```bash
jupyter notebook code.ipynb
```

Run cells sequentially. The notebook expects `Fake.csv` and `True.csv` in the same directory.

### Inference

After running the notebook, three artifacts are saved: `model.pkl`, `tfidf_word.pkl`, and `tfidf_char.pkl`. Load them and use the `predict()` function defined at the end of the notebook:

```python
import joblib
from scipy import sparse

model     = joblib.load("model.pkl")
tfidf_word = joblib.load("tfidf_word.pkl")
tfidf_char = joblib.load("tfidf_char.pkl")

result = predict("Your news article text here.")
print(result)
# {'class': 'REAL', 'confidence': 0.97, 'P(FAKE)': 0.03, 'P(REAL)': 0.97}
```

---

## Project Structure

```
fake-news-detection/
├── code.ipynb          # Main notebook: EDA, training, evaluation, bias analysis
├── Fake.csv            # Fake news articles (not included, download from Kaggle)
├── True.csv            # Real news articles (not included, download from Kaggle)
├── model.pkl           # Serialized best model (generated after running notebook)
├── tfidf_word.pkl      # Word-level TF-IDF vectorizer
├── tfidf_char.pkl      # Character-level TF-IDF vectorizer
└── README.md
```

---

## Possible Improvements

- **Transformer-based models** — BERT or RoBERTa fine-tuned on this corpus would capture semantic context that TF-IDF misses entirely, though they would not solve the underlying dataset bias problem.
- **Dataset debiasing** — Removing or downsampling Reuters articles from the real class, or sourcing fake news from domains closer in style to wire reporting, would produce a harder and more honest benchmark.
- **External fact-checking integration** — Connecting article claims to a knowledge base or fact-checking API (e.g., Google Fact Check Tools) would move the system closer to actual veracity detection, as opposed to style classification.
- **Cross-dataset evaluation** — Testing on LIAR, FakeNewsNet, or similar datasets would immediately expose how much of the learned signal fails to transfer.
- **Source-agnostic features** — Explicitly engineering features around emotional tone, claim density, hedging language, and citation patterns rather than raw n-grams would produce a more generalizable model.

---

## Conclusion

The pipeline works well for what it is: a clean, reproducible implementation of classical NLP classification applied to a structured benchmark dataset. It demonstrates a thoughtful feature engineering approach, a disciplined evaluation setup, and a clear-eyed analysis of why the numbers look the way they do.

The bias analysis embedded in the notebook is as important as the model itself. Knowing that "reuters" alone is a near-sufficient predictor changes how you interpret a 99% accuracy figure — it does not mean the model is intelligent, it means the dataset has a strong exploitable structure. Building systems that are actually robust to adversarial fake news requires either better data or a fundamentally different approach. This project makes that limitation explicit rather than burying it.
