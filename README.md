# fake-news-detection


📓 NOTEBOOK STRUCTURE
Cell 0 — Environment Setup & Reproducibility
python# Install all required packages at the top
# !pip install pandas numpy scikit-learn matplotlib seaborn wordcloud nltk xgboost lightgbm transformers torch imbalanced-learn shap optuna

Set global random seed (RANDOM_STATE = 42) used everywhere.
Import all libraries upfront in one organized block (stdlib → data → ML → NLP → viz → explainability).
Display library versions for reproducibility.


Cell 1 — Data Loading & Sanity Checks

Load Fake.csv and True.csv, add label column, concatenate, shuffle.
Print: shape, dtypes, null counts, duplicate rows (drop duplicates).
Show 3 sample rows from each class.
Assert no nulls remain after cleaning.


Cell 2 — Deep Exploratory Data Analysis (EDA)
Perform thorough EDA with clear visualizations:

Class Distribution: Bar chart + percentage annotation. Check for imbalance (if >60/40 split, flag it and apply SMOTE or class_weight='balanced' downstream).
Text Length Analysis: Compute word_count and char_count per article. Plot:

Overlapping KDE plots (REAL vs FAKE) for word count.
Box plots for char count by class.
Annotate median lines.


Top N-grams: Using CountVectorizer, extract and plot top 20 unigrams and bigrams separately for REAL and FAKE classes (4 bar charts total). Use stopword-filtered counts.
Word Clouds: Generate word clouds for REAL and FAKE separately using wordcloud library.
Source/Subject Distribution (if column exists): Plot top sources per class.
Vocabulary Overlap Analysis: Compute Jaccard similarity between REAL and FAKE vocabularies to understand separability.

All plots: use seaborn style, proper titles, axis labels, legends. Save figures using plt.tight_layout().

Cell 3 — Advanced Text Preprocessing
Build a reusable preprocess_text(text) function that applies the following pipeline in order:

Lowercase conversion.
URL removal (re.sub).
HTML tag stripping.
Remove punctuation and special characters (keep spaces).
Remove digits.
Tokenization (nltk.word_tokenize).
Stopword removal (nltk.corpus.stopwords, English).
Lemmatization (nltk.stem.WordNetLemmatizer) — use POS tagging (nltk.pos_tag) for accurate lemmatization (noun/verb/adj aware).
Remove tokens shorter than 2 characters.
Remove rare tokens: build vocabulary, remove tokens appearing in fewer than 5 documents.
Rejoin tokens into cleaned string.

Apply with df['cleaned_text'] = df['text_col'].apply(preprocess_text) using tqdm progress bar.
Download required NLTK data: punkt, stopwords, wordnet, averaged_perceptron_tagger.
Show before/after examples for 3 random rows.

Cell 4 — Train / Validation / Test Split

Split: 70% train / 15% validation / 15% test using stratified splits (preserve class balance).
Use train_test_split twice: first split off test (15%), then split remaining into train/val.
Print class distribution for each split to confirm stratification worked.
Never touch the test set until final evaluation.


Cell 5 — Feature Engineering
Implement and compare two feature extraction strategies:
Strategy A: TF-IDF with N-grams
pythontfidf = TfidfVectorizer(
    ngram_range=(1, 3),       # unigrams, bigrams, trigrams
    max_features=100000,
    sublinear_tf=True,        # log normalization
    min_df=3,
    max_df=0.95,
    analyzer='word',
    strip_accents='unicode'
)
Fit on training set only. Transform train/val/test separately.
Strategy B: TF-IDF + Handcrafted Features (Feature Union)
Engineer additional numeric features per article:

word_count, char_count, avg_word_length
exclamation_count, question_count, uppercase_ratio
unique_word_ratio (type-token ratio)
stopword_ratio
digit_ratio

Use scipy.sparse.hstack to concatenate TF-IDF sparse matrix with scaled numeric features (StandardScaler). Wrap in a FeatureUnion or manual hstack pipeline.
Label these as X_tfidf and X_enriched respectively. Train baseline models on both and report which performs better.

Cell 6 — Baseline Models
Train the following on X_tfidf (train split only):
6.1 Logistic Regression
pythonLogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', class_weight='balanced', random_state=42)
6.2 Multinomial Naive Bayes
pythonMultinomialNB(alpha=0.1)
Note: MNB requires non-negative features; use X_tfidf only (not enriched).
6.3 Linear SVM (SGD)
pythonSGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, max_iter=100, random_state=42, class_weight='balanced')
SVM often outperforms LR on high-dimensional sparse text — include it.
For each model:

Fit on train.
Evaluate on validation set (NOT test).
Report: Accuracy, Precision, Recall, F1 (weighted and macro), ROC-AUC.
Store all metrics in a results dictionary for the final comparison table.


Cell 7 — Advanced Ensemble Models
7.1 Random Forest
pythonRandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2, class_weight='balanced', random_state=42, n_jobs=-1)
Use X_enriched (TF-IDF + features). Note: RF on sparse TF-IDF is slow — use max_features='sqrt' and limit max_features in TF-IDF to 50,000 for RF only.
7.2 LightGBM (preferred over XGBoost for speed/performance on text)
pythonLGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=63,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    subsample=0.8,
    colsample_bytree=0.8
)
Convert sparse matrix to float32 before passing to LightGBM.
7.3 Voting Ensemble (Hard + Soft)
Combine best-performing models from Cells 6 and 7 using VotingClassifier:

Hard voting: majority vote.
Soft voting: average predicted probabilities (use models that support predict_proba).
Report both variants.

7.4 Stacking Ensemble
Use StackingClassifier with:

Base learners: LR, LinearSVC, LightGBM.
Meta-learner: Logistic Regression with cv=5.

Evaluate all on validation set. Store metrics.

Cell 8 — Hyperparameter Optimization
Apply Optuna (preferred over GridSearchCV for efficiency) to tune the top-2 performing models from Cells 6–7:
pythonimport optuna

def objective(trial):
    C = trial.suggest_float('C', 1e-3, 10, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    model = LogisticRegression(C=C, solver=solver, max_iter=1000, random_state=42)
    score = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring='f1_weighted').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30, show_progress_bar=True)

Run 30–50 trials per model.
Plot optimization history and parameter importances using optuna.visualization.
Retrain best model on full train set with tuned parameters.
Fallback: If compute is limited, use RandomizedSearchCV with n_iter=20 and cv=5 instead. Include both paths with a USE_OPTUNA = True flag.


Cell 9 — Transformer-Based Model (BERT)
9.1 Setup
Use distilbert-base-uncased from HuggingFace Transformers (lighter than full BERT, ~97% of BERT performance):
pythonfrom transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
9.2 Dataset Class
Build a FakeNewsDataset(torch.utils.data.Dataset) class that:

Tokenizes text using DistilBertTokenizerFast with max_length=256, truncation=True, padding='max_length'.
Returns input_ids, attention_mask, labels as tensors.

9.3 Training
pythontraining_args = TrainingArguments(
    output_dir='./distilbert_fakenews',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    warmup_steps=200,
    weight_decay=0.01,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_steps=50,
    fp16=torch.cuda.is_available(),  # auto mixed precision if GPU available
)
Define a compute_metrics function returning accuracy, F1, precision, recall.
9.4 GPU/CPU Fallback
pythonDEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on: {DEVICE}")
If no GPU: reduce num_train_epochs=1, max_length=128, add markdown note explaining compute trade-off.
9.5 Evaluate
Run trainer.evaluate() on validation set. Store metrics in results dict.

Cell 10 — Final Evaluation on Test Set
After selecting the best model(s) based on validation performance:

Evaluate all models on the held-out test set (first time touching it).
For each model report:

Accuracy, Precision (macro/weighted), Recall (macro/weighted), F1 (macro/weighted), ROC-AUC.
Confusion matrix (plot as heatmap using seaborn).
Classification report (full sklearn.metrics.classification_report).


Plot ROC curves for all models on one figure (one curve per model, labeled).
Plot Precision-Recall curves for all models.


Cell 11 — Comprehensive Model Comparison Table
Build a pandas DataFrame comparing ALL models across:
| Model | Val Accuracy | Val F1 (weighted) | Test Accuracy | Test F1 (weighted) | ROC-AUC | Training Time |
Sort by Test F1 descending. Style with df.style.background_gradient(cmap='RdYlGn') for visual ranking.
Print a clear markdown winner declaration cell explaining which model won and why (metrics + robustness).

Cell 12 — Model Interpretability
12.1 Logistic Regression Coefficients

Extract top 30 positive (REAL) and top 30 negative (FAKE) coefficients.
Plot two horizontal bar charts side-by-side.

12.2 LightGBM Feature Importance

Plot plot_importance(lgbm_model, importance_type='gain', max_num_features=30).

12.3 SHAP Values (for LightGBM or LR)
pythonimport shap
explainer = shap.LinearExplainer(lr_model, X_test_tfidf, feature_perturbation='interventional')
shap_values = explainer.shap_values(X_test_tfidf[:200])
shap.summary_plot(shap_values, X_test_tfidf[:200], feature_names=tfidf.get_feature_names_out(), max_display=30)
Also show a SHAP force plot for one REAL and one FAKE prediction.
12.4 Error Analysis

Identify misclassified examples (false positives + false negatives from best model).
Display 5 examples of each with true label, predicted label, and prediction probability.
Comment on patterns in errors.


Cell 13 — Final Model Selection & Justification
Write a structured markdown section covering:

Performance Summary: Which model achieved best F1, AUC, and generalization gap (val vs test).
Efficiency Trade-off: Training time vs performance gain chart (scatter plot: x=train_time, y=test_f1, annotated by model name).
Production Recommendation: Justify choice between TF-IDF+LR (fast, interpretable) vs DistilBERT (highest accuracy) based on deployment constraints.
Limitations: Note dataset-specific biases, domain generalization concerns, temporal drift.
