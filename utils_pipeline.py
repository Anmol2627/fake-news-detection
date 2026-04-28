import re
import numpy as np
from scipy import sparse

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag

# ── NLTK setup ──
STOP_WORDS_SET = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# ── Regex ──
_RE_URL = re.compile(r'https?://\S+|www\.\S+')
_RE_HTML = re.compile(r'<[^>]+>')
_RE_PUNCT = re.compile(r'[^\w\s]')
_RE_DIGIT = re.compile(r'\d+')
_RE_WS = re.compile(r'\s+')


# ── POS helper ──
def _get_wordnet_pos(tag):
    return {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'N': wordnet.NOUN,
        'R': wordnet.ADV
    }.get(tag[0].upper(), wordnet.NOUN)


# ── TEXT PREPROCESSING ──
def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "unknown"

    text = text.lower()
    text = _RE_URL.sub(" ", text)
    text = _RE_HTML.sub(" ", text)
    text = _RE_PUNCT.sub(" ", text)
    text = _RE_DIGIT.sub(" ", text)
    text = _RE_WS.sub(" ", text).strip()

    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in STOP_WORDS_SET]

    pos_tags = pos_tag(tokens)
    tokens = [LEMMATIZER.lemmatize(w, _get_wordnet_pos(t)) for w, t in pos_tags]

    tokens = [t for t in tokens if len(t) >= 2]

    cleaned = " ".join(tokens).strip()
    return cleaned if cleaned else "unknown"


# ── WRAPPER FOR PIPELINE ──
def preprocess_wrapper(texts):
    return [preprocess_text(t) for t in texts]


# ── NUMERIC FEATURES (MUST MATCH TRAINING: 8 FEATURES) ──
def numeric_features_transform(texts):
    feats = []

    for text in texts:
        words = text.split()

        n_words = max(len(words), 1)
        n_chars = max(len(text), 1)
        n_alpha = max(sum(c.isalpha() for c in text), 1)

        avg_wl = sum(len(w) for w in words) / n_words if words else 0.0
        stop_cnt = sum(1 for w in words if w.lower() in STOP_WORDS_SET)
        digits = sum(c.isdigit() for c in text)

        feats.append([
            n_words,
            n_chars,
            avg_wl,
            text.count("!"),                         # ✔ keep
            sum(c.isupper() for c in text) / n_alpha,
            len(set(words)) / n_words,
            stop_cnt / n_words,
            digits / n_chars,
        ])

    return np.array(feats, dtype=np.float32)


# ── FEATURE BUILDER ──
def build_features(texts, tfidf, scaler):
    tfidf_vec = tfidf.transform(texts)

    num = numeric_features_transform(texts)
    num_scaled = scaler.transform(num)
    num_sparse = sparse.csr_matrix(num_scaled)

    return sparse.hstack([tfidf_vec, num_sparse], format="csr")

TFIDF = None
SCALER = None

def set_vectorizer_scaler(tfidf, scaler):
    global TFIDF, SCALER
    TFIDF = tfidf
    SCALER = scaler

def build_features_global(texts):
    tfidf_vec = TFIDF.transform(texts)

    num = numeric_features_transform(texts)
    num_scaled = SCALER.transform(num)
    num_sparse = sparse.csr_matrix(num_scaled)

    return sparse.hstack([tfidf_vec, num_sparse], format="csr")