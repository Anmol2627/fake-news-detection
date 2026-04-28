import streamlit as st
import joblib
import numpy as np
import plotly.graph_objects as go
import time
from scipy import sparse
import os

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ── Simple Preprocessing (must match training) ───────────────
import re
RE_SPACE = re.compile(r"\s+")

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\(reuters\)", " ", text)
    text = re.sub(r"\breuters\b", " ", text)
    text = RE_SPACE.sub(" ", text)
    return text.strip()


# ── Load Model + Vectorizers ─────────────────────────────────
@st.cache_resource
def load_components():
    try:
        model = joblib.load("model.pkl")
        tfidf_word = joblib.load("tfidf_word.pkl")
        tfidf_char = joblib.load("tfidf_char.pkl")
        return model, tfidf_word, tfidf_char
    except:
        return None, None, None


# ── Prediction ───────────────────────────────────────────────
def predict(model, tfidf_word, tfidf_char, text):
    text_clean = preprocess_text(text)

    X_word = tfidf_word.transform([text_clean])
    X_char = tfidf_char.transform([text_clean])

    X = sparse.hstack([X_word, X_char])

    proba = model.predict_proba(X)[0]

    prob_fake = float(proba[0])
    prob_real = float(proba[1])

    label = "FAKE" if prob_fake >= prob_real else "REAL"
    confidence = max(prob_fake, prob_real)

    return label, confidence, prob_fake, prob_real


# ── Visualization ───────────────────────────────────────────
def render_gauge(conf):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=conf * 100,
        title={"text": "Confidence (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#00c853"},
        }
    ))
    st.plotly_chart(fig, use_container_width=True)


def render_bar(fake, real):
    fig = go.Figure(data=[
        go.Bar(name="FAKE", x=["FAKE"], y=[fake]),
        go.Bar(name="REAL", x=["REAL"], y=[real])
    ])
    fig.update_layout(barmode="group")
    st.plotly_chart(fig, use_container_width=True)


# ── Main App ────────────────────────────────────────────────
def main():
    st.title("📰 Fake News Detection System")
    st.caption("⚠️ Detects writing patterns, not factual truth")

    model, tfidf_word, tfidf_char = load_components()

    if model is None:
        st.error("❌ Missing model files. Ensure model.pkl, tfidf_word.pkl, tfidf_char.pkl exist.")
        return

    text = st.text_area(
        "Enter News Article",
        height=200,
        placeholder="Paste news article here..."
    )

    st.caption(f"{len(text)} characters")

    if st.button("Analyze"):
        if not text.strip():
            st.warning("Please enter text")
            return

        with st.spinner("Analyzing..."):
            time.sleep(0.4)
            label, conf, pf, pr = predict(model, tfidf_word, tfidf_char, text)

        # Result
        if label == "REAL":
            st.success(f"Prediction: REAL")
        else:
            st.error(f"Prediction: FAKE")

        st.write(f"Confidence: {conf:.4f}")

        # Charts
        render_gauge(conf)
        render_bar(pf, pr)

        st.info("This model detects linguistic patterns, not factual truth.")


# ── Run App ─────────────────────────────────────────────────
if __name__ == "__main__":
    main()