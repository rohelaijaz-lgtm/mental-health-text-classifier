
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import re
import pickle

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Mental Health Text Classifier", page_icon="🧠")

st.title("🧠 Mental Health Text Classifier")
st.write("Type a statement below and the model will classify it into a mental health category.")

# ── Load model and vectoriser ────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectoriser.pkl", "rb") as f:
        vectoriser = pickle.load(f)
    return model, vectoriser

model, vectoriser = load_model()

# ── Classify function ────────────────────────────────────────
def classify(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.strip()
    vec = vectoriser.transform([text])
    return model.predict(vec)[0]

# ── UI ───────────────────────────────────────────────────────
user_input = st.text_area("Enter your statement here:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a statement first.")
    else:
        result = classify(user_input)
        st.success(f"Predicted category: **{result}**")
        st.info("⚠️ This is not a clinical diagnosis. If you are struggling, please speak to a mental health professional.")
