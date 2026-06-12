import streamlit as st
import pickle
import re
import requests

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

# ── Clean text ───────────────────────────────────────────────
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text.strip()

# ── Classify ─────────────────────────────────────────────────
def classify(text):
    cleaned = clean_text(text)
    vec = vectoriser.transform([cleaned])
    return model.predict(vec)[0]

# ── LLM Response via OpenRouter ──────────────────────────────
def get_llm_response(category, user_text):
    api_key = st.secrets["OPENROUTER_API_KEY"]
    
    prompt = f"""A person has written the following:
"{user_text}"

This has been classified as: {category}

Please respond with a brief, compassionate, and supportive message acknowledging what they may be feeling. 
Encourage them to seek professional help if appropriate. Keep it to 3-4 sentences. Do not diagnose."""

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "nvidia/nemotron-3-ultra-550b-a55b:free",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    
    result = response.json()
    return result["choices"][0]["message"]["content"]

# ── UI ───────────────────────────────────────────────────────
user_input = st.text_area("Enter your statement here:", height=150)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter a statement first.")
    else:
        category = classify(user_input)
        st.success(f"Predicted category: **{category}**")
        
        with st.spinner("Generating response..."):
            llm_response = get_llm_response(category, user_input)
            st.info(llm_response)
        
        st.caption("⚠️ This is not a clinical diagnosis. If you are struggling, please speak to a mental health professional.")
