import streamlit as st
import joblib
import re
import nltk
import os

# ---------------------------
# FIX NLTK DATA (IMPORTANT)
# ---------------------------
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")

if not os.path.exists(nltk_data_path):
    os.makedirs(nltk_data_path)

nltk.data.path.append(nltk_data_path)

# Download required datasets
nltk.download('stopwords', download_dir=nltk_data_path)
nltk.download('wordnet', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------------------
# LOAD MODEL + VECTORIZER
# ---------------------------
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------------------
# NLP TOOLS
# ---------------------------
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# PREPROCESS FUNCTION
# ---------------------------
def preprocess(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 2
    ]
    return " ".join(tokens)

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("📧 Spam Email Classifier")

st.write("Enter an email message below to check whether it is Spam or Ham.")

user_input = st.text_area("✉️ Email Content")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        cleaned = preprocess(user_input)

        st.subheader("🔍 Preprocessed Text")
        st.write(cleaned)

        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probability = model.predict_proba(vector)[0]

        confidence = max(probability)

        st.subheader("📊 Result")

        if prediction == 1:
            st.error(f"🚨 Spam (Confidence: {confidence:.2f})")
        else:
            st.success(f"✅ Ham (Confidence: {confidence:.2f})")