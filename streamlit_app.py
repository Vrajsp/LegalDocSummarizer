import streamlit as st
import spacy
from transformers import pipeline
import pdfplumber
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from PIL import Image
import matplotlib.pyplot as plt
from nrclex import NRCLex

# 📌 Initialize session state safely
if "summary_method" not in st.session_state:
    st.session_state["summary_method"] = "Extractive"  # or "Abstractive" if you prefer

# 📌 Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("spaCy model not found. Please make sure 'en_core_web_sm' is installed.")
    st.stop()

# 📌 Load transformers model
summarizer = pipeline("summarization")

# 📌 App title
st.title("🧠 Legal Document Summarizer")

# 📌 Upload section
uploaded_file = st.file_uploader("Upload a legal document (PDF)", type=["pdf"])

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"

    st.subheader("📄 Extracted Text")
    st.text_area("Text from PDF:", text, height=200)

    # 📌 Summary type selector
    st.radio("Choose Summary Type:", ["Extractive", "Abstractive"], key="summary_method")

    # 📌 Generate summary
    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            if st.session_state["summary_method"] == "Extractive":
                kw_model = KeyBERT()
                keywords = kw_model.extract_keywords(text, top_n=5)
                st.subheader("📌 Key Phrases (Extractive)")
                for kw in keywords:
                    st.write(f"- {kw[0]}")
            else:
                summary = summarizer(text[:1024])[0]['summary_text']
                st.subheader("📝 Summary (Abstractive)")
                st.write(summary)

    # 📌 Sentiment analysis
    if st.button("Analyze Emotion"):
        with st.spinner("Analyzing..."):
            doc = NRCLex(text)
            emotions = doc.raw_emotion_scores
            if emotions:
                st.subheader("❤️ Emotion Distribution")
                fig, ax = plt.subplots()
                ax.bar(emotions.keys(), emotions.values(), color="skyblue")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.write("No clear emotional content detected.")
