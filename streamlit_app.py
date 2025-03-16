import streamlit as st
import pdfplumber
from transformers import pipeline
from nrclex import NRCLex
import spacy
import matplotlib.pyplot as plt

# Load summarizer pipeline
summarizer = pipeline("summarization")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Page setup
st.set_page_config(page_title="LegalDocSummarizer", layout="wide")
st.title("📄 Legal Document Summarizer")
st.markdown("Upload a legal PDF and get a quick AI-generated summary, emotion analysis, and argument breakdown.")

# File uploader
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    st.subheader("📃 Extracted Text")
    with st.expander("View full document text"):
        st.text_area("Text", full_text, height=300)

    # Summarization
    st.subheader("✂️ Summary")
    summary_chunks = []
    for i in range(0, len(full_text), 1000):
        chunk = full_text[i:i+1000]
        if len(chunk.strip()) > 50:
            result = summarizer(chunk, max_length=130, min_length=30, do_sample=False)
            summary_chunks.append(result[0]["summary_text"])

    summary = " ".join(summary_chunks)
    st.success(summary)

    # Emotion Detection
    st.subheader("💬 Emotion Detection")
    emotion = NRCLex(full_text)
    top_emotions = emotion.top_emotions

    if top_emotions:
        labels, scores = zip(*top_emotions)
        fig, ax = plt.subplots()
        ax.bar(labels, scores, color="orange")
        st.pyplot(fig)
    else:
        st.info("No strong emotional tones detected.")

    # Argument Mapping
    st.subheader("🧠 Argument Structure")
    doc = nlp(full_text[:1500])  # limit for performance

    claims, evidences, conclusions = [], [], []
    for sent in doc.sents:
        lower = sent.text.lower()
        if "we argue" in lower or "this suggests" in lower:
            claims.append(sent.text)
        elif "because" in lower or "due to" in lower:
            evidences.append(sent.text)
        elif "therefore" in lower or "hence" in lower or "thus" in lower:
            conclusions.append(sent.text)

    with st.expander("📌 Claims"):
        for i, c in enumerate(claims): st.markdown(f"**{i+1}.** {c}")
    with st.expander("🔍 Evidences"):
        for i, e in enumerate(evidences): st.markdown(f"**{i+1}.** {e}")
    with st.expander("✅ Conclusions"):
        for i, con in enumerate(conclusions): st.markdown(f"**{i+1}.** {con}")
