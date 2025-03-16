import streamlit as st
from transformers import pipeline
import pdfplumber
import torch
import nltk
from nltk.corpus import stopwords
from keybert import KeyBERT
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from sentence_transformers import SentenceTransformer

nltk.download('stopwords')

# Initialize models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
kw_model = KeyBERT(model=SentenceTransformer('all-MiniLM-L6-v2'))

# Streamlit UI
st.title("🧠 Legal Document Summarizer & Keyword Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    st.subheader("📄 Original Text")
    st.write(text[:2000] + ("..." if len(text) > 2000 else ""))

    if st.button("🔍 Summarize & Analyze"):
        # Summarize
        summary = summarizer(text[:1024], max_length=150, min_length=30, do_sample=False)[0]['summary_text']

        st.subheader("📝 Summary")
        st.write(summary)

        # Keywords
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=10)

        st.subheader("🏷️ Top Keywords")
        for kw, score in keywords:
            st.markdown(f"- **{kw}** ({score:.2f})")

        # Matplotlib chart
        st.subheader("📊 Keyword Scores")
        words = [kw[0] for kw in keywords]
        scores = [kw[1] for kw in keywords]
        fig, ax = plt.subplots()
        ax.barh(words[::-1], scores[::-1])
        st.pyplot(fig)

        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, "Summary:\n" + summary + "\n\nKeywords:\n" + "\n".join(f"{kw[0]} ({kw[1]:.2f})" for kw in keywords))
        output_path = "summary_output.pdf"
        pdf.output(output_path)

        with open(output_path, "rb") as f:
            st.download_button("📥 Download Summary PDF", f, file_name="summary_output.pdf", mime="application/pdf")

