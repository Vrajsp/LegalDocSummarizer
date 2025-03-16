import streamlit as st
import fitz  # PyMuPDF
import nltk
import io
import pandas as pd
import matplotlib.pyplot as plt
import base64
from nltk.tokenize import sent_tokenize
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

# Download punkt for NLTK
nltk.download('punkt')

# Load summarizer
summarizer = pipeline("summarization")

# Define red flags
red_flags = [
    "non-compete clause", "low notice period", "salary delay", "no paid leave",
    "discrimination", "mandatory arbitration", "harassment", "no severance",
    "unfair termination", "no health benefits"
]

# Simple keyword-based red flag detector using TF-IDF
vectorizer = TfidfVectorizer().fit(red_flags)


def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def summarize_text(text):
    if len(text.split()) < 50:
        return text
    summary = summarizer(text[:1000], max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']


def detect_red_flags(text):
    sentences = sent_tokenize(text)
    scores = []
    for flag in red_flags:
        vecs = vectorizer.transform([flag] + sentences)
        sims = cosine_similarity(vecs[0:1], vecs[1:]).flatten()
        scores.append((flag, max(sims)))
    return sorted(scores, key=lambda x: x[1], reverse=True)


def download_link(object_to_download, download_filename, download_link_text):
    if isinstance(object_to_download, pd.DataFrame):
        towrite = io.BytesIO()
        object_to_download.to_csv(towrite, index=False)
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
    else:
        b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'


# Streamlit UI
st.set_page_config(page_title="Legal Doc Analyzer", layout="wide")
st.title("📄 Legal Document Analyzer Dashboard")

uploaded_file = st.file_uploader("Upload a legal PDF", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)

    with st.spinner("Generating summary..."):
        summary = summarize_text(text)

    st.subheader("📌 Document Summary")
    st.markdown(f"**Paragraph Summary:**\n{text if len(text.split()) < 50 else summary}")

    st.markdown("**\n\n🔹 Bullet Points:**")
    bullets = sent_tokenize(summary)
    for b in bullets:
        st.markdown(f"- {b}")

    with st.spinner("Analyzing red flags..."):
        red_flag_scores = detect_red_flags(text)
        df_flags = pd.DataFrame(red_flag_scores, columns=["Red Flag", "Score"])

    st.subheader("⚠️ Red Flag Analysis")
    st.dataframe(df_flags)

    st.subheader("📊 Detected Red Flags")
    fig, ax = plt.subplots()
    df_flags_sorted = df_flags.sort_values("Score", ascending=True)
    ax.barh(df_flags_sorted["Red Flag"], df_flags_sorted["Score"], color='crimson')
    ax.set_xlabel("Risk Score")
    ax.set_xlim([0, 1])
    st.pyplot(fig)

    st.subheader("📥 Export Results")
    csv_link = download_link(df_flags, "red_flags.csv", "Download Red Flags CSV")
    st.markdown(csv_link, unsafe_allow_html=True)

    summary_link = download_link(summary, "summary.txt", "Download Summary")
    st.markdown(summary_link, unsafe_allow_html=True)

    st.success("✅ Dashboard generated successfully!")
