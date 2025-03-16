import streamlit as st
import pdfplumber
import fitz  # PyMuPDF
from transformers import pipeline
from keybert import KeyBERT
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
import pandas as pd
import torch

# Downloads
nltk.download('punkt')
nltk.download('stopwords')

# Models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
kw_model = KeyBERT()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Sidebar
st.sidebar.title("⚖️ Legal Doc Summarizer")
feature = st.sidebar.radio("Choose a feature", ['Upload PDF', 'Summarize Text', 'Keyword Extraction', 'Keyword Chart'])

# File uploader
def read_pdf_pdfplumber(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = ''.join(page.extract_text() or '' for page in pdf.pages)
    return text

def read_pdf_pymupdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

# Summarization
def generate_summary(text):
    if len(text.split()) < 50:
        return "Text too short for summarization."
    summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Keyword Extraction
def extract_keywords(text, num_keywords=10):
    keywords = kw_model.extract_keywords(text, top_n=num_keywords)
    return keywords

# Plotting
def plot_keywords(keywords):
    words, scores = zip(*keywords)
    plt.figure(figsize=(10, 5))
    plt.bar(words, scores, color='skyblue')
    plt.title('Top Keywords')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# Main Features
if feature == 'Upload PDF':
    st.title("📄 Upload and View PDF")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_file:
        method = st.radio("Choose text extraction method", ["pdfplumber", "PyMuPDF"])
        text = read_pdf_pdfplumber(uploaded_file) if method == "pdfplumber" else read_pdf_pymupdf(uploaded_file)
        st.subheader("Extracted Text:")
        st.text_area("Text", text, height=300)

elif feature == 'Summarize Text':
    st.title("📝 Text Summarizer")
    input_text = st.text_area("Paste your legal text here")
    if st.button("Summarize"):
        with st.spinner("Summarizing..."):
            summary = generate_summary(input_text)
            st.success("Summary generated!")
            st.subheader("Summary:")
            st.write(summary)

elif feature == 'Keyword Extraction':
    st.title("🔍 Keyword Extraction")
    input_text = st.text_area("Paste text for keyword extraction")
    top_n = st.slider("Number of keywords", 5, 20, 10)
    if st.button("Extract Keywords"):
        with st.spinner("Extracting..."):
            keywords = extract_keywords(input_text, top_n)
            st.subheader("Keywords:")
            for word, score in keywords:
                st.write(f"• **{word}** (Score: {score:.2f})")

elif feature == 'Keyword Chart':
    st.title("📊 Keyword Frequency Chart")
    input_text = st.text_area("Paste text for frequency analysis")
    top_n = st.slider("Top N Keywords", 5, 20, 10)
    if st.button("Show Chart"):
        keywords = extract_keywords(input_text, top_n)
        plot_keywords(keywords)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ using `Streamlit`, `Transformers`, `KeyBERT`, and more.")
