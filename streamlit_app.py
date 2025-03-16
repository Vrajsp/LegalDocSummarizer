import streamlit as st
from transformers import pipeline
import pdfplumber
import tempfile

# Page configuration
st.set_page_config(page_title="LegalDocSummarizer", layout="wide", page_icon="📄")

# Custom CSS
st.markdown("""
    <style>
        html, body, [class*="css"]  {
            font-family: 'Segoe UI', sans-serif;
        }
        .main {
            background-color: #f7f9fc;
        }
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.5em;
            border: none;
            font-size: 1rem;
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #357ab8;
        }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 style='text-align: center; color: #333;'>📄 Legal Document Summarizer</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Upload your legal document (PDF) and get a quick summary powered by AI.</p>", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader("Choose a legal document (PDF)", type="pdf")

# Pipeline
summarizer = pipeline("summarization")

# Process PDF
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    with pdfplumber.open(file_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    if full_text.strip() == "":
        st.error("No text found in the PDF.")
    else:
        st.subheader("📝 Extracted Text Preview")
        st.text_area("Text", full_text[:3000], height=200)

        if st.button("Summarize"):
            with st.spinner("Summarizing..."):
                summary = summarizer(full_text, max_length=512, min_length=100, do_sample=False)[0]["summary_text"]
            st.success("Done!")
            st.subheader("🧠 Summary")
            st.write(summary)

