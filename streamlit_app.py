import streamlit as st
import fitz  # PyMuPDF
import pandas as pd
import matplotlib.pyplot as plt
import base64
from transformers import pipeline
import nltk
from io import BytesIO
nltk.download("punkt")
from nltk.tokenize import sent_tokenize

st.set_page_config(layout="wide")
st.title("📄 Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload a Legal Document (PDF)", type=["pdf"])

if uploaded_file:
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = "\n".join(page.get_text() for page in doc)

    st.subheader("📃 Extracted Text Preview")
    with st.expander("Click to expand"):
        st.write(text)

    # Summarization
    summarizer = pipeline("summarization")
    sentences = sent_tokenize(text)
    chunks = [" ".join(sentences[i:i+5]) for i in range(0, len(sentences), 5)]
    summary = ""
    for chunk in chunks:
        summary_piece = summarizer(chunk, max_length=100, min_length=30, do_sample=False)[0]["summary_text"]
        summary += summary_piece + "\n"

    st.subheader("📝 Document Summary")
    st.markdown(f"**Summary:**\n\n{summary}")

    bullet_points = "\n".join([f"- {sent}" for sent in sent_tokenize(summary)])
    st.markdown("**\nBullet Points:**")
    st.markdown(bullet_points)

    # Red Flag Detection
    red_flags = [
        "non-compete clause",
        "low notice period",
        "salary delay",
        "no paid leave",
        "discrimination",
        "mandatory arbitration",
        "harassment",
        "no severance",
        "unfair termination",
        "no health benefits"
    ]

    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    results = classifier(text, red_flags)

    df = pd.DataFrame({"Red Flag": results["labels"], "Score": results["scores"]})
    st.subheader("⚠️ Red Flag Analysis")
    st.dataframe(df)

    fig, ax = plt.subplots()
    df_sorted = df.sort_values("Score")
    ax.barh(df_sorted["Red Flag"], df_sorted["Score"], color="crimson")
    ax.set_xlabel("Confidence Score")
    ax.set_title("Detected Red Flags")
    st.pyplot(fig)

    # Export Results
    export_df = df.copy()
    export_df["Summary"] = summary
    buffer = BytesIO()
    export_df.to_csv(buffer, index=False)
    buffer.seek(0)
    b64 = base64.b64encode(buffer.read()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="red_flag_analysis.csv">📥 Download Red Flag Report as CSV</a>'
    st.markdown(href, unsafe_allow_html=True)
