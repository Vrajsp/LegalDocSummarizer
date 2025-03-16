# app.py
import streamlit as st
import fitz  # PyMuPDF
import base64
import io
import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer
from keybert import KeyBERT

# Load Summarizer & Models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
model_ckpt = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
redflag_model = AutoModelForSequenceClassification.from_pretrained(model_ckpt)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_model = KeyBERT(model=embed_model)

red_flags = [
    "non-compete clause", "low notice period", "salary delay", "no paid leave",
    "discrimination", "mandatory arbitration", "harassment", "no severance",
    "unfair termination", "no health benefits"
]

def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    return "\n".join([page.get_text() for page in doc])

def summarize_text(text):
    chunks = [text[i:i+800] for i in range(0, len(text), 800)]
    summarized = [summarizer(chunk, max_length=80, min_length=25, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summarized)

def generate_bullet_points(text):
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
    return [kw[0] for kw in keywords]

def detect_red_flags(text):
    inputs = tokenizer([text]*len(red_flags), padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = redflag_model(**inputs).logits
        probs = torch.sigmoid(outputs).squeeze().numpy()
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(probs.reshape(-1,1)).flatten()
    return list(zip(red_flags, scores))

def generate_red_flag_chart(scores):
    labels = [item[0] for item in scores]
    values = [item[1] for item in scores]
    fig, ax = plt.subplots()
    ax.barh(labels, values, color="crimson")
    ax.set_title("Detected Red Flags")
    st.pyplot(fig)

def get_table_download_link(df, filename="report.csv", filetype="csv"):
    if filetype == "csv":
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV Report</a>'
    return href

# Streamlit App
st.set_page_config(page_title="Smart Legal Analyzer", layout="centered")
st.title("📄 Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload a legal PDF", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("PDF uploaded and parsed successfully!")

    with st.spinner("Summarizing document..."):
        summary = summarize_text(text)
        bullets = generate_bullet_points(summary)

    with st.spinner("Analyzing for red flags..."):
        redflag_scores = detect_red_flags(text)
        df = pd.DataFrame(redflag_scores, columns=["Red Flag", "Score"])

    # Show Summary
    with st.expander("📑 Document Summary", expanded=True):
        st.markdown(f"**Paragraph Summary:**\n\n{summary}")
        st.markdown("**🔹 Bullet Points:**")
        for b in bullets:
            st.markdown(f"- {b}")

    # Show Red Flags
    with st.expander("⚠️ Red Flag Analysis", expanded=True):
        st.dataframe(df.style.background_gradient(cmap='Reds'))
        generate_red_flag_chart(redflag_scores)

    # Export Options
    st.markdown(get_table_download_link(df), unsafe_allow_html=True)
    st.download_button("📥 Download Summary as TXT", data=summary, file_name="summary.txt")
