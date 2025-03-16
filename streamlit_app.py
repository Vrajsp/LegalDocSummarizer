import streamlit as st
import pdfplumber
import pandas as pd
import matplotlib.pyplot as plt
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import textwrap


st.set_page_config(page_title="Legal Document Analyzer", layout="wide")

st.title("📄 Legal Document Analyzer")

uploaded_file = st.file_uploader("Upload a legal PDF document", type="pdf")

# Predefined red flag phrases
red_flags = [
    "non-compete clause", "low notice period", "salary delay", "no paid leave", "discrimination",
    "mandatory arbitration", "harassment", "no severance", "unfair termination", "no health benefits"
]

model = SentenceTransformer('all-MiniLM-L6-v2')
summarizer = pipeline("summarization")

if uploaded_file:
    with pdfplumber.open(uploaded_file) as pdf:
        text = " ".join(page.extract_text() or '' for page in pdf.pages)

    st.subheader("📌 Document Summary")
    summary = summarizer(text, max_length=120, min_length=30, do_sample=False)[0]['summary_text']
    st.markdown(f"**Paragraph Summary:**\n{textwrap.fill(summary, width=120)}")

    bullet_pipe = pipeline("summarization")
    bullet_points = bullet_pipe(text, max_length=60, min_length=15, do_sample=False, truncation=True)[0]['summary_text']
    st.markdown("**\n\n🔹 Bullet Points:**")
    for point in bullet_points.split('.'):
        if point.strip():
            st.markdown(f"- {point.strip()}")

    st.subheader("⚠️ Red Flag Analysis")
    text_emb = model.encode(text, convert_to_tensor=True)
    scores = []

    for phrase in red_flags:
        phrase_emb = model.encode(phrase, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(text_emb, phrase_emb)
        scores.append(round(float(sim.max()), 4))

    df = pd.DataFrame({"Red Flag": red_flags, "Score": scores})
    df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
    st.dataframe(df.style.background_gradient(cmap='Reds', subset="Score"), use_container_width=True)

    st.subheader("📊 Detected Red Flags")
    fig, ax = plt.subplots()
    red_df = df[df['Score'] > 0.01]
    ax.barh(red_df['Red Flag'], red_df['Score'], color="crimson")
    ax.invert_yaxis()
    ax.set_xlabel("Score")
    st.pyplot(fig)

    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Red Flag Report (CSV)", csv, "red_flags.csv", "text/csv")
