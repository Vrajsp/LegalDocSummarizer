import streamlit as st
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import pdfplumber

# -------- Extract Text from PDF -------- #
def extract_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# -------- Load Models -------- #
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
summarizer = pipeline("summarization", model="t5-small")

# -------- Red Flag Labels -------- #
red_flags = [
    "unfair termination",
    "non-compete clause",
    "salary delay",
    "discrimination",
    "harassment",
    "no severance",
    "low notice period",
    "mandatory arbitration",
    "no health benefits",
    "no paid leave"
]

# -------- Red Flag Detection -------- #
def detect_red_flags(text):
    results = classifier(text, red_flags, multi_label=True)
    return dict(zip(results['labels'], results['scores']))

# -------- Generate PDF Report -------- #
def generate_pdf(data):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    text_obj = c.beginText(50, 750)
    text_obj.setFont("Helvetica", 12)
    text_obj.textLine("Red Flag Detection Report")
    text_obj.textLine("")
    for index, row in data.iterrows():
        text_obj.textLine(f"{row['Red Flag']}: {round(row['Score'], 3)}")
    c.drawText(text_obj)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# -------- Streamlit UI -------- #
st.set_page_config(page_title="Legal AI Dashboard", layout="centered")
st.title("📄 Legal Document AI Dashboard")
st.markdown("Upload a legal PDF to get an automatic summary and red flag detection report.")

uploaded_file = st.file_uploader("📁 Upload PDF Document", type="pdf")

if uploaded_file:
    with st.spinner("Processing document..."):
        text = extract_text(uploaded_file)

    if text:
        with st.spinner("Summarizing..."):
            summary = summarizer(text[:1000])[0]['summary_text']
            bullet_summary = "\n".join([f"- {line.strip().capitalize()}" for line in summary.split('.') if line.strip()])

        with st.spinner("Detecting red flags..."):
            flag_scores = detect_red_flags(text)
            df_flags = pd.DataFrame(flag_scores.items(), columns=["Red Flag", "Score"]).sort_values("Score", ascending=False)

        st.success("✅ Done! Here's your AI-generated dashboard:")

        # --- Summary Section --- #
        st.subheader("📄 Document Summary")
        st.markdown("**📝 Paragraph Summary:**")
        st.write(summary)

        st.markdown("**📌 Bullet Point View:**")
        st.markdown(bullet_summary)

        # --- Red Flag Table --- #
        st.subheader("⚠️ Red Flag Analysis")
        st.dataframe(df_flags)

        # --- Bar Chart --- #
        fig, ax = plt.subplots()
        ax.barh(df_flags["Red Flag"], df_flags["Score"], color='crimson')
        ax.set_xlabel("Confidence Score")
        ax.set_title("Detected Red Flags")
        st.pyplot(fig)

        # --- Download CSV --- #
        csv = df_flags.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV Report", data=csv, file_name="red_flag_report.csv", mime="text/csv")

        # --- Download PDF --- #
        pdf_file = generate_pdf(df_flags)
        st.download_button("📄 Download PDF Report", data=pdf_file, file_name="red_flag_report.pdf", mime="application/pdf")

    else:
        st.error("❌ No readable text found in the uploaded PDF.")
