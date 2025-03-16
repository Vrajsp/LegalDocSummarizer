# streamlit_app.py
import streamlit as st
import pdfplumber
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
import time

# App config
st.set_page_config(
    page_title="LegalDocSummarizer",
    page_icon="⚖️",
    layout="centered"
)

# Initialize models
@st.cache_resource
def load_summarizer():
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    return model, tokenizer

@st.cache_resource
def load_redflag_detector():
    classifier = pipeline("zero-shot-classification",
                        model="facebook/bart-large-mnli")
    return classifier

# PDF processing
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        text = " ".join([page.extract_text() for page in pdf.pages])
    return text

# Text processing
def chunk_text(text, max_length=512):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def summarize_text(model, tokenizer, text):
    chunks = chunk_text(text)
    summaries = []
    for chunk in chunks:
        inputs = tokenizer.encode("summarize: " + chunk, 
                                return_tensors="pt",
                                max_length=512,
                                truncation=True)
        outputs = model.generate(inputs,
                                max_length=150,
                                min_length=40,
                                length_penalty=2.0,
                                num_beams=4,
                                early_stopping=True)
        summaries.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return " ".join(summaries)

# Red flag detection
def detect_redflags(classifier, text):
    candidate_labels = [
        "unfair termination", "non-compete clause", "salary delay",
        "discrimination", "harassment", "no severance",
        "low notice period", "mandatory arbitration",
        "no health benefits", "no paid leave"
    ]
    
    results = classifier(text, candidate_labels, multi_label=True)
    return pd.DataFrame({
        "Red Flag": results["labels"],
        "Confidence": results["scores"]
    }).sort_values("Confidence", ascending=False)

# Report generation
def generate_csv_report(df):
    return df.to_csv(index=False).encode()

def generate_pdf_report(summary, df, chart_path):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    
    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(72, 750, "Legal Document Analysis Report")
    
    # Summary
    c.setFont("Helvetica", 12)
    c.drawString(72, 700, "Document Summary:")
    text = c.beginText(72, 680)
    text.setFont("Helvetica", 10)
    for line in summary.split("\n"):
        text.textLine(line)
    c.drawText(text)
    
    # Red Flags
    c.drawString(72, 600, "Identified Risks:")
    c.drawImage(chart_path, 72, 400, width=400, height=200)
    
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer

# Main app
def main():
    st.title("⚖️ LegalDocSummarizer")
    st.markdown("Automated legal document analysis with AI-powered risk detection")
    
    uploaded_file = st.file_uploader("Upload legal document (PDF)", type="pdf")
    
    if uploaded_file:
        with st.spinner("Analyzing document..."):
            # Extract text
            text = extract_text_from_pdf(uploaded_file)
            
            # Load models
            sum_model, sum_tokenizer = load_summarizer()
            classifier = load_redflag_detector()
            
            # Generate summary
            summary = summarize_text(sum_model, sum_tokenizer, text)
            
            # Show summaries
            with st.expander("📝 Paragraph Summary"):
                st.write(summary)
                
            with st.expander("📌 Bullet Point Summary"):
                bullet_summary = summary.replace(". ", ".\n\n• ").replace(" .", ".")
                st.markdown(f"• {bullet_summary}")
            
            # Red flag detection
            st.subheader("🔍 Risk Analysis")
            redflags_df = detect_redflags(classifier, text[:2000])  # Use first 2000 chars for speed
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(redflags_df.style.format({"Confidence": "{:.2%}"}))
            
            with col2:
                fig, ax = plt.subplots()
                redflags_df.sort_values("Confidence").plot.barh(
                    x="Red Flag", y="Confidence", ax=ax)
                plt.title("Risk Confidence Scores")
                plt.xlabel("Confidence")
                plt.tight_layout()
                st.pyplot(fig)
            
            # Report generation
            st.subheader("📤 Generate Reports")
            
            csv_report = generate_csv_report(redflags_df)
            st.download_button(
                label="Download CSV Report",
                data=csv_report,
                file_name="legal_analysis.csv",
                mime="text/csv"
            )
            
            if st.button("Download PDF Report"):
                with st.spinner("Generating PDF..."):
                    chart_path = "temp_chart.png"
                    fig.savefig(chart_path, bbox_inches="tight")
                    pdf_report = generate_pdf_report(summary, redflags_df, chart_path)
                    
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_report,
                        file_name="legal_analysis.pdf",
                        mime="application/pdf"
                    )

if __name__ == "__main__":
    main()
