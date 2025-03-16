# streamlit_app.py
import streamlit as st
import pdfplumber
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BartForSequenceClassification, BartTokenizer
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor
import textwrap

# Configuration
st.set_page_config(
    page_title="LegalDocSummarizer Pro",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Model caching with quantization
@st.cache_resource(show_spinner=False)
def load_models():
    """Load models with optimized settings"""
    device = 0 if torch.cuda.is_available() else -1
    
    # Quantized summarization model
    summarizer = pipeline(
        "summarization",
        model="t5-small",
        tokenizer="t5-small",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )
    
    # Optimized BART model
    redflag_model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )
    
    return summarizer, redflag_model

# Parallel text processing
def chunk_text(text, max_chunk=512):
    """Split text into optimized chunks"""
    return textwrap.wrap(text, width=max_chunk, break_long_words=False)

def parallel_summarize(chunks, summarizer, max_workers=4):
    """Parallel summary generation"""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(summarizer, chunk, 
                                  max_length=150, 
                                  min_length=40,
                                  do_sample=False) 
                  for chunk in chunks]
        return [f.result()[0]['summary_text'] for f in futures]

# Enhanced PDF processing
def extract_text_with_metadata(uploaded_file, max_pages=20):
    """Extract text with smart page limiting"""
    text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for i, page in enumerate(pdf.pages):
            if i >= max_pages:
                break
            text.append(page.extract_text(x_tolerance=1, y_tolerance=1))
    return "\n".join(text), min(max_pages, len(pdf.pages))

# Advanced report generation
def create_pdf_report(summary, df, fig):
    """Generate professional PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("<b>Legal Document Analysis Report</b>", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Summary Section
    story.append(Paragraph("<b>Document Summary:</b>", styles['Heading2']))
    wrapped_summary = textwrap.fill(summary, width=100)
    story.append(Paragraph(wrapped_summary, styles['BodyText']))
    story.append(Spacer(1, 24))
    
    # Risk Analysis
    story.append(Paragraph("<b>Risk Analysis:</b>", styles['Heading2']))
    
    # Save figure
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight')
    img_buffer.seek(0)
    
    # Add image
    story.append(Image(img_buffer, width=400, height=300))
    story.append(Spacer(1, 24))
    
    # Risk Details
    story.append(Paragraph("<b>Detailed Findings:</b>", styles['Heading3']))
    for _, row in df.iterrows():
        text = f"{row['Red Flag']}: {row['Confidence']:.1%} confidence"
        story.append(Paragraph(text, styles['BodyText']))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Main application
def main():
    st.title("⚖️ LegalDocSummarizer Pro")
    st.markdown("AI-powered legal document analysis with enterprise-grade performance")
    
    # Model loading with progress
    with st.spinner("Loading AI Engine (This happens once)..."):
        summarizer, redflag_model = load_models()
    
    uploaded_file = st.file_uploader("Upload Legal Document (PDF)", type="pdf")
    
    if uploaded_file:
        start_time = time.time()
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Text extraction
        status_text.markdown("📄 **Extracting text from document...**")
        raw_text, pages_processed = extract_text_with_metadata(uploaded_file)
        progress_bar.progress(10)
        
        # Text chunking
        status_text.markdown("✂️ **Preparing document for analysis...**")
        chunks = chunk_text(raw_text)
        progress_bar.progress(20)
        
        # Parallel summarization
        status_text.markdown("📝 **Generating summary (Parallel processing)...**")
        summary_chunks = parallel_summarize(chunks, summarizer)
        full_summary = " ".join(summary_chunks)
        progress_bar.progress(60)
        
        # Red flag detection
        status_text.markdown("🚩 **Identifying potential risks...**")
        candidate_labels = [
            "unfair termination", "non-compete clause", "salary delay",
            "discrimination", "harassment", "no severance",
            "low notice period", "mandatory arbitration",
            "no health benefits", "no paid leave"
        ]
        results = redflag_model(
            raw_text[:3000],  # Optimized input length
            candidate_labels,
            multi_label=True,
            truncation=True
        )
        redflags_df = pd.DataFrame({
            "Red Flag": results["labels"],
            "Confidence": results["scores"]
        }).sort_values("Confidence", ascending=False)
        progress_bar.progress(85)
        
        # Visualization
        fig, ax = plt.subplots(figsize=(8, 4))
        redflags_df.sort_values("Confidence").plot.barh(
            x="Red Flag", y="Confidence", ax=ax)
        plt.title("Risk Confidence Scores")
        plt.xlabel("Confidence Score")
        plt.tight_layout()
        
        # Report generation
        status_text.markdown("📊 **Compiling reports...**")
        csv_report = redflags_df.to_csv(index=False).encode()
        pdf_report = create_pdf_report(full_summary, redflags_df, fig)
        progress_bar.progress(100)
        
        # Display results
        st.success(f"Analysis completed in {time.time()-start_time:.1f} seconds")
        st.subheader(f"Document Insights (Processed {pages_processed} pages)")
        
        # Summary Tabs
        tab1, tab2 = st.tabs(["Paragraph Summary", "Bullet Points"])
        with tab1:
            st.write(full_summary)
        with tab2:
            st.markdown("\n".join([f"- {point}" for point in full_summary.split(". ")]))
        
        # Analysis Section
        col1, col2 = st.columns([2, 3])
        with col1:
            st.dataframe(
                redflags_df.style.format({"Confidence": "{:.1%}"}),
                height=400
            )
        with col2:
            st.pyplot(fig)
        
        # Report Downloads
        st.download_button(
            label="📥 Download CSV Report",
            data=csv_report,
            file_name="legal_analysis.csv",
            mime="text/csv"
        )
        
        st.download_button(
            label="📄 Download PDF Report",
            data=pdf_report,
            file_name="legal_analysis.pdf",
            mime="application/pdf"
        )

if __name__ == "__main__":
    main()
