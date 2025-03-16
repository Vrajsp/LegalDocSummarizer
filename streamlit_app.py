# streamlit_app.py
import streamlit as st
import pdfplumber
import torch
import time
import numpy as np
import pandas as pd
from transformers import pipeline, AutoTokenizer
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from concurrent.futures import ThreadPoolExecutor

# Configure page
st.set_page_config(
    page_title="LegalMind Pro",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Cache optimized models
@st.cache_resource(ttl=3600)
def load_models():
    """Load quantized models with optimized settings"""
    return {
        'summarizer': pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        ),
        'classifier': pipeline(
            "zero-shot-classification",
            model="typeform/distilbert-base-uncased-mnli",
            device=0 if torch.cuda.is_available() else -1
        )
    }

# Professional CSS
st.markdown("""
<style>
:root {
    --primary: #1A73E8;
    --secondary: #34A853;
    --background: #F8F9FA;
    --text: #202124;
}

* {
    font-family: 'Google Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: var(--background);
}

.header-container {
    background: white;
    padding: 2rem;
    box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

.analysis-card {
    background: white;
    border-radius: 12px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    transition: transform 0.2s;
}

.analysis-card:hover {
    transform: translateY(-2px);
}

.progress-bar {
    height: 6px;
    background: rgba(0,0,0,0.1);
    border-radius: 3px;
    margin: 1rem 0;
}

.progress-fill {
    height: 100%;
    background: var(--primary);
    border-radius: 3px;
    transition: width 0.4s ease;
}

.stButton>button {
    background: var(--primary) !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.8rem 2rem !important;
    border: none !important;
    font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

def process_pdf(uploaded_file):
    """Efficient PDF text extraction with metadata"""
    with pdfplumber.open(uploaded_file) as pdf:
        meta = pdf.metadata
        text = "\n".join([page.extract_text() for page in pdf.pages[:50]])  # Limit to 50 pages
    return {
        'text': text,
        'pages': len(pdf.pages),
        'author': meta.get('Author', 'Unknown'),
        'title': meta.get('Title', uploaded_file.name)
    }

def analyze_document(text, models):
    """Parallel document processing"""
    with ThreadPoolExecutor() as executor:
        summary_future = executor.submit(
            models['summarizer'],
            text[:10000],  # Process first 10k chars for speed
            max_length=300,
            min_length=100,
            do_sample=False
        )
        
        risks_future = executor.submit(
            models['classifier'],
            text[:5000],
            candidate_labels=[
                "unfair termination", "non-compete", "salary delay",
                "discrimination", "harassment", "no severance",
                "short notice", "mandatory arbitration",
                "poor benefits", "intellectual property"
            ],
            multi_label=True
        )
        
        return {
            'summary': summary_future.result()[0]['summary_text'],
            'risks': pd.DataFrame({
                'Risk': risks_future.result()['labels'],
                'Confidence': risks_future.result()['scores']
            }).sort_values('Confidence', ascending=False)
        }

def create_executive_report(data, fig):
    """Generate professional PDF report"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Header
    story.append(Paragraph(f"<b>{data['title']}</b>", styles['Title']))
    story.append(Spacer(1, 12))
    
    # Metadata
    meta_table = Table([
        ["Author", data['author']],
        ["Pages", str(data['pages'])],
        ["Analyzed On", pd.Timestamp.now().strftime("%Y-%m-%d")]
    ], colWidths=[100, 400])
    story.append(meta_table)
    story.append(Spacer(1, 24))
    
    # Summary
    story.append(Paragraph("<b>Executive Summary</b>", styles['Heading2']))
    story.append(Paragraph(data['summary'], styles['BodyText']))
    story.append(Spacer(1, 24))
    
    # Risk Analysis
    story.append(Paragraph("<b>Risk Analysis</b>", styles['Heading2']))
    img_buffer = BytesIO()
    fig.write_image(img_buffer)
    story.append(Image(img_buffer, width=500, height=300))
    story.append(Spacer(1, 24))
    
    # Detailed Findings
    story.append(Paragraph("<b>Key Findings</b>", styles['Heading3']))
    risk_items = [
        [Paragraph(f"• {row['Risk']}", styles['BodyText']),
         Paragraph(f"{row['Confidence']:.1%}", styles['BodyText'])]
        for _, row in data['risks'].iterrows()
    ]
    risk_table = Table(risk_items, colWidths=[400, 100])
    story.append(risk_table)
    
    doc.build(story)
    return buffer

def main():
    """Main application flow"""
    st.markdown('<div class="header-container">', unsafe_allow_html=True)
    st.title("⚖️ LegalMind Pro")
    st.markdown("Enterprise-grade legal document analysis with AI-powered insights")
    st.markdown('</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        with st.container():
            st.subheader("Document Upload")
            uploaded_file = st.file_uploader(
                "Upload legal document (PDF)",
                type="pdf",
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                with st.spinner("Analyzing document..."):
                    start_time = time.time()
                    models = load_models()
                    
                    # Process document
                    doc_data = process_pdf(uploaded_file)
                    analysis = analyze_document(doc_data['text'], models)
                    
                    # Create visualization
                    fig = px.bar(
                        analysis['risks'],
                        x='Confidence',
                        y='Risk',
                        orientation='h',
                        color='Confidence',
                        color_continuous_scale='Bluered'
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)'
                    )
                    
                    # Generate report
                    report_buffer = create_executive_report({
                        **doc_data,
                        **analysis
                    }, fig)
                    
                    st.success(f"Analysis completed in {time.time()-start_time:.1f}s")
    
    if uploaded_file:
        with col2:
            tab1, tab2, tab3 = st.tabs(["Summary", "Risk Analysis", "Full Report"])
            
            with tab1:
                st.subheader("Document Summary")
                st.markdown(f'<div class="analysis-card">{analysis["summary"]}</div>', 
                           unsafe_allow_html=True)
                
                st.subheader("Metadata")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Pages", doc_data['pages'])
                col_b.metric("Author", doc_data['author'])
                col_c.metric("Processed In", f"{time.time()-start_time:.1f}s")
            
            with tab2:
                st.subheader("Risk Assessment")
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(
                    analysis['risks'].style.format({'Confidence': '{:.1%}'}),
                    height=400,
                    use_container_width=True
                )
            
            with tab3:
                st.subheader("Generate Reports")
                col_x, col_y = st.columns(2)
                
                with col_x:
                    st.download_button(
                        "Download PDF Report",
                        report_buffer.getvalue(),
                        "legal_analysis.pdf",
                        "application/pdf",
                        key='pdf-report'
                    )
                
                with col_y:
                    st.download_button(
                        "Export Data (CSV)",
                        analysis['risks'].to_csv().encode(),
                        "risk_analysis.csv",
                        "text/csv"
                    )

if __name__ == "__main__":
    main()
