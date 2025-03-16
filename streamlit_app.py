import streamlit as st
import pdfplumber
import torch
from transformers import pipeline
import pandas as pd
import plotly.express as px
from io import BytesIO
import time
from concurrent.futures import ThreadPoolExecutor

# Modern CSS for a sleek UI
st.markdown("""
<style>
:root {
    --primary: #2563eb;
    --secondary: #4f46e5;
    --background: #0f172a;
    --text: #f8fafc;
    --card-bg: rgba(15, 23, 42, 0.7);
    --card-border: rgba(255, 255, 255, 0.1);
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
    background: var(--background);
    color: var(--text);
}

.header {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    padding: 2rem 0;
    border-bottom: 1px solid var(--card-border);
    text-align: center;
}

.card {
    background: var(--card-bg);
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid var(--card-border);
    backdrop-filter: blur(10px);
    margin: 1rem 0;
    transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
}

.progress {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
}

.progress-bar {
    height: 100%;
    background: linear-gradient(90deg, var(--primary), var(--secondary));
    transition: width 0.4s ease;
}

.stButton>button {
    background: linear-gradient(135deg, var(--primary), var(--secondary)) !important;
    border: none !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 0.75rem 1.5rem !important;
    transition: transform 0.2s !important;
}

.stButton>button:hover {
    transform: translateY(-2px);
}

h1, h2, h3 {
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# Load models with caching
@st.cache_resource
def load_models():
    """Load optimized models for summarization and classification"""
    return {
        'summarizer': pipeline(
            "summarization",
            model="philschmid/bart-large-cnn-samsum",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        ),
        'classifier': pipeline(
            "zero-shot-classification",
            model="valhalla/distilbart-mnli-12-3",
            device=0 if torch.cuda.is_available() else -1
        )
    }

# Extract text from PDF
def extract_text(file, max_pages=20):
    """Extract text from PDF with a page limit"""
    with pdfplumber.open(file) as pdf:
        return " ".join([p.extract_text() for p in pdf.pages[:max_pages]])

# Process document in parallel
def process_document(text, models):
    """Parallel processing for summarization and risk detection"""
    with ThreadPoolExecutor() as executor:
        summary_future = executor.submit(
            models['summarizer'], 
            text[:3000],  # Optimized input size
            max_length=256,
            min_length=128,
            do_sample=False
        )
        
        risks_future = executor.submit(
            models['classifier'],
            text[:2000],
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
            }).head(5)
        }

# Main app
def main():
    """Main application interface"""
    st.markdown('<div class="header">', unsafe_allow_html=True)
    st.title("⚡ LegalDocSummarizer Pro")
    st.markdown("AI-powered legal document analysis with real-time insights")
    st.markdown('</div>', unsafe_allow_html=True)

    # File uploader
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    
    if uploaded_file:
        start_time = time.time()
        models = load_models()
        
        with st.status("Analyzing document...", expanded=True) as status:
            # Extract text
            st.write("📄 Extracting text...")
            text = extract_text(uploaded_file)
            
            # Process document
            st.write("🧠 Processing content...")
            results = process_document(text, models)
            
            # Generate visualization
            st.write("📊 Generating insights...")
            fig = px.bar(
                results['risks'],
                x='Confidence',
                y='Risk',
                orientation='h',
                color='Confidence',
                color_continuous_scale='Bluered'
            )
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
            
            status.update(label="Analysis Complete", state="complete")
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📝 Executive Summary")
            st.write(results['summary'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🚨 Risk Analysis")
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                results['risks'].style.format({'Confidence': '{:.1%}'}),
                height=200,
                use_container_width=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance metrics
        st.markdown(f"""
        <div class="card">
            <div style="display: flex; justify-content: space-between;">
                <div>⏱️ Total Processing Time</div>
                <div style="color: #4ade80">{time.time()-start_time:.2f}s</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
