# streamlit_app.py
import streamlit as st
import pdfplumber
import torch
from transformers import pipeline
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO
import time
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Apple-inspired CSS
st.markdown("""
<style>
:root {
    --space-black: #1D1D1F;
    --silver: #F5F5F7;
    --gold: #FFD700;
    --gradient-start: #000000;
    --gradient-end: #2C2C2E;
}

* {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
}

.stApp {
    background: linear-gradient(165deg, var(--gradient-start) 30%, var(--gradient-end) 100%);
}

.apple-card {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(10px);
    border-radius: 18px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    padding: 2rem;
    margin: 1.5rem 0;
    transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.apple-card:hover {
    transform: translateY(-4px);
}

.apple-button {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: white !important;
    border-radius: 12px;
    padding: 0.8rem 2rem;
    font-weight: 500;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.apple-button:hover {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.1));
    border: 1px solid rgba(255, 255, 255, 0.2);
    transform: scale(1.03);
}

.progress-bar {
    height: 4px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 2px;
    overflow: hidden;
}

.progress-fill {
    height: 100%;
    background: var(--gold);
    width: 0%;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.dynamic-header {
    font-size: 2.5rem;
    font-weight: 700;
    background: linear-gradient(90deg, #FFF, var(--silver));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
    margin-bottom: 1.5rem;
}
</style>
""", unsafe_allow_html=True)

def apple_style_header():
    st.markdown("""
    <div style="text-align: center; padding: 4rem 0 2rem;">
        <h1 class="dynamic-header">LegalDoc Pro</h1>
        <p style="color: rgba(255, 255, 255, 0.6); font-size: 1.1rem;">
            Intelligent Document Analysis · Precision at Scale
        </p>
    </div>
    """, unsafe_allow_html=True)

def minimalist_uploader():
    return st.file_uploader(
        " ",
        type=["pdf"],
        key="luxury-uploader",
        help="Drag and drop legal document",
        label_visibility="collapsed"
    )

def create_apple_chart(df):
    fig = px.bar(
        df,
        x='Confidence',
        y='Red Flag',
        orientation='h',
        color='Confidence',
        color_continuous_scale='Viridis',
        text='Confidence',
        height=400
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False),
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    fig.update_traces(
        texttemplate='%{text:.0%}',
        textposition='outside',
        marker_line_width=0,
        textfont_size=14
    )
    
    return fig

def generate_luxury_report(summary, df, fig):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    
    story = []
    
    # Header
    header = Paragraph("<font color='#FFFFFF'>LegalDoc Pro Analysis Report</font>", 
                      styles['Title'])
    story.append(header)
    story.append(Spacer(1, 24))
    
    # Summary
    summary_style = styles["BodyText"].clone('summary')
    summary_style.textColor = colors.HexColor('#FFFFFF')
    story.append(Paragraph("<b>Executive Summary:</b>", summary_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph(summary, summary_style))
    story.append(Spacer(1, 24))
    
    # Chart
    img_buffer = BytesIO()
    fig.write_image(img_buffer)
    story.append(Image(img_buffer, width=500, height=300))
    
    # Findings
    story.append(Spacer(1, 24))
    story.append(Paragraph("<b>Key Findings:</b>", summary_style))
    for _, row in df.iterrows():
        p = Paragraph(
            f"▪ {row['Red Flag']}: <font color='#FFD700'>{row['Confidence']:.0%}</font>",
            summary_style
        )
        story.append(p)
        story.append(Spacer(1, 8))
    
    doc.build(story)
    return buffer

def main():
    apple_style_header()
    
    with st.container():
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            <div style="padding: 2rem; text-align: center;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">⚖️</div>
                <p style="color: rgba(255, 255, 255, 0.8); line-height: 1.6;">
                    Advanced AI-powered legal document analysis with precision-engineered insights.
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            with st.markdown('<div class="apple-card">', unsafe_allow_html=True):
                uploaded_file = minimalist_uploader()
                
                if uploaded_file:
                    st.markdown("""
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: 100%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    with st.spinner(" "):
                        time.sleep(2)  # Simulated processing
                        
                        # Demo data
                        summary = "This comprehensive analysis identifies key provisions and potential risk factors..."
                        df = pd.DataFrame({
                            'Red Flag': ['Non-compete Clause', 'Termination Terms', 'Arbitration Agreement'],
                            'Confidence': [0.92, 0.85, 0.78]
                        })
                        
                        with st.container():
                            st.markdown("""
                            <div style="margin: 2rem 0;">
                                <h3 style="color: white; margin-bottom: 1rem;">Document Insights</h3>
                            """, unsafe_allow_html=True)
                            
                            fig = create_apple_chart(df)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Report generation
                            report_buffer = generate_luxury_report(summary, df, fig)
                            
                            # Download buttons
                            col_a, col_b = st.columns(2)
                            with col_a:
                                st.download_button(
                                    "Export PDF Report",
                                    report_buffer.getvalue(),
                                    "legal_analysis.pdf",
                                    "application/pdf",
                                    key='pdf-report',
                                    use_container_width=True,
                                    type='primary'
                                )
                            
                            with col_b:
                                st.download_button(
                                    "Export Data",
                                    df.to_csv().encode(),
                                    "analysis_data.csv",
                                    "text/csv",
                                    use_container_width=True
                                )

if __name__ == "__main__":
    main()
