
import streamlit as st
import pdfplumber
from transformers import pipeline
from keybert import KeyBERT
from googletrans import Translator

st.set_page_config(page_title="Legal Document Summarizer", layout="wide")

st.markdown(
    "<h1 style='text-align: center; color: #00ffae;'>⚖️ Legal Document Summarizer</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center; font-size: 18px;'>Summarize, extract keywords, translate, and ask questions about legal documents.</p>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("📁 Upload Legal Document (PDF only)", type=["pdf"])

if uploaded_file is not None:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ' '.join([page.extract_text() for page in pdf.pages if page.extract_text()])

    if not text:
        st.error("⚠️ No readable text found in the PDF.")
    else:
        st.success("✅ Document processed successfully!")

        # Summarize
        with st.spinner("Summarizing..."):
            summarizer = pipeline("summarization", model="t5-small")
            summary = summarizer(text[:1000])[0]['summary_text']

        st.subheader("📄 Summary (Plain Text)")
        st.code(summary, language='text')

        # Bullet points
        bullet_summary = "\n".join([f"- {line.strip().capitalize()}" for line in summary.split('.') if line.strip()])
        st.subheader("🧾 Summary (Bullet Points)")
        st.markdown(bullet_summary)

        # Keywords
        with st.spinner("Extracting Keywords..."):
            kw_model = KeyBERT()
            keywords = kw_model.extract_keywords(text, top_n=10)

        st.subheader("🔑 Keywords")
        for kw, score in keywords:
            st.markdown(f"- **{kw}** (Score: {score:.2f})")

        # Translations
        with st.spinner("🔁 Translating..."):
            translator = Translator()
            hindi = translator.translate(summary, dest='hi').text
            marathi = translator.translate(summary, dest='mr').text

        st.subheader("🇮🇳 Hindi Translation")
        st.write(hindi)

        st.subheader("🇮🇳 Marathi Translation")
        st.write(marathi)

        # Q&A Section
        st.subheader("❓ Ask Questions About the Document")
        user_question = st.text_input("Type your question here:")
        if user_question:
            with st.spinner("Thinking..."):
                qa_pipeline = pipeline("question-answering")
                answer = qa_pipeline(context=text[:3000], question=user_question)
                st.markdown(f"**Answer:** {answer['answer']}")

        # Download Summary
        summary_output = f"SUMMARY (Plain):\n{summary}\n\nSUMMARY (Bullets):\n{bullet_summary}\n\nKEYWORDS:\n" +                          "\n".join([f"{kw[0]} ({kw[1]:.2f})" for kw in keywords]) +                          f"\n\nHindi:\n{hindi}\n\nMarathi:\n{marathi}"
        st.download_button("⬇️ Download Summary as TXT", summary_output, file_name="legal_summary.txt")
    st.markdown("---")
    st.markdown("<p style='text-align: center;'>Made with ❤️ for Hackathons</p>", unsafe_allow_html=True)
else:
    st.info("Upload a PDF document to get started!")
