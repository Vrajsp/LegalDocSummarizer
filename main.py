
import pdfplumber
from transformers import pipeline
from keybert import KeyBERT
from googletrans import Translator

# Extract text from PDF
def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return ' '.join([page.extract_text() for page in pdf.pages if page.extract_text()])

# Summarize text using T5
summarizer = pipeline("summarization", model="t5-small")
def summarize_text(text):
    return summarizer(text[:1000])[0]['summary_text']

# Extract keywords using KeyBERT
kw_model = KeyBERT()
def extract_keywords(text):
    return kw_model.extract_keywords(text, top_n=10)

# Translate text using Google Translator
translator = Translator()
def translate_summary(summary, dest='hi'):
    return translator.translate(summary, dest=dest).text

# Demo usage
if __name__ == "__main__":
    file_path = "sample_docs/sample_legal_doc.pdf"
    text = extract_text(file_path)
    summary = summarize_text(text)
    keywords = extract_keywords(text)
    translation = translate_summary(summary, 'hi')

    print("Summary:", summary)
    print("Keywords:", keywords)
    print("Hindi Translation:", translation)
