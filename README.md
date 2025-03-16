# 🧠 LegalDocSummarizer

![Logo](https://i.imgur.com/Hf1JcLn.png)

A powerful AI-powered tool for NGOs, citizens, and legal consultants to summarize legal documents and detect risky clauses — all in one click.

---

## Features

 Upload PDF legal docs  
 Summarize complex content using AI (T5)  
 Bullet-point & paragraph summary views  
 Zero-shot Red Flag Detection using BART  
 Visual bar chart of risk scores  
 Download reports as PDF or CSV  
 Built with Streamlit (one-click web UI)

---

## 📸 Demo Screenshot

![Screenshot](https://i.imgur.com/rHJjAnK.png)

---

## 🧰 Tech Stack

- Streamlit (UI)
- Hugging Face Transformers
  - `t5-small` for summarization
  - `facebook/bart-large-mnli` for red flag detection
- Matplotlib (bar chart)
- pdfplumber (PDF parsing)
- Pandas, ReportLab (export CSV & PDF)

---

## ⚙️ Installation

```bash
git clone https://github.com/Vrajsp/LegalDocSummarizer.git
cd LegalDocSummarizer
pip install -r requirements.txt
streamlit run streamlit_app.py
```


## Use Cases

   - NGOs reviewing human rights policies
   - Legal advisors scanning environmental laws
   - Citizens reading contracts & government acts
   - Students analyzing legal case documents


