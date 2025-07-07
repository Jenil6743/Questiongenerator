# 📚 Indian Educational-Board Question Generator

A one-stop Streamlit app that **reads any curriculum PDF (even scanned!), builds an internal knowledge base, and auto-writes board-aligned exam questions & answer keys** in seconds.  Export everything as clean PDFs ready for the photocopier.

---

## ✨  What It Does

1. **Upload PDF** – text-based or scanned.  
2. **Text Extraction**  
   * Text PDFs → `PyPDF2`  
   * Scanned PDFs → `pdf2image` + `tesseract-ocr`  
3. **Vector Indexing** – splits text, embeds with **Google Generative AI** embeddings, stores in **FAISS** for retrieval.  
4. **Question Generation** – prompts **Groq’s Llama-3** to write MCQs, short/long answers, true-false, fill-in-blanks or match-the-following.  
5. **Optional Validation** – AI reviewer polishes clarity, difficulty & grammar.  
6. **Answer-Key Creation** – same model returns numbered solutions.  
7. **PDF Export** – questions and answer key rendered with **FPDF** and offered for download.

Everything runs client-side in Streamlit; no server is required beyond the LLM APIs.

---

## 🖼️ Interface at a Glance

| Sidebar | Main Area |
|---------|-----------|
| • Board / Class / Subject pickers  <br>• OCR toggle  <br>• PDF uploader  <br>• Question-type, complexity & count sliders  | • Live preview of extracted text  <br>• Buttons to generate questions and answers  <br>• Richly formatted output  <br>• Download buttons for Questions.pdf & AnswerKey.pdf |

---

## 🛠️ Prerequisites

| Requirement | Notes |
|-------------|-------|
| **Python 3.9 +** | |
| **Tesseract-OCR** | For scanned PDFs. Install system package (`sudo apt-get install tesseract-ocr`) or Windows binary. |
| **Poppler** | `pdf2image` backend. Install `poppler-utils` / download binaries for Windows. |
| **GROQ_API_KEY** | From [groq.com](https://console.groq.com). |
| **GOOGLE_API_KEY** | Generative AI Key with embeddings scope. |

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-org>/indian-ed-board-qg.git
cd indian-ed-board-qg
pip install -r requirements.txt   # see requirements.txt

# 2. Add API keys
mkdir -p .streamlit
cat > .streamlit/secrets.toml <<EOF
GROQ_API_KEY = "<your-groq-key>"
GOOGLE_API_KEY = "<your-google-gemini-key>"
EOF

# 3. Launch
streamlit run app.py
