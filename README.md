# 📚 PDF Chatbot using Gemini + FAISS

This is a simple chatbot app that lets you chat with PDFs. It uses Google’s Gemini API for smart answers and FAISS for fast searching through PDF content. You can upload your own PDFs or load files from a local folder.

🚀 What It Does
Reads and combines text from PDF files (uploaded or from a folder).
Breaks the text into small chunks so it's easy to search.
Converts the text chunks into embeddings using GoogleGenerativeAIEmbeddings.
Stores these embeddings using FAISS, a fast vector search engine.
Uses Gemini to answer questions based on the text inside your PDFs.

You can ask questions like:
👉 "What is mentioned about GST?"
👉 "Summarize this document"
and it will reply smartly using the content inside your PDFs!

🧰 Tech Used
Python
Streamlit – for the simple web interface
PyPDF2 – for reading PDFs
LangChain – to manage the question-answer flow
FAISS – for vector database (fast document search)
Google Gemini API – to understand and answer questions


## 🛠 How to Run

Clone the repo:

```bash
git clone <your-repo-url>
cd <project-folder>
```


 Create a .env file and add your Gemini API key:

```bash
GOOGLE_API_KEY=your_api_key_here
```

## Install the required packages

```bash
pip install -r requirements.txt
```

## Run the app
```bash
streamlit run main.py
```

## 📁 Folder Structure
```bash
📂 your_project/
├── main.py             # Streamlit app
├── pdfs/               # Folder to store local PDF files
├── faiss_index/        # FAISS index directory (auto-created)
├── .env                # API key file
└── README.md           # Project description
```

💡 Features
Upload multiple PDF files
Automatically reads PDFs from a pdfs/ folder
Merges uploaded + local files into one searchable source
Builds a local FAISS index for fast search
Smart answers using Google Gemini
Easy-to-use Streamlit interface

❗ Notes
Works best with text-based PDFs (not scanned images).
If you upload new PDFs, click Submit & Process again to rebuild the index.
Make sure .env contains a valid GOOGLE_API_KEY.

🙋‍♀️ Example Usage
Place a few PDFs in the pdfs/ folder.

## Run the app:
```bash
streamlit run main.py
```

## Ask a question like:
What is the conclusion of the report?

✅ Output
Gemini gives accurate, context-based replies from your documents.
