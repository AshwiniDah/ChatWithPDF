import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# ✅ Load API key from .env file
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# ✅ Read and combine text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# ✅ Read and combine text from preloaded local PDFs (e.g. from "pdfs" folder)
def load_preloaded_pdfs(folder):
    folder_path = os.path.abspath(folder)
    st.write("🔍 Looking in folder:", folder_path)

    if not os.path.exists(folder_path):
        st.warning("⚠️ Folder does not exist.")
        return ""

    text = ""
    files = [f for f in os.listdir(folder_path) if f.endswith(".pdf")]

    if not files:
        st.warning("⚠️ No PDF files found in folder.")
        return ""

    st.write("📂 Preloaded files found:", files)

    for filename in files:
        file_path = os.path.join(folder_path, filename)
        with open(file_path, "rb") as f:
            pdf_reader = PdfReader(f)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
    return text

# ✅ Split large text into manageable chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# ✅ Generate and store embeddings using FAISS vector DB
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    batch_size = 5
    vector_store = None

    for i in range(0, len(text_chunks), batch_size):
        batch = text_chunks[i:i + batch_size]
        for attempt in range(3):
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    new_vectors = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(new_vectors)
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(5)
                else:
                    st.error(f"❌ Failed on batch {i//batch_size + 1}: {e}")
                    return
    if vector_store:
        vector_store.save_local("faiss_index")

# ✅ Setup Gemini conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, just say: "answer is not available in the context."

     Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# ✅ Process user question and display answer
def user_input(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
    st.write("💬 Reply:", response["output_text"])

# ✅ Streamlit UI
def main():
    st.set_page_config(page_title="PDF Chatbot")
    st.header("🤖 Chat with PDF using Gemini + FAISS")

    # User question input
    question = st.text_input("Ask a question based on the PDFs:")

    if question:
        user_input(question)

    # Sidebar for PDF upload and processing
    with st.sidebar:
        st.title("📄 PDF Loader")

        pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

        if st.button("🔍 Submit & Process"):
            with st.spinner("Processing PDFs..."):
                try:
                    # Combine uploaded and preloaded PDFs
                    uploaded_text = get_pdf_text(pdf_docs) if pdf_docs else ""
                    preloaded_text = load_preloaded_pdfs("pdfs")
                    all_text = uploaded_text + preloaded_text

                    if not all_text.strip():
                        st.warning("⚠️ No text found to process.")
                        return

                    chunks = get_text_chunks(all_text)
                    get_vector_store(chunks)

                    st.success("✅ Ready! Ask questions above.")
                except Exception as e:
                    st.error(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
