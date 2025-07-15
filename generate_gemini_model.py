import pandas as pd
import os
import time
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredExcelLoader # Note: Requires installing unstructured
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.schema import HumanMessage, AIMessage

# --- Recommended Installations (for a standard environment) ---
# Uncomment and run these if you don't have the libraries installed.
# They are not valid Python code, but are included here for reference.
# !pip install pandas openpyxl
# !pip install -qU google-generativeai
# !pip install -qU langchain_google_genai langchain langchain-community
# !pip install -qU faiss-cpu
# !pip install -qU sentence-transformers
# !pip install unstructured # Unstructured for Excel loading

# --- Step 1: Set up the Gemini API Key ---
# IMPORTANT: Never hardcode your API key in production. Use environment variables.
os.environ["GOOGLE_API_KEY"] = os.getenv("OUR_GOOGLE_API_KEY")

# --- RAG Setup ---
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


# --- Step 2: Function to Create and Save the FAISS Index ---
def create_faiss_index(file_path, save_directory):
    """
    Loads data from an Excel file, creates documents, and builds a FAISS index.
    """
    if not os.path.exists(file_path):
        print(f"Error: Excel file not found at {file_path}")
        return None

    print(f"Loading data from {file_path}...")
    df = pd.read_excel(file_path)

    documents = []
    for index, row in df.iterrows():
        content_parts = [f"{col}: {val}" for col, val in row.items() if pd.notna(val)]
        content = "\n".join(content_parts)
        
        metadata = {col: str(row[col]) for col in df.columns if pd.notna(row[col])}
        metadata['source_file'] = file_path
        metadata['row_number'] = index
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Number of documents/chunks created from Excel rows: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Number of chunks created: {len(chunks)}")

    print("Creating FAISS Vector Store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_directory)
    print("FAISS Vector Store created and saved successfully.")
    return vector_store

# --- Main Execution Block ---
if __name__ == "__main__":
    EXCEL_FILE_PATH = "opencart_data_1.xlsx"
    FAISS_SAVE_DIRECTORY = "./faiss_index_opencart"

    # --- Run this block to create the FAISS index ---
    vector_store_from_creation = create_faiss_index(EXCEL_FILE_PATH, FAISS_SAVE_DIRECTORY)

    # --- Step 3: Load the saved FAISS Vector Store ---
    if os.path.exists(FAISS_SAVE_DIRECTORY):
        print(f"Loading FAISS index from {FAISS_SAVE_DIRECTORY}...")
        try:
            vector_store = FAISS.load_local(FAISS_SAVE_DIRECTORY, embeddings, allow_dangerous_deserialization=True)
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            exit()
    else:
        print(f"Error: FAISS index directory not found at {FAISS_SAVE_DIRECTORY}.")
        print("Please run the index creation step first.")
        exit()