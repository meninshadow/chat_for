import os
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import json
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


# Initialize the Flask app
app = Flask(__name__)

# Define the allowed file extensions for the file upload endpoint
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

################## this is GEMINI WORK ##########################
## ConversationalRetrievalChain for memory 

# --- Configuration ---
# Retrieve API key securely from environment variables
# (Cloud Run injects these, DO NOT hardcode in production)
def createGeminiModel(file_path, website):
    os.environ["GOOGLE_API_KEY"] = os.getenv("OUR_GOOGLE_API_KEY")
    save_directory = "./upload/faiss_index/"+website+"/"
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
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

    response_data = {
        "status": "success",
        "message": "JSON data received. File at given path is ready for processing.",
        "received_file_path": file_path,
        "received_website_name": website,
        "mode_saved":"success"
    }

    #return vector_store
    return response_data

   
################## this is GEMINI WORK end ##########################



def allowed_file(filename):
    """Checks if the file's extension is in the allowed list."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the API! This endpoint handles both file uploads and JSON data for file processing."


@app.route('/api/upload_file', methods=['POST'])
def upload_data_file():
    # --- Check the Content-Type of the request ---
    content_type = request.content_type

    if content_type and 'application/json' in content_type:
        # --- Handling JSON payload ---
        try:
            data = request.get_json(force=True)
            file_path = data.get("file_path")
            website_name = data.get("website_name")
        except Exception as e:
            return jsonify({"error": f"Invalid JSON or missing fields: {e}"}), 400

        if not file_path or not website_name:
            return jsonify({"error": "JSON body must contain 'file_path' and 'website_name'"}), 400
        
        # This part of the code needs to be updated with your actual file processing logic.
        print(f"Received JSON data for processing:")
        print(f"File Path: {file_path}")
        print(f"Website Name: {website_name}")

        response_data = createGeminiModel(file_path, website_name)
        return jsonify(response_data), 200

    else:
        # --- Handling multipart/form-data (original functionality) ---
        if 'data_file' not in request.files:
            return jsonify({"error": "No 'data_file' part in the request"}), 400
        
        file = request.files['data_file']
        website_name = request.form.get('website_name')

        if not website_name:
            return jsonify({"error": "No website_name provided in the form data"}), 400

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            safe_website_name = secure_filename(website_name)
            upload_path = os.path.join('uploads', safe_website_name)
            os.makedirs(upload_path, exist_ok=True)

            original_filename = secure_filename(file.filename)
            name_part, extension = os.path.splitext(original_filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{name_part}_{timestamp}{extension}"
            full_filepath = os.path.join(upload_path, new_filename)
            file.save(full_filepath)

            # --- NEW: Append the record to current_file.json ---
            current_file_path = os.path.join(upload_path, "current_file.json")
            
            # The new record to be added
            new_record = {
                "file_path": full_filepath,
                "website_name": website_name
            }
            
            # Load existing records if the file exists, otherwise start with an empty list
            records = []
            if os.path.exists(current_file_path):
                try:
                    with open(current_file_path, "r") as json_file:
                        records = json.load(json_file)
                except (json.JSONDecodeError, FileNotFoundError):
                    # Handle cases where the file is empty or corrupted
                    records = []

            # Append the new record to the list
            records.append(new_record)
            
            # Write the complete, updated list back to the file
            with open(current_file_path, "w") as json_file:
                json.dump(records, json_file, indent=2)
            # --- END OF NEW CODE ---
            
            print(f"File uploaded and saved to: {full_filepath}")
            return jsonify({
                "message": "File uploaded successfully",
                "saved_path": full_filepath
            }), 200
        
        return jsonify({
            "error": f"Invalid file type. Allowed types are: {list(ALLOWED_EXTENSIONS)}"
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5001)  