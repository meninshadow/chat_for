import os
from datetime import datetime
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Initialize the Flask app
app = Flask(__name__)

# Define the allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xls', 'xlsx'}

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

        # At this point, you have the file_path and website_name from the JSON.
        # Now, you can process the file located at the given path.
        # This part of the code needs to be updated with your actual file processing logic.
        
        # We will use the existing file-saving logic to "replicate" the process,
        # but in a real-world scenario, you would be processing an *already* existing file.

        # --- Replicating the "save file" logic with the JSON data ---
        # This part assumes that the file_path from the JSON is a valid path on your server.
        # If the file exists, you can proceed to process it.
        # For demonstration, we'll just log the information.

        # You would typically do something like this with the path:
        # import pandas as pd
        # df = pd.read_excel(file_path)
        # ... process the dataframe ...

        # For now, let's return a success message based on the received JSON data.
        print(f"Received JSON data for processing:")
        print(f"File Path: {file_path}")
        print(f"Website Name: {website_name}")

        response_data = {
            "status": "success",
            "message": "JSON data received. File at given path is ready for processing.",
            "received_file_path": file_path,
            "received_website_name": website_name
        }
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