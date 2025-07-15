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

# Renamed endpoint to be more generic

@app.route('/', methods=['GET'])


# Renamed endpoint to be more generic
@app.route('/api/upload_file', methods=['POST'])
def upload_data_file():
    # --- 1. Check if all required data is in the request ---
    
    # Renamed the file part to 'data_file' to be more generic
    if 'data_file' not in request.files:
        return jsonify({"error": "No data_file part in the request"}), 400

    file = request.files['data_file']
    website_name = request.form.get('website_name')

    if not website_name:
        return jsonify({"error": "No website_name provided in the form data"}), 400

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # --- 2. If a file is provided, validate and save it ---
    if file and allowed_file(file.filename):
        # Sanitize the website name to create a safe directory name
        safe_website_name = secure_filename(website_name)

        # Create the target directory: uploads/website_name/
        upload_path = os.path.join('uploads', safe_website_name)
        os.makedirs(upload_path, exist_ok=True)

        # Sanitize the original filename
        original_filename = secure_filename(file.filename)
        name_part, extension = os.path.splitext(original_filename)

        # Create a timestamp string (e.g., 20250715_132137)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create the new filename
        new_filename = f"{name_part}_{timestamp}{extension}"

        # Combine the path and new filename and save the file
        full_filepath = os.path.join(upload_path, new_filename)
        file.save(full_filepath)

        # --- 3. Return a success response ---
        print(f"File saved to: {full_filepath}") # For logging on Render
        return jsonify({
            "message": "File uploaded successfully",
            "saved_path": full_filepath
        }), 200
    
    # If the file extension is not allowed, return an error
    return jsonify({
        "error": f"Invalid file type. Allowed types are: {list(ALLOWED_EXTENSIONS)}"
    }), 400


# This part is for local testing and not used by Render's Gunicorn server
if __name__ == '__main__':
    app.run(debug=True, port=5001)