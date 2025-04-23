from flask import Flask, request, jsonify, send_file
import os
import subprocess
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    try:
        # Secure filename and create temp file path
        filename = secure_filename(file.filename)
        temp_path = os.path.join(BACKEND_DIR, filename)
        
        # Save temporarily
        file.save(temp_path)
        
        # Run MATLAB detection
        subprocess.run([
            "matlab",
            "-batch",
            f"addpath('{BACKEND_DIR}'); classify_image_simple('{temp_path}'); exit;"
        ], check=True)
        
        # Clean up temp file
        os.remove(temp_path)
        
        # Check for results
        result_filename = os.path.splitext(filename)[0] + '_output.jpg'
        result_path = os.path.join(RESULTS_DIR, result_filename)
        
        if not os.path.exists(result_path):
            return jsonify({'error': 'Processing completed but result file not found'}), 500
        
        # Return both the image and detection results
        return send_file(
            result_path,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=result_filename
        )
        
    except subprocess.CalledProcessError as e:
        return jsonify({'error': 'MATLAB processing failed', 'details': str(e)}), 500
    except Exception as e:
        return jsonify({'error': 'Unexpected error', 'details': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_connection():
    return jsonify({'status': 'Server is running', 
                   'results_dir': RESULTS_DIR})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)