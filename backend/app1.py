from flask import Flask, request, jsonify, send_file
import os
import subprocess
from werkzeug.utils import secure_filename
import uuid
import logging
import time
from datetime import datetime
from PIL import Image  # Added for image validation

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')
UPLOADS_DIR = os.path.join(BACKEND_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MATLAB_TIMEOUT = 30  # seconds

# Ensure directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)  # Ensure uploads directory exists

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(filepath):
    """Verify the image file is valid using Pillow"""
    try:
        with Image.open(filepath) as img:
            img.verify()  # Verify the file is a valid image
        return True
    except Exception as e:
        logger.error(f"Invalid image file: {str(e)}")
        return False

def generate_unique_filename(original_filename):
    """Generate a unique filename with timestamp and UUID"""
    name, ext = os.path.splitext(secure_filename(original_filename))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"{name}_{timestamp}_{unique_id}{ext}"

@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing"""
    start_time = time.time()
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    # Generate unique filenames
    original_filename = secure_filename(file.filename)
    processed_filename = generate_unique_filename(file.filename)
    temp_path = os.path.join(UPLOADS_DIR, processed_filename)
    result_filename = f"{os.path.splitext(original_filename)[0]}_output{os.path.splitext(original_filename)[1]}"
    result_path = os.path.join(RESULTS_DIR, result_filename)

    try:
        # Save uploaded file
        file.save(temp_path)
        if not validate_image_file(temp_path):
            raise ValueError("Uploaded file is not a valid image")

        # Prepare MATLAB command
        matlab_cmd = [
            "matlab",
            "-batch",
            f"addpath('{BACKEND_DIR}'); try; detect_face_manual('{temp_path}'); catch e; disp(getReport(e, 'extended')); exit(1); end; exit;"
        ]
        
        logger.info(f"Executing MATLAB: {' '.join(matlab_cmd)}")
        
        try:
            result = subprocess.run(
            matlab_cmd,
            check=True,
            timeout=MATLAB_TIMEOUT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=BACKEND_DIR
            )
        
        # Parse MATLAB output for the exact path
            output_path = None
            for line in result.stdout.split('\n'):
                if line.startswith('OUTPUT_PATH:'):
                    output_path = line.split('OUTPUT_PATH:')[1].strip()
                    break
        
            if not output_path:
                raise Exception("Could not determine output path from MATLAB")
        
            if not os.path.exists(output_path):
                raise FileNotFoundError(f"Output file not found at {output_path}")
        
        # Generate final result filename
            original_name = secure_filename(file.filename)
            result_filename = f"{os.path.splitext(original_name)[0]}_output{os.path.splitext(original_name)[1]}"
            result_path = os.path.join(RESULTS_DIR, result_filename)
        
        # Move result to results directory
            if os.path.exists(result_path):
                os.remove(result_path)
            os.rename(output_path, result_path)
        
            return send_file(
                result_path,
                mimetype=f'image/{os.path.splitext(result_path)[1][1:]}',
                as_attachment=True,
                download_name=result_filename
           )
            
        except subprocess.TimeoutExpired:
            raise Exception("MATLAB processing timed out")
        except subprocess.CalledProcessError as e:
            error_msg = f"MATLAB error (code {e.returncode}): {e.stderr}"
            if result.stdout:
                error_msg += f"\nMATLAB output:\n{result.stdout}"
            raise Exception(error_msg)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Clean up files
        for path in [temp_path, result_path, temp_path.replace('.', '_output.')]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as cleanup_error:
                    logger.error(f"Failed to clean up {path}: {cleanup_error}")
        
        return jsonify({
            'error': 'Image processing failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/test', methods=['GET'])
def test_connection():
    """Basic connectivity test"""
    return jsonify({
        'status': 'Server is running',
        'results_dir': RESULTS_DIR,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
