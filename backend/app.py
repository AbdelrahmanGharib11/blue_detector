from flask import Flask, request, jsonify, send_file
import os
import subprocess
import logging
import time
import uuid
from datetime import datetime
from werkzeug.utils import secure_filename
from PIL import Image  # For image validation
import base64
import json

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("flask_matlab_bridge.log"),
             logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))

# Directories for first MATLAB application (detect2)
RESULTS_DIR = os.path.join(BACKEND_DIR, 'results')
UPLOADS_DIR = os.path.join(BACKEND_DIR, 'uploads')

# Directories for second MATLAB application (face recognition)
RECOGNITION_UPLOAD_FOLDER = os.path.join(BACKEND_DIR, 'recognition_uploads')
RECOGNITION_RESULTS_FOLDER = os.path.join(BACKEND_DIR, 'recognition_results')
MATLAB_SCRIPT_PATH = os.path.join(BACKEND_DIR, 'face_recognition.m')

# Ensure all directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RECOGNITION_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RECOGNITION_RESULTS_FOLDER, exist_ok=True)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

# Timeouts for MATLAB processing
MATLAB_TIMEOUT = 30  # seconds for detect2
RECOGNITION_TIMEOUT = 60  # seconds for face recognition

# Helper functions
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

def encode_image_to_base64(image_path):
    """Convert an image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    except Exception as e:
        logger.error(f"Error encoding image to base64: {str(e)}")
        return None

# First MATLAB application route (detect2)
@app.route('/upload', methods=['POST'])
def upload_image():
    """Handle image upload and processing with detect2 MATLAB function"""
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

        # Prepare MATLAB command for detect2
        matlab_cmd = [
            "matlab",
            "-batch",
            f"addpath('{BACKEND_DIR}'); try; detect_face_manual('{temp_path}'); catch e; disp(getReport(e, 'extended')); exit(1); end; exit;"
        ]
        
        logger.info(f"Executing MATLAB detect_face_manual: {' '.join(matlab_cmd)}")
        
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

# Second MATLAB application routes (face recognition)
@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint for face recognition service"""
    return jsonify({
        'status': 'ok', 
        'message': 'Face recognition service is running',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/recognize', methods=['POST'])
def recognize_face():
    """
    API endpoint to recognize a face from an uploaded image
    Accepts image in multipart/form-data format
    Returns recognition results including matched face, confidence, and result images
    """
    try:
        logger.info("Received face recognition request")
        
        # Check if the post request has the file part
        if 'image' not in request.files:
            logger.warning("No image part in the request")
            return jsonify({'error': 'No image part'}), 400
        
        file = request.files['image']
        
        # If user does not select file, browser may also submit an empty part without filename
        if file.filename == '':
            logger.warning("No image selected")
            return jsonify({'error': 'No image selected'}), 400
        
        if file and allowed_file(file.filename):
            # Create a timestamped folder for this request
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            request_id = f"{timestamp}_{uuid.uuid4().hex[:8]}"
            request_folder = os.path.join(RECOGNITION_UPLOAD_FOLDER, f"request_{request_id}")
            os.makedirs(request_folder, exist_ok=True)
            
            # Save the uploaded file
            input_image_path = os.path.join(request_folder, "input_image.jpg")
            file.save(input_image_path)
            logger.info(f"Image saved to: {input_image_path}")
            
            # Prepare output file path for MATLAB results
            output_json_path = os.path.join(request_folder, "result.json")
            
            # Call MATLAB script via subprocess
            try:
                # Command to run MATLAB in non-interactive mode with our script
                matlab_cmd = [
                    'matlab', 
                    '-nodisplay', 
                    '-nodesktop',
                    '-nosplash', 
                    '-batch', 
                    f"addpath('{BACKEND_DIR}'); try; face_recognition('{input_image_path}', '{output_json_path}'); catch e; disp(getReport(e, 'extended')); exit(1); end; exit;"
                ]
                
                logger.info(f"Running MATLAB face_recognition: {' '.join(matlab_cmd)}")
                process = subprocess.run(
                    matlab_cmd,
                    check=True,
                    timeout=RECOGNITION_TIMEOUT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=BACKEND_DIR
                )
                
                # Log MATLAB output for debugging
                logger.info(f"MATLAB stdout: {process.stdout}")
                if process.stderr:
                    logger.warning(f"MATLAB stderr: {process.stderr}")
                
                # Check if result file exists
                if not os.path.exists(output_json_path):
                    logger.error(f"Output JSON file not found: {output_json_path}")
                    return jsonify({'error': 'Result file not generated'}), 500
                
                # Read the JSON result file
                with open(output_json_path, 'r') as f:
                    result_data = json.load(f)
                
                # Check if there was an error from MATLAB
                if 'error' in result_data and result_data['error'] == True:
                    logger.error(f"MATLAB processing error: {result_data.get('message', 'Unknown error')}")
                    return jsonify({
                        'error': 'Face recognition processing failed',
                        'details': result_data.get('message', 'Unknown error')
                    }), 500
                
                # Add base64 images to the response
                if 'comparison_image' in result_data and os.path.exists(result_data['comparison_image']):
                    # Convert comparison image to base64
                    base64_img = encode_image_to_base64(result_data['comparison_image'])
                    if base64_img:
                        result_data['comparison_image_base64'] = base64_img
                    
                    # Store the filename for the /api/results endpoint
                    comparison_filename = os.path.basename(result_data['comparison_image'])
                    result_copy_path = os.path.join(RECOGNITION_RESULTS_FOLDER, comparison_filename)
                    
                    # Copy the file to the results folder for direct access via /api/results
                    try:
                        import shutil
                        shutil.copy2(result_data['comparison_image'], result_copy_path)
                        # Update the path to be relative for the /api/results endpoint
                        result_data['comparison_image_url'] = f'/api/results/{comparison_filename}'
                    except Exception as copy_error:
                        logger.error(f"Error copying comparison image: {str(copy_error)}")
                
                # Also make matched image available via URL if it exists
                if 'matched_db_image' in result_data and result_data['matched_db_image'] and os.path.exists(result_data['matched_db_image']):
                    matched_filename = os.path.basename(result_data['matched_db_image'])
                    matched_copy_path = os.path.join(RECOGNITION_RESULTS_FOLDER, matched_filename)
                    
                    try:
                        import shutil
                        shutil.copy2(result_data['matched_db_image'], matched_copy_path)
                        result_data['matched_image_url'] = f'/api/results/{matched_filename}'
                    except Exception as copy_error:
                        logger.error(f"Error copying matched image: {str(copy_error)}")
                
                # Clean up response data - remove absolute paths for security
                for key in ['comparison_image', 'test_image', 'matched_db_image']:
                    if key in result_data:
                        # Keep just the filename part
                        result_data[f'{key}_filename'] = os.path.basename(result_data[key])
                        # Remove the absolute path
                        del result_data[key]
                
                return jsonify(result_data)
                
            except subprocess.TimeoutExpired:
                logger.error("MATLAB process timeout")
                return jsonify({'error': 'Processing timeout'}), 504
            except subprocess.CalledProcessError as e:
                error_msg = f"MATLAB error (code {e.returncode}): {e.stderr}"
                if e.stdout:
                    error_msg += f"\nMATLAB output:\n{e.stdout}"
                logger.error(error_msg)
                return jsonify({'error': error_msg}), 500
            except Exception as e:
                logger.error(f"Error running MATLAB: {str(e)}")
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
            
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/results/<path:filename>', methods=['GET'])
def get_result_file(filename):
    """Serve result files (like images) from the recognition results folder"""
    try:
        # Secure the filename to prevent directory traversal
        safe_filename = secure_filename(filename)
        if safe_filename != filename:
            logger.warning(f"Invalid filename attempt: {filename}")
            return jsonify({'error': 'Invalid filename'}), 400
            
        filepath = os.path.join(RECOGNITION_RESULTS_FOLDER, safe_filename)
        
        # Verify the file exists and is within the results directory
        if not os.path.exists(filepath):
            logger.warning(f"Result file not found: {filepath}")
            return jsonify({'error': 'File not found'}), 404
            
        # Verify the file is an allowed type
        if not allowed_file(filepath):
            logger.warning(f"Attempt to access disallowed file type: {filepath}")
            return jsonify({'error': 'File type not allowed'}), 400
            
        # Determine MIME type based on file extension
        mimetype = None
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.jpg', '.jpeg']:
            mimetype = 'image/jpeg'
        elif ext == '.png':
            mimetype = 'image/png'
        elif ext == '.bmp':
            mimetype = 'image/bmp'
        else:
            mimetype = 'application/octet-stream'
            
        return send_file(
            filepath,
            mimetype=mimetype,
            as_attachment=False,
            download_name=safe_filename
        )
        
    except Exception as e:
        logger.error(f"Error serving result file: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Startup checks
def perform_startup_checks():
    """Verify all required directories and permissions"""
    required_dirs = [
        RESULTS_DIR,
        UPLOADS_DIR,
        RECOGNITION_UPLOAD_FOLDER,
        RECOGNITION_RESULTS_FOLDER
    ]
    
    for directory in required_dirs:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {str(e)}")
                raise

    # Verify MATLAB is available
    try:
        matlab_version = subprocess.run(
            ['matlab', '-batch', 'disp(version()); exit;'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10
        )
        if matlab_version.returncode != 0:
            logger.error(f"MATLAB check failed: {matlab_version.stderr}")
            raise Exception("MATLAB is not properly configured")
        logger.info(f"MATLAB version: {matlab_version.stdout.strip()}")
    except Exception as e:
        logger.error(f"MATLAB check failed: {str(e)}")
        raise

# Run startup checks when the app starts
perform_startup_checks()

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000, debug=True)