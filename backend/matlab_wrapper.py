import subprocess
import os

def run_matlab_detection(image_path):
    abs_path = os.path.abspath(image_path)
    matlab_script = f"""
    addpath('{os.path.dirname(abs_path)}');
    detect2('{abs_path}');
    exit;
    """
    try:
        subprocess.run(
            [
                "/usr/local/MATLAB/MATLAB_Runtime/v910/bin/matlab",
                "-batch",
                matlab_script
            ],
            check=True,
            timeout=30
        )
        output_path = abs_path.replace('.', '_output.')
        return output_path if os.path.exists(output_path) else None
    except Exception as e:
        print(f"MATLAB Runtime error: {e}")
        return None
    