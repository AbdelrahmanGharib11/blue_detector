# Base image
FROM python:3.9-slim

# Install system deps + MATLAB Runtime (R2023a)
RUN apt-get update && apt-get install -y \
    wget \
    unzip \
    libxt6 \
    libxmu6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Download and install MATLAB Runtime (~2GB)
RUN wget -q https://ssd.mathworks.com/supportfiles/downloads/R2023a/Release/5/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2023a_Update_5_glnxa64.zip \
    && unzip MATLAB_Runtime_R2023a_Update_5_glnxa64.zip -d /tmp/matlab \
    && /tmp/matlab/install -mode silent -agreeToLicense yes \
    && rm -rf /tmp/matlab MATLAB_Runtime_R2023a_Update_5_glnxa64.zip

# Set MATLAB Runtime paths
ENV LD_LIBRARY_PATH=/usr/local/MATLAB/MATLAB_Runtime/v910/runtime/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v910/bin/glnxa64:/usr/local/MATLAB/MATLAB_Runtime/v910/sys/os/glnxa64
ENV XAPPLRESDIR=/usr/local/MATLAB/MATLAB_Runtime/v910/X11/app-defaults

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Run Flask app
CMD ["gunicorn", "--bind", "0.0.0.0:$PORT", "app:app"]