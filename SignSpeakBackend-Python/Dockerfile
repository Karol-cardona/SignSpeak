# Use a lightweight Python version
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for OpenCV (cv2)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a directory for HuggingFace cache to avoid permission errors
RUN mkdir -p /app/hf_cache && chmod 777 /app/hf_cache

# Expose the port the API runs on
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]