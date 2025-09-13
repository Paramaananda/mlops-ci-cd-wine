# Base image
FROM python:3.10-slim

# Working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src
COPY main.py .

# Copy model artifact (dari workflow download)
COPY models/model.joblib ./models/model.joblib

# Expose port
EXPOSE 5000

# Start Flask API
CMD ["python", "main.py"]
