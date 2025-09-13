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
COPY best_model.joblib ./best_model.joblib  

# di-copy dari artifact pipeline

# Expose port
EXPOSE 5000

# Start Flask API (kalau ada API di main.py)
CMD ["python", "main.py"]
