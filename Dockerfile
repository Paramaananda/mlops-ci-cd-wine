FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY main.py .
COPY app.py .
COPY best_model.joblib ./best_model.joblib
COPY scaler.joblib ./scaler.joblib

EXPOSE 5000

CMD ["python", "app.py"]
