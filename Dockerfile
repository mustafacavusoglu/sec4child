FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Cloud Run PORT env variable kullan
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
