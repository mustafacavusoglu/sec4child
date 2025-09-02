FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "from optimum.pipelines import pipeline as ort_pipe; pipe = ort_pipe('text-classification', 'cardiffnlp/twitter-roberta-base-sentiment-latest')"

COPY app.py .

ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port $PORT"]
