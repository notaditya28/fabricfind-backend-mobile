FROM python:3.11-slim

# Install Tesseract
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080
CMD ["sh", "-c", "gunicorn server:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120"]
