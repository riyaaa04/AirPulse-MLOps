FROM python:3.9-slim

WORKDIR /app

# Install system build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY dataset/ dataset/
COPY src/ src/
COPY app/ app/
COPY models/ models/
COPY data/ data/
COPY streamlit_app.py .
COPY entrypoint.sh .

EXPOSE 8000 8501

ENTRYPOINT ["./entrypoint.sh"]
