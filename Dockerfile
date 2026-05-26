# Use official lightweight Python image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set working directory
WORKDIR /app

# Install standard system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application files (including serialized pipeline joblib)
COPY . .

# Hugging Face Spaces expects traffic on port 7860 by default
EXPOSE 7860

# Run Flask application with Gunicorn production server
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
