# Use official Python base image
FROM python:3.9-slim

# Set work directory
WORKDIR /app

# Copy files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY entrez_submission_api.py .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "entrez_submission_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
