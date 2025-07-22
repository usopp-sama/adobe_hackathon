FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model only (no NLTK since it's unused)
RUN python -m spacy download en_core_web_sm

# Copy application files
COPY process_pdfs.py .
COPY sample_dataset/schema /app/schema

# Set default command
CMD ["python", "process_pdfs.py"]
