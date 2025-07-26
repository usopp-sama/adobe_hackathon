FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm
#RUN python -m nltk.downloader punkt

# Copy application files
COPY process_pdfs.py .
COPY nlp_utils.py .
COPY __init__.py .
COPY semantic_matcher.py .


COPY sample_dataset /app/sample_dataset

# Set default command
CMD ["python", "process_pdfs.py"]

