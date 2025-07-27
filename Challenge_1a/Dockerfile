# Use a slim, minimal base image
FROM --platform=linux/amd64 python:3.10-slim


# Set working directory
WORKDIR /app

# Install system dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy only the needed application files
COPY process_pdfs.py .
COPY pdf_processor_pipeline.py .
COPY nlp_utils.py .
COPY sample_dataset /app/sample_dataset

# Set default command
CMD ["python", "process_pdfs.py"]