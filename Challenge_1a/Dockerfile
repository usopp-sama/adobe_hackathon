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



# FROM --platform=linux/amd64 python:3.10-slim

# # Set working directory
# WORKDIR /app

# # Install system dependencies
# RUN apt-get update && apt-get install -y build-essential

# # Install Python dependencies
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # ✅ Download spaCy model
# RUN python -m spacy download en_core_web_sm

# # ✅ Download sentence-transformers model into cache
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-MiniLM-L6-v2')"

# # Copy application code
# COPY . .

# # Set environment variable to ensure offline use (optional)
# ENV TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

# # Default command
# CMD ["python", "process_pdfs.py"]
