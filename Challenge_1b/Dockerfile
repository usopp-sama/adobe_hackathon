FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy and install only necessary dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Download small spaCy model
RUN python -m spacy download en_core_web_sm

# Copy source code
COPY semantic_matcher.py .
COPY pdf_processor_pipeline.py .
COPY nlp_utils.py .

# Copy input collections (optional: could mount instead during runtime)
COPY collections /app/collections

# Create output directory
RUN mkdir -p /app/outputs

# Default command
CMD ["python", "semantic_matcher.py"]
