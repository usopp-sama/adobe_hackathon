FROM --platform=linux/amd64 python:3.10-slim

# Set working directory
WORKDIR /app

# Copy dependencies
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python script
COPY process_pdfs.py .

# Optional: copy schema folder (only if you use it in future)
COPY sample_dataset/schema /app/schema

# Default command
CMD ["python", "process_pdfs.py"]
