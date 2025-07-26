# --- Modified version with nlp_utils integrated ---

import os
import json
import fitz  # PyMuPDF
from pathlib import Path
import time
from Challenge_1a.pdf_processor_pipeline import extract_document_outline

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

# INPUT_DIR = Path("Challenge_1a/sample_dataset/pdfs")
# OUTPUT_DIR = Path("Challenge_1a/sample_dataset/outputs")


# ------------------------ Batch Processing ------------------------
def process_pdfs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    for pdf_file in pdf_files:
        print(f"\n\u23F3 Processing {pdf_file.name}...")
        start_time = time.time()

        extracted_data = extract_document_outline(pdf_file)

        end_time = time.time()
        elapsed = end_time - start_time

        output_file = OUTPUT_DIR / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2)

        print(f"✅ Done: {output_file.name} (Processed in {elapsed:.2f} seconds)")

# ------------------------ Entry Point ------------------------
if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing pdfs")
