import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import time
from pdf_processor_pipeline import extract_document_outline  # Assuming your processing function is here

# ------------------------ Hardcoded Paths ------------------------
INPUT_PATH = Path("/home/go4av05/adobe/Challenge_1b/Collection 2/challenge1b_input.json")
OUTPUT_PATH = Path("/home/go4av05/adobe/Challenge_1b/Collection 2/challenge1b_output_me.json")
PDF_JSON_DIR = Path("/home/go4av05/adobe/Challenge_1b/Collection 2/json_output")  # New subfolder for JSONs
PDF_JSON_DIR.mkdir(exist_ok=True)
PDF_DIR = Path("/home/go4av05/adobe/Challenge_1b/Collection 2/PDFs")  # <-- point to raw PDF files


# ------------------------ Core Functions ------------------------

def load_input(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_pdf_json(pdf_json_dir: Path, filename: str):
    path = pdf_json_dir / filename
    if not path.exists():
        print(f"⚠️ Missing PDF JSON file: {filename}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def collect_chunks(pdf_data, file_name):
    chunks = []
    if not pdf_data or "outline" not in pdf_data:
        return chunks

    for section in pdf_data["outline"]:
        heading = section.get("text", "")
        page = section.get("page", 0)
        paragraphs = section.get("paragraphs", [])
        text = " ".join(paragraphs).strip()

        if len(text) < 10:
            continue

        chunks.append({
            "file": file_name,
            "page": page,
            "heading": heading,
            "text": text,
            "keywords": section.get("keywords", [])
        })

    return chunks

def find_matches(task: str, chunks, model, top_k=10):
    print("🧠 Generating embeddings and running semantic similarity...")
    t0 = time.time()

    task_embedding = model.encode(task, convert_to_tensor=True)
    texts = [chunk["text"] for chunk in chunks]
    if not texts:
        return []

    text_embeddings = model.encode(texts, convert_to_tensor=True)
    scores = util.cos_sim(task_embedding, text_embeddings)[0]
    top_results = torch.topk(scores, k=min(top_k, len(chunks)))

    results = []
    for score, idx in zip(top_results[0], top_results[1]):
        chunk = chunks[idx]
        results.append({
            "pdf_name": chunk["file"],
            "page": chunk["page"],
            "section_heading": chunk["heading"],
            "matched_content": chunk["text"],
            "keywords": chunk["keywords"],
            "score": float(score),
            "semantic_summary": ""  # Placeholder for future use
        })

    print(f"✅ Embedding + similarity computation time: {time.time() - t0:.2f} sec")
    return results

# ------------------------ Main ------------------------
def main():
    t_start = time.time()
    print("🚀 Starting semantic matcher...")

    input_data = load_input(INPUT_PATH)
    task = input_data["job_to_be_done"]["task"]
    pdf_files = [doc["filename"] for doc in input_data["documents"]]
    file_names = [f.replace(".pdf", ".json") for f in pdf_files]

    print(f"📌 Task: {task}")
    print(f"📁 PDFs: {pdf_files}")

    # Step 1: Run PDF Processing
    print("🛠️  Extracting document outlines from PDFs...")
    t0 = time.time()
    for pdf_file in pdf_files:
        pdf_path = PDF_DIR / pdf_file  # ✅ Corrected
        json_path = PDF_JSON_DIR / pdf_file.replace(".pdf", ".json")

        if not pdf_path.exists():
            print(f"⚠️ Skipping missing PDF file: {pdf_file}")
            continue

        print(f"📄 Processing {pdf_file}")
        outline_data = extract_document_outline(pdf_path)
        
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(outline_data, jf, indent=2)


    print(f"✅ PDF processing complete in {time.time() - t0:.2f} sec")

    # Step 2: Load SentenceTransformer model
    t1 = time.time()
    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    print(f"📦 Model loaded in {time.time() - t1:.2f} sec")

    # Step 3: Load processed chunks
    all_chunks = []
    t2 = time.time()
    for f in file_names:
        print(f"📄 Parsing {f}")
        data = load_pdf_json(PDF_JSON_DIR, f)
        if data:
            chunks = collect_chunks(data, f)
            all_chunks.extend(chunks)

    print(f"✅ Total parsing time: {time.time() - t2:.2f} sec")
    print(f"🔍 Matching from {len(all_chunks)} extracted sections...")

    # Step 4: Semantic search
    top_matches = find_matches(task, all_chunks, model, top_k=10)

    # Step 5: Write output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"task": task, "matches": top_matches}, f, indent=2)

    print(f"✅ Output saved to {OUTPUT_PATH}")
    print(f"⏱️  Total time: {time.time() - t_start:.2f} sec")

if __name__ == "__main__":
    main()
