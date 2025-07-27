import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
from pdf_processor_pipeline import extract_document_outline  # 👈 from your modular pipeline

# ------------------------ Hardcoded Paths ------------------------
INPUT_PATH = Path("/home/go4av05/adobe/Challenge_1b/Collection 2/challenge1b_input.json")
OUTPUT_PATH = Path("/home/go4av05/adobe/Challenge_1b/Collection 2/challenge1b_output_me.json")
PDF_DIR = Path("/home/go4av05/adobe/Challenge_1b/Collection 2/PDFs")  # <-- point to raw PDF files

# ------------------------ Core Functions ------------------------

def load_input(input_path: Path):
    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_and_parse_pdf(pdf_dir: Path, filename: str):
    path = pdf_dir / filename
    if not path.exists():
        print(f"⚠️ Missing PDF file: {filename}")
        return None
    try:
        return extract_document_outline(path)
    except Exception as e:
        print(f"❌ Error parsing {filename}: {e}")
        return None

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
    if not chunks:
        return []

    task_embedding = model.encode(task, convert_to_tensor=True)
    texts = [chunk["text"] for chunk in chunks]
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
            "semantic_summary": ""  # placeholder
        })

    return results

# ------------------------ Main ------------------------
def main():
    print("🚀 Starting semantic matcher...")

    input_data = load_input(INPUT_PATH)
    task = input_data["job_to_be_done"]["task"]
    file_names = [doc["filename"] for doc in input_data["documents"]]

    print(f"📌 Task: {task}")
    print(f"📁 PDFs: {file_names}")

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    all_chunks = []
    for f in file_names:
        print(f"📄 Parsing {f}")
        data = load_and_parse_pdf(PDF_DIR, f)
        if data:
            chunks = collect_chunks(data, f)
            all_chunks.extend(chunks)

    print(f"🔍 Matching from {len(all_chunks)} extracted sections...")
    top_matches = find_matches(task, all_chunks, model, top_k=10)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"task": task, "matches": top_matches}, f, indent=2)

    print(f"✅ Done! Output saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
