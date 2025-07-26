import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import torch
import argparse

# ------------------------ Argument Parsing ------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Semantic Matcher for Challenge 1B")
    parser.add_argument(
        "--input",
        type=str,
        default="/app/input/challenge1b_input.json",
        help="Path to input JSON task file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/app/output/challenge1b_output.json",
        help="Path to write output JSON results"
    )
    parser.add_argument(
        "--pdf_dir",
        type=str,
        default="/app/input",
        help="Directory where extracted PDF .json files are located"
    )
    return parser.parse_args()

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
            "semantic_summary": ""  # Placeholder for optional LLM summary
        })

    return results

# ------------------------ Main ------------------------
def main():
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    pdf_dir = Path(args.pdf_dir)

    print("🚀 Starting semantic matcher...")
    input_data = load_input(input_path)
    task = input_data["task"]
    file_names = input_data["pdfs"]

    print(f"📌 Task: {task}")
    print(f"📁 PDFs: {file_names}")

    model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

    all_chunks = []
    for f in file_names:
        print(f"📄 Loading {f}")
        data = load_pdf_json(pdf_dir, f)
        if data:
            chunks = collect_chunks(data, f)
            all_chunks.extend(chunks)

    print(f"🔍 Matching from {len(all_chunks)} extracted sections...")
    top_matches = find_matches(task, all_chunks, model, top_k=10)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"task": task, "matches": top_matches}, f, indent=2)

    print(f"✅ Done! Output saved to {output_path}")

if __name__ == "__main__":
    main()
