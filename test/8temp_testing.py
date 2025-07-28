import fitz  # PyMuPDF
import json
import re
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer, util
import argparse

# Load semantic model
semantic_model = SentenceTransformer("all-MiniLM-L6-v2")

# -------- Text & Font Extraction --------
def extract_text_with_fonts(pdf_path):
    doc = fitz.open(pdf_path)
    blocks = []
    for page_num, page in enumerate(doc):
        text_blocks = page.get_text("dict").get("blocks", [])
        for block in text_blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                text = ""
                sizes = []
                font_names = []
                flags = []
                positions = []
                for span in line["spans"]:
                    span_text = span["text"].strip()
                    if span_text:
                        text += span_text + " "
                        sizes.append(span["size"])
                        font_names.append(span["font"])
                        flags.append(span["flags"])
                        positions.append(span["bbox"])
                if text.strip():
                    blocks.append({
                        "text": text.strip(),
                        "size": round(sum(sizes) / len(sizes), 2) if sizes else 0,
                        "font": font_names[0] if font_names else "",
                        "flags": flags[0] if flags else 0,
                        "bbox": positions[0] if positions else (0, 0, 0, 0),
                        "page": page_num + 1
                    })
    return blocks

# -------- Heuristics --------
def is_noise_heading(text):
    text = text.strip().lower()
    return (
        re.match(r"^page\s+\d+\s+of\s+\d+$", text) or
        re.match(r"^version\s+\d+(\.\d+)*$", text) or
        re.match(r"^\d{1,2}\s+[a-zA-Z]{3,9}\s+\d{4}$", text)
    )

def is_bullet_point(text):
    return bool(re.match(r"^([\u2022\-\*\d+\.]\s+).+", text.strip()))

def is_valid_heading(text):
    text = text.strip()
    if not text or is_bullet_point(text) or is_noise_heading(text):
        return False
    lower = text.lower()
    return (
        2 <= len(text.split()) <= 25 and
        not re.match(r"^[\d\W_]+$", text) and
        not text.endswith((':', ';', ',', '.', '...')) and
        not text.islower() and
        not lower.startswith("from:") and
        not lower.startswith("to:")
    ) or re.match(r"^(unit|chapter|module)\s+[\divx]+", lower)

# -------- Heading Detection --------
def classify_headings(blocks, pdf_path=""):
    if not blocks:
        return {"title": "", "outline": []}

    sizes = sorted({b["size"] for b in blocks}, reverse=True)
    size_to_level = {}
    for i, size in enumerate(sizes[:4]):
        size_to_level[size] = ["title", "H1", "H2", "H3"][i]

    outline = []
    seen = set()
    title_block = None

    for block in blocks:
        raw_text = block["text"].strip()
        if not raw_text or is_noise_heading(raw_text):
            continue

        size = block["size"]
        font_name = block.get("font", "").lower()
        flags = block.get("flags", 0)
        bbox = block.get("bbox", (0, 0, 0, 0))
        x0 = bbox[0]
        page = block["page"]
        level = size_to_level.get(size, "H3")

        is_bold = flags & 2 != 0
        is_centered = 200 < x0 < 400
        is_styled = is_bold or "bold" in font_name or is_centered

        key = (raw_text.lower(), page)
        if key in seen:
            continue
        seen.add(key)

        if not title_block and level == "title" and is_valid_heading(raw_text) and is_styled:
            title_block = raw_text
            continue

        if is_valid_heading(raw_text) and is_styled:
            outline.append({
                "level": level,
                "text": raw_text,
                "page": page,
                "index": blocks.index(block)
            })

    full_title = title_block or Path(pdf_path).stem
    return {
        "title": full_title.strip(),
        "outline": outline
    }

# -------- Semantic Ranking --------
def rank_headings(outline, persona, job, top_k=5):
    query = f"{persona}. Task: {job}"
    heading_texts = [item["text"] for item in outline]
    heading_embeddings = semantic_model.encode(heading_texts, convert_to_tensor=True)
    query_embedding = semantic_model.encode(query, convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, heading_embeddings, top_k=top_k)[0]
    ranked = []
    for hit in hits:
        index = hit["corpus_id"]
        item = outline[index]
        item["score"] = round(float(hit["score"]), 4)
        ranked.append(item)
    return ranked

# -------- Main Script --------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True, help="Path to collection folder (e.g., Collection 1)")
    parser.add_argument("--topk", type=int, default=5, help="Number of top headings to return")
    args = parser.parse_args()

    collection_path = Path(args.collection)
    input_json = collection_path / "challenge1b_input.json"
    pdf_dir = collection_path / "PDFs"
    output_path = collection_path / "challenge1b_output.json"

    with open(input_json, encoding="utf-8") as f:
        input_data = json.load(f)

    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    documents = [doc["filename"] for doc in input_data["documents"]]

    print("🛠️ Starting heading extraction and ranking...")
    start_time = time.time()

    all_headings = []
    block_map = {}  # Store blocks for refined text

    for doc_name in documents:
        pdf_path = pdf_dir / doc_name
        blocks = extract_text_with_fonts(str(pdf_path))
        doc_outline = classify_headings(blocks, str(pdf_path))
        block_map[doc_name] = blocks

        for item in doc_outline["outline"]:
            item["document"] = doc_name
        all_headings.extend(doc_outline["outline"])

    # Ranking
    ranked = rank_headings(all_headings, persona, job, top_k=args.topk)

    extracted_sections = []
    subsection_analysis = []

    for i, item in enumerate(ranked):
        doc_name = item["document"]
        blocks = block_map[doc_name]
        idx = item.get("index", 0)

        refined = ""
        for b in blocks[idx + 1:]:
            if b["page"] != item["page"]:
                break
            if is_valid_heading(b["text"]):
                break
            refined += b["text"] + " "
        refined = refined.strip()

        extracted_sections.append({
            "document": doc_name,
            "section_title": item["text"],
            "importance_rank": i + 1,
            "page_number": item["page"]
        })
        subsection_analysis.append({
            "document": doc_name,
            "refined_text": refined or item["text"],
            "page_number": item["page"]
        })

    result = {
        "metadata": {
            "input_documents": documents,
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - start_time
    print(f"✅ Completed in {elapsed:.2f} seconds. Output saved to: {output_path}")