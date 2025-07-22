import os
import json
import fitz  # PyMuPDF
from pathlib import Path
import re
from collections import Counter

# INPUT_DIR = Path("/app/input")
# OUTPUT_DIR = Path("/app/output")


INPUT_DIR = Path("/home/go4av05/adobe/Challenge_1a/sample_dataset/pdfs")
OUTPUT_DIR = Path("/home/go4av05/adobe/Challenge_1a/sample_dataset/outputs")


# ------------------------ Semantic Labeling ------------------------
def label_semantics(heading_text: str) -> str:
    heading = heading_text.lower()

    if any(k in heading for k in ["abstract", "summary", "overview"]):
        return "Abstract"
    if any(k in heading for k in ["introduction", "background", "scope"]):
        return "Introduction"
    if any(k in heading for k in ["method", "approach", "technique", "methodology"]):
        return "Methodology"
    if any(k in heading for k in ["result", "finding", "outcome"]):
        return "Results"
    if any(k in heading for k in ["discussion", "analysis", "interpretation"]):
        return "Discussion"
    if any(k in heading for k in ["conclusion", "summary", "future work"]):
        return "Conclusion"
    if any(k in heading for k in ["reference", "bibliography"]):
        return "References"

    return "Other"

# ------------------------ Persona Tagging ------------------------
def tag_personas(heading_text: str) -> list:
    heading = heading_text.lower()
    personas = []

    if any(k in heading for k in ["executive", "summary", "overview"]):
        personas.append("Executives")
    if any(k in heading for k in ["technical", "architecture", "implementation", "developer"]):
        personas.append("Technical Team")
    if any(k in heading for k in ["user", "guide", "manual", "instruction"]):
        personas.append("End Users")
    if not personas:
        personas.append("Everyone")
    return personas

# ------------------------ Font Size Analysis ------------------------
def get_font_size_stats(doc):
    font_sizes = []
    for page in doc:
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" in b:
                for line in b["lines"]:
                    for span in line["spans"]:
                        font_sizes.append(span["size"])
    most_common = Counter(font_sizes).most_common(1)
    return most_common[0][0] if most_common else 12

# ------------------------ Main PDF Processor ------------------------
def process_pdf(file_path: Path):
    doc = fitz.open(file_path)
    title = file_path.stem
    toc = doc.get_toc()
    outline = []

    body_font_size = get_font_size_stats(doc)
    h1_min_size = body_font_size + 2
    h2_min_size = body_font_size + 1
    h3_min_size = body_font_size  # relax threshold for wider detection

    current_section = None

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                line_text = " ".join(span["text"].strip() for span in line["spans"]).strip()
                if not line_text:
                    continue

                font_sizes = [span["size"] for span in line["spans"]]
                max_font_size = max(font_sizes)

                if max_font_size >= h1_min_size:
                    level = "H1"
                elif max_font_size >= h2_min_size:
                    level = "H2"
                elif max_font_size >= h3_min_size:
                    level = "H3"
                else:
                    level = "Paragraph"

                if level != "Paragraph" and len(line_text) >= 5 and len(line_text.split()) < 50:
                    # Save previous section if exists
                    if current_section:
                        outline.append(current_section)

                    semantic = label_semantics(line_text)
                    persona = tag_personas(line_text)

                    current_section = {
                        "level": level,
                        "text": line_text,
                        "page": page_num + 1,
                        "section_type": semantic,
                        "persona_focus": persona,
                        "paragraphs": []
                    }

                    print(f"[OUTLINE] Added heading: {line_text} | Page: {page_num+1} | Semantic: {semantic}")

                elif current_section and level == "Paragraph":
                    current_section["paragraphs"].append(line_text)

    # Append last section
    if current_section:
        outline.append(current_section)

    # Write to JSON
    output = {
        "title": title,
        "toc": toc,
        "outline": outline
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / f"{title}.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"✅ Processed: {file_path.name} → {title}.json")

# ------------------------ Entry ------------------------
def main():
    for file in INPUT_DIR.glob("*.pdf"):
        try:
            process_pdf(file)
        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

if __name__ == "__main__":
    main()
