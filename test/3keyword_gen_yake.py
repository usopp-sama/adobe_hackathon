import os
import json
import fitz  # PyMuPDF
from pathlib import Path
import re
from collections import Counter
import yake  # NEW
import time
import re

# INPUT_DIR = Path("/app/input")
# OUTPUT_DIR = Path("/app/output")


INPUT_DIR = Path("Challenge_1a/sample_dataset/pdfs")
OUTPUT_DIR = Path("Challenge_1a/sample_dataset/outputs")

# ------------------------ Helper Function ------------------------
def is_heading_candidate(line_text, spans, vertical_gap, font_size_thresholds):
    text = line_text.strip()
    avg_font_size = sum(span["size"] for span in spans) / len(spans)
    is_bold = any("bold" in span["font"].lower() for span in spans)
    is_short = len(text.split()) <= 10
    is_caps = text.isupper() or text.istitle()
    is_centered = all(abs(span["bbox"][0] - span["bbox"][2]) < 400 for span in spans)

    font_based = avg_font_size >= font_size_thresholds["h1"]
    visual_clue = (is_bold and is_short and is_caps and vertical_gap > 5)

    return font_based or visual_clue

# ------------------------ Keyword Extractor ------------------------

import re

def extract_keywords_yake(text: str, max_keywords: int = 10) -> list:
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,                    # 1–3 word phrases
        top=30,                 # collect more, filter later
        dedupLim=0.9
    )

    raw_keywords = kw_extractor.extract_keywords(text)
    keywords = []

    for kw, score in raw_keywords:
        kw = kw.strip("•o•").strip().lower()

        # Remove measurement units, numbers, and fragments
        if kw in {"cup", "tablespoon", "teaspoon", "ingredient", "instructions"}:
            continue
        if len(kw) < 3 or kw.isdigit():
            continue
        if re.match(r"^(page|version|may|june|july|©|international|qualifications board)", kw):
            continue

        # Remove numeric prefixes and common prefixes like "page 10", "version 2014"
        kw = re.sub(r"^(page|version|©)?\s?\d{1,4}", '', kw).strip()

        keywords.append(kw)

    # Deduplicate by removing substrings (keep longest)
    final_keywords = []
    for kw in keywords:
        if not any(kw in other and kw != other for other in keywords):
            final_keywords.append(kw)

    return list(dict.fromkeys(final_keywords))[:max_keywords]


#-------------------------- Paragraph Cleaner ------------------------
# Clean and filter paragraph text before keywording
def clean_paragraph_lines(paragraphs: list[str]) -> list[str]:
    filtered = []
    for line in paragraphs:
        line = line.strip()

        # Skip headers/footers or noisy lines
        if not line:
            continue
        if re.match(r"^page\s?\d+", line.lower()):
            continue
        if re.match(r"^version\s?\d{4}", line.lower()):
            continue
        if "qualifications board" in line.lower():
            continue
        if "©" in line or "copyright" in line.lower():
            continue
        if line.isdigit():
            continue
        if len(line) < 4:
            continue

        filtered.append(line)
    return filtered




# ------------------------ Core Processing ------------------------
def extract_document_outline(pdf_path: Path) -> dict:
    title = ""
    outline = []
    toc = []

    try:
        doc = fitz.open(pdf_path)

        # ----------- Built-in TOC -----------
        raw_toc = doc.get_toc()
        for item in raw_toc:
            level, text, page = item
            toc.append({"level": level, "text": text.strip(), "page": page})

        # ----------- Font Size Analysis -----------
        font_sizes = []
        for page_num in range(doc.page_count):
            blocks = doc.load_page(page_num).get_text('dict')['blocks']
            for b in blocks:
                if b['type'] == 0:
                    for line in b['lines']:
                        for span in line['spans']:
                            font_sizes.append(round(span['size'], 1))

        if not font_sizes:
            return {"title": "No Title Found", "outline": [], "toc": toc}

        font_size_counts = Counter(font_sizes)
        body_font_size = font_size_counts.most_common(1)[0][0]
        font_thresholds = {
            "h1": body_font_size + 3,
            "h2": body_font_size + 2,
            "h3": body_font_size + 1
        }

        # ----------- Title Extraction -----------
        if doc.page_count > 0:
            first_page = doc.load_page(0)
            first_blocks = first_page.get_text('dict')['blocks']
            potential_titles = []
            for b in first_blocks:
                if b['type'] == 0:
                    for line in b['lines']:
                        for span in line['spans']:
                            if span['text'].strip() and span['size'] > (body_font_size + 2):
                                potential_titles.append((span['size'], span['text'].strip()))
            if potential_titles:
                potential_titles.sort(key=lambda x: (-x[0], x[1]))
                title = potential_titles[0][1]

        # ----------- Heading & Paragraph Extraction -----------
        current_section = None
        prev_y = None

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text('dict')['blocks']

            for b in blocks:
                if b['type'] != 0:
                    continue

                for line in b['lines']:
                    if not line['spans']:
                        continue

                    line_text = " ".join([span['text'] for span in line['spans']]).strip()
                    if not line_text:
                        continue

                    line_y = line['spans'][0]['bbox'][1]
                    vertical_gap = line_y - prev_y if prev_y is not None else 0
                    prev_y = line_y

                    if is_heading_candidate(line_text, line['spans'], vertical_gap, font_thresholds):
                        if current_section and current_section["paragraphs"]:
                            # Add keywords before appending
                            cleaned_paragraphs = clean_paragraph_lines(current_section["paragraphs"])
                            full_text = " ".join(current_section["paragraphs"])
                            current_section["keywords"] = extract_keywords_yake(full_text)

                        current_section = {
                            "level": "H1",
                            "text": line_text,
                            "page": page_num + 1,
                            "paragraphs": [],
                            "keywords": []
                        }
                        outline.append(current_section)
                    elif current_section:
                        current_section["paragraphs"].append(line_text)

        # Final section keyword generation (if not yet done)
        if current_section and current_section["paragraphs"] and not current_section["keywords"]:
            full_text = " ".join(current_section["paragraphs"])
            current_section["keywords"] = extract_keywords_yake(full_text)

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        title = f"Error processing {pdf_path.name}"
        outline = []

    finally:
        doc.close()

    if not title:
        title = pdf_path.stem.replace("_", " ").title()

    return {
        "title": title,
        "toc": toc,
        "outline": outline
    }

# ------------------------ Batch Processing ------------------------
def process_pdfs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_files = list(INPUT_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {INPUT_DIR}")
        return

    for pdf_file in pdf_files:
        print(f"\n⏳ Processing {pdf_file.name}...")
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
