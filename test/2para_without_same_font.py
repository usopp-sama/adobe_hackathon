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
        return "Summary"
    elif any(k in heading for k in ["intro", "welcome", "start here"]):
        return "Introduction"
    elif any(k in heading for k in ["background", "context", "history"]):
        return "Background"
    elif any(k in heading for k in ["method", "approach", "process", "workflow", "procedure"]):
        return "Methodology"
    elif any(k in heading for k in ["data", "dataset", "statistics"]):
        return "Data"
    elif any(k in heading for k in ["experiment", "implementation", "setup", "execution"]):
        return "Execution"
    elif any(k in heading for k in ["result", "output", "analysis", "evaluation", "findings"]):
        return "Results"
    elif any(k in heading for k in ["discussion", "conclusion", "takeaways", "closing"]):
        return "Conclusion"
    elif any(k in heading for k in ["faq", "help", "support", "guidelines"]):
        return "Help"
    elif any(k in heading for k in ["contact", "team", "about us", "reach us"]):
        return "About / Contact"
    elif any(k in heading for k in ["references", "bibliograph"]):
        return "References"
    elif any(k in heading for k in ["appendix", "extra", "annexure", "attachment"]):
        return "Appendix"
    elif any(k in heading for k in ["invitation", "you are invited", "event"]):
        return "Invitation"
    elif any(k in heading for k in ["agenda", "schedule", "itinerary", "plan"]):
        return "Schedule"
    elif any(k in heading for k in ["announcement", "notice", "circular"]):
        return "Announcement"
    elif any(k in heading for k in ["report", "findings", "summary report"]):
        return "Report"
    else:
        return "Unknown"

def tag_persona_focus(semantic_label: str) -> list:
    mapping = {
        "Introduction": ["Everyone"],
        "Background": ["Researcher", "Student", "Professional"],
        "Summary": ["Everyone"],
        "Methodology": ["Researcher", "Engineer", "Developer"],
        "Data": ["Analyst", "Scientist"],
        "Execution": ["Engineer", "Developer", "Event Planner"],
        "Results": ["Analyst", "Manager"],
        "Conclusion": ["Manager", "Student", "Researcher"],
        "Help": ["Everyone"],
        "About / Contact": ["Everyone"],
        "References": ["Researcher", "Student"],
        "Appendix": ["Researcher", "Admin"],
        "Invitation": ["Event Planner", "Guest"],
        "Schedule": ["Traveler", "Event Planner"],
        "Announcement": ["Everyone"],
        "Report": ["Manager", "Analyst"],
        "Unknown": []
    }
    return mapping.get(semantic_label, [])

# ------------------------ Helper Function ------------------------
def is_heading_candidate(line_text, spans, vertical_gap, font_size_thresholds):
    text = line_text.strip()
    avg_font_size = sum(span["size"] for span in spans) / len(spans)
    is_bold = any("bold" in span["font"].lower() for span in spans)
    is_short = len(text.split()) <= 10
    is_caps = text.isupper() or text.istitle()
    is_centered = all(abs(span["bbox"][0] - span["bbox"][2]) < 400 for span in spans)  # crude center estimate

    font_based = avg_font_size >= font_size_thresholds["h1"]
    visual_clue = (is_bold and is_short and is_caps and vertical_gap > 5)

    return font_based or visual_clue

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
        body_font_size = font_size_counts.most_common()[0][0]
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
                        semantic = label_semantics(line_text)
                        persona = tag_persona_focus(semantic)

                        current_section = {
                            "level": "H1",
                            "text": line_text,
                            "page": page_num + 1,
                            "section_type": semantic,
                            "persona_focus": persona,
                            "paragraphs": []
                        }
                        outline.append(current_section)
                    elif current_section:
                        current_section["paragraphs"].append(line_text)

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
        print(f"Processing {pdf_file.name}...")
        extracted_data = extract_document_outline(pdf_file)

        output_file = OUTPUT_DIR / f"{pdf_file.stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2)
        print(f"Generated {output_file.name}")

# ------------------------ Entry Point ------------------------
if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing pdfs")
