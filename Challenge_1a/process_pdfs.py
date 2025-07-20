import os
import json
import fitz  # PyMuPDF
from pathlib import Path
import re
from collections import Counter

INPUT_DIR = Path("/app/input")
OUTPUT_DIR = Path("/app/output")

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

# ------------------------ Core Processing ------------------------
def extract_document_outline(pdf_path: Path) -> dict:
    title = ""
    outline = []

    try:
        doc = fitz.open(pdf_path)

        # Step 1: Extract font sizes
        font_sizes = []
        for page_num in range(doc.page_count):
            blocks = doc.load_page(page_num).get_text('dict')['blocks']
            for b in blocks:
                if b['type'] == 0:
                    for line in b['lines']:
                        for span in line['spans']:
                            font_sizes.append(round(span['size'], 1))

        if not font_sizes:
            return {"title": "No Title Found", "outline": []}

        font_size_counts = Counter(font_sizes)
        unique_sorted_font_sizes = sorted(font_size_counts.keys(), reverse=True)

        body_font_size = unique_sorted_font_sizes[-1] if unique_sorted_font_sizes else 10
        sorted_by_freq = font_size_counts.most_common()
        if sorted_by_freq:
            body_candidate = sorted_by_freq[0][0]
            if body_candidate > 16 and len(unique_sorted_font_sizes) > 1:
                body_font_size = sorted_by_freq[1][0]
            else:
                body_font_size = body_candidate

        # Thresholds
        h1_min_size = body_font_size * 1.6
        h2_min_size = body_font_size * 1.3
        h3_min_size = body_font_size * 1.1

        if len(unique_sorted_font_sizes) < 3:
            h1_min_size = body_font_size + 6
            h2_min_size = body_font_size + 4
            h3_min_size = body_font_size + 2

        if h2_min_size >= h1_min_size: h2_min_size = h1_min_size - 1
        if h3_min_size >= h2_min_size: h3_min_size = h2_min_size - 1
        if h3_min_size < body_font_size: h3_min_size = body_font_size + 0.5

        # Step 2: Extract title (first page only)
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
                if len(title.split()) < 3 and len(potential_titles) > 1:
                    if potential_titles[0][0] == potential_titles[1][0] and len(potential_titles[1][1].split()) > 3:
                        title = potential_titles[1][1]

        # Step 3: Extract headings from all pages
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

                    avg_line_font_size = sum(span['size'] for span in line['spans']) / len(line['spans'])

                    level = None
                    if avg_line_font_size >= h1_min_size:
                        level = "H1"
                    elif avg_line_font_size >= h2_min_size:
                        level = "H2"
                    elif avg_line_font_size >= h3_min_size:
                        level = "H3"
                    elif avg_line_font_size >= body_font_size + 1:
                        level = "H3"  # fallback level for large text

                    is_likely_heading = False
                    if level and len(line_text.split()) < 30:
                        is_likely_heading = True
                        if re.match(r'^\d+(\.\d+)*\s+', line_text) or \
                           re.match(r'^[A-Z]\.\s+', line_text) or \
                           re.match(r'^[IVX]+\.\s+', line_text, re.IGNORECASE):
                            is_likely_heading = True

                    if is_likely_heading:
                        semantic = label_semantics(line_text)
                        persona = tag_persona_focus(semantic)
                        if not outline or outline[-1]['text'].lower() != line_text.lower():
                            outline.append({
                                "level": level,
                                "text": line_text,
                                "page": page_num + 1,
                                "section_type": semantic,
                                "persona_focus": persona
                            })

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

if __name__ == "__main__":
    print("Starting processing pdfs")
    process_pdfs()
    print("Completed processing pdfs")
