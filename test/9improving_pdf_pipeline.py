from pathlib import Path
import fitz
import re
from collections import Counter
import yake
import unicodedata

from nlp_utils import clean_text, analyze_text, get_sentences

# ------------------------ Helper: Improved Heading Detector ------------------------
def is_heading_candidate(line_text, spans, vertical_gap, font_size_thresholds, next_line_indent=False):
    text = line_text.strip()
    avg_font_size = sum(span["size"] for span in spans) / len(spans)
    is_bold = any("bold" in span["font"].lower() for span in spans)
    is_italic = any("italic" in span["font"].lower() for span in spans)
    is_underlined = any(span.get("flags", 0) & 4 for span in spans)

    is_short = len(text.split()) <= 10
    is_caps = text.isupper() or text.istitle()

    font_based = avg_font_size >= font_size_thresholds["h1"]
    visual_clue = (is_bold and is_short and is_caps and vertical_gap > 5)
    indent_clue = next_line_indent

    # NEW: Force heading if underline + (bold or italic)
    force_heading = is_underlined and (is_bold or is_italic)

    return font_based or visual_clue or indent_clue or force_heading

# ------------------------ Improved Heading Level Mapper ------------------------
def get_heading_level(avg_font_size, font_size_thresholds, spans, body_font_size):
    if avg_font_size >= font_size_thresholds["h1"]:
        level = "H1"
    elif avg_font_size >= font_size_thresholds["h2"] and avg_font_size > body_font_size:
        level = "H2"
    elif avg_font_size >= font_size_thresholds["h3"] and avg_font_size > body_font_size:
        level = "H3"
    else:
        level = "H3"

    is_bold = any("bold" in span["font"].lower() for span in spans)
    is_italic = any("italic" in span["font"].lower() for span in spans)
    is_underlined = any(span.get("flags", 0) & 4 for span in spans)

    # Only underline + (bold or italic) at body font size forces H2
    if round(avg_font_size, 1) == round(body_font_size, 1):
        if is_underlined and (is_bold or is_italic):
            level = "H2"

    return level

# ------------------------ Keyword Extractor ------------------------
def extract_keywords_yake(text: str, max_keywords: int = 10) -> list:
    kw_extractor = yake.KeywordExtractor(
        lan="en",
        n=3,
        top=30,
        dedupLim=0.9
    )

    raw_keywords = kw_extractor.extract_keywords(text)
    keywords = []

    for kw, score in raw_keywords:
        kw = kw.strip("\u2022o•").strip().lower()
        if kw in {"cup", "tablespoon", "teaspoon", "ingredient", "instructions"}:
            continue
        if len(kw) < 3 or kw.isdigit():
            continue
        if re.match(r"^(page|version|may|june|july|©|international|qualifications board)", kw):
            continue
        kw = re.sub(r"^(page|version|©)?\s?\d{1,4}", '', kw).strip()
        keywords.append(kw)

    final_keywords = []
    for kw in keywords:
        if not any(kw in other and kw != other for other in keywords):
            final_keywords.append(kw)

    return list(dict.fromkeys(final_keywords))[:max_keywords]

# ------------------------ Paragraph Cleaner ------------------------
def clean_paragraph_lines(paragraphs: list[str]) -> list[str]:
    filtered = []
    for line in paragraphs:
        line = line.strip()
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


# ------------------------ Unicode Cleaner ------------------------
def clean_text(text):
    if not text:
        return ""
    # Replace common artifacts with proper equivalents
    replacements = {
        "\u2022": "-",  # bullet
        "\u2019": "'",  # smart single quote
        "\u201c": '"',  # smart left double quote
        "\u201d": '"',  # smart right double quote
        "\u2014": "-",  # em dash
        "\u2013": "-",  # en dash
        "\ufb00": "ff", # ff ligature
        "\ufb01": "fi", # fi ligature
        "\u2018": "'",  # left single quote
        "\u2026": "...", # ellipsis
        "\u00a0": " ",  # non-breaking space
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Normalize other Unicode characters
    text = unicodedata.normalize('NFKD', text)
    # Remove remaining non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text.strip()

def clean_section_data(section):
    """Clean all text fields in a section dictionary"""
    section["text"] = clean_text(section["text"])
    section["paragraphs"] = [clean_text(p) for p in section["paragraphs"]]
    section["keywords"] = [clean_text(k) for k in section["keywords"]]
    section["sentences"] = [clean_text(s) for s in section["sentences"]]
    
    if "semantic" in section:
        semantic = section["semantic"]
        for key in ["tokens", "nouns", "verbs", "lemmas"]:
            if key in semantic:
                semantic[key] = [clean_text(t) for t in semantic[key]]
    
    return section


# ------------------------ Core Processing ------------------------
from pathlib import Path
import fitz
import re
import yake
from collections import Counter
from nlp_utils import clean_text, analyze_text, get_sentences

def extract_document_outline(pdf_path: Path) -> dict:
    def is_valid_heading(text):
        text = text.strip()
        if not text or len(text.split()) > 25:
            return False
        if re.match(r"^[\d\W_]+$", text):
            return False
        if text.endswith((':', ';', ',', '.', '...')) or text.islower():
            return False
        if text.lower().startswith(("from:", "to:")):
            return False
        return True

    def extract_keywords_yake(text, max_keywords=10):
        extractor = yake.KeywordExtractor(lan="en", n=3, top=30, dedupLim=0.9)
        keywords = extractor.extract_keywords(text)
        return list(dict.fromkeys([kw.strip().lower() for kw, _ in keywords if len(kw.strip()) > 2]))[:max_keywords]

    def clean_paragraph_lines(lines):
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line or len(line) < 4:
                continue
            if any(skip in line.lower() for skip in ["copyright", "©", "qualifications board"]):
                continue
            if re.match(r"^page\s?\d+", line.lower()):
                continue
            cleaned.append(line)
        return cleaned

    try:
        doc = fitz.open(pdf_path)
        all_spans = []
        for page_num in range(doc.page_count):
            blocks = doc.load_page(page_num).get_text("dict")["blocks"]
            for block in blocks:
                if "lines" not in block:
                    continue
                for line in block["lines"]:
                    text = " ".join(span["text"].strip() for span in line["spans"] if span["text"].strip())
                    if not text:
                        continue
                    sizes = [span["size"] for span in line["spans"]]
                    fonts = [span["font"] for span in line["spans"]]
                    flags = [span["flags"] for span in line["spans"]]
                    bbox = line["spans"][0]["bbox"]
                    all_spans.append({
                        "text": text.strip(),
                        "size": sum(sizes) / len(sizes),
                        "font": fonts[0],
                        "flags": flags[0],
                        "bbox": bbox,
                        "page": page_num + 1
                    })

        if not all_spans:
            return {"title": "No Title Found", "toc": [], "outline": []}

        sizes = sorted({round(s["size"], 1) for s in all_spans}, reverse=True)
        size_to_level = {}
        for i, size in enumerate(sizes[:4]):
            size_to_level[size] = ["title", "H1", "H2", "H3"][i]

        outline = []
        title = ""
        current_section = None

        for i, span in enumerate(all_spans):
            text = span["text"]
            size = round(span["size"], 1)
            font = span["font"].lower()
            flags = span["flags"]
            x0 = span["bbox"][0]
            page = span["page"]

            is_bold = flags & 2 != 0
            is_centered = 200 < x0 < 400
            is_styled = is_bold or "bold" in font or is_centered
            level = size_to_level.get(size, "H3")

            if level == "title" and is_styled and not title:
                title = text
                continue

            if is_valid_heading(text) and is_styled:
                # Close previous section
                if current_section and current_section["paragraphs"]:
                    para_text = " ".join(clean_paragraph_lines(current_section["paragraphs"]))
                    current_section["keywords"] = extract_keywords_yake(para_text)
                    current_section["sentences"] = get_sentences(para_text)
                    current_section["semantic"] = analyze_text(clean_text(para_text))
                    outline.append(current_section)

                current_section = {
                    "level": level,
                    "text": text,
                    "page": page,
                    "paragraphs": [],
                    "keywords": [],
                    "sentences": [],
                    "semantic": {}
                }

                # Capture lines after heading
                for j in range(i + 1, len(all_spans)):
                    next_span = all_spans[j]
                    if next_span["page"] != page:
                        break
                    if is_valid_heading(next_span["text"]):
                        break
                    current_section["paragraphs"].append(next_span["text"])

        if current_section and current_section["paragraphs"]:
            para_text = " ".join(clean_paragraph_lines(current_section["paragraphs"]))
            current_section["keywords"] = extract_keywords_yake(para_text)
            current_section["sentences"] = get_sentences(para_text)
            current_section["semantic"] = analyze_text(clean_text(para_text))
            outline.append(current_section)

        doc.close()

        return {
            "title": clean_text(title or pdf_path.stem.replace("_", " ").title()),
            "toc": [],
            "outline": outline
        }

    except Exception as e:
        print(f"❌ Error processing {pdf_path}: {e}")
        return {"title": f"Error: {pdf_path.name}", "toc": [], "outline": []}

