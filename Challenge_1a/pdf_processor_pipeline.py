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
def extract_document_outline(pdf_path: Path) -> dict:
    title = ""
    outline = []
    toc = []

    try:
        doc = fitz.open(pdf_path)

        raw_toc = doc.get_toc()
        for item in raw_toc:
            level, text, page = item
            toc.append({"level": level, "text": text.strip(), "page": page})

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

        current_section = None
        prev_y = None

        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            blocks = page.get_text('dict')['blocks']

            lines = []
            for b in blocks:
                if b['type'] == 0:
                    for line in b['lines']:
                        lines.append(line)

            i = 0
            while i < len(lines):
                line = lines[i]
                if not line['spans']:
                    i += 1
                    continue

                line_text = " ".join([span['text'] for span in line['spans']]).strip()
                if not line_text:
                    i += 1
                    continue

                line_y = line['spans'][0]['bbox'][1]
                vertical_gap = line_y - prev_y if prev_y is not None else 0
                prev_y = line_y

                this_x = line['spans'][0]['bbox'][0]

                avg_font_size = sum(span["size"] for span in line['spans']) / len(line['spans'])
                heading_level = get_heading_level(avg_font_size, font_thresholds, line['spans'], body_font_size)

                next_line_indent = False
                if i + 1 < len(lines):
                    next_x = lines[i + 1]['spans'][0]['bbox'][0]
                    if next_x - this_x > 10:
                        next_line_indent = True

                if is_heading_candidate(line_text, line['spans'], vertical_gap, font_thresholds, next_line_indent):
                    if current_section and current_section["paragraphs"]:
                        cleaned = clean_paragraph_lines(current_section["paragraphs"])
                        full_text = " ".join(cleaned)
                        current_section["keywords"] = extract_keywords_yake(full_text)
                        current_section["sentences"] = get_sentences(full_text)
                        current_section["semantic"] = analyze_text(clean_text(full_text))

                    current_section = {
                        "level": heading_level,
                        "text": line_text,
                        "page": page_num + 1,
                        "paragraphs": [],
                        "keywords": [],
                        "sentences": [],
                        "semantic": {}
                    }
                    outline.append(current_section)

                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        if not next_line['spans']:
                            j += 1
                            continue

                        next_line_text = " ".join([span['text'] for span in next_line['spans']]).strip()
                        if not next_line_text:
                            j += 1
                            continue

                        next_x = next_line['spans'][0]['bbox'][0]
                        if next_x - this_x > 10:
                            current_section["paragraphs"].append(next_line_text)
                            j += 1
                        else:
                            break
                    i = j
                elif current_section:
                    current_section["paragraphs"].append(line_text)
                    i += 1
                else:
                    i += 1

        if current_section and current_section["paragraphs"] and not current_section["keywords"]:
            cleaned = clean_paragraph_lines(current_section["paragraphs"])
            full_text = " ".join(cleaned)
            current_section["keywords"] = extract_keywords_yake(full_text)
            current_section["sentences"] = get_sentences(full_text)
            current_section["semantic"] = analyze_text(clean_text(full_text))

    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        title = f"Error processing {pdf_path.name}"
        outline = []

    finally:
        doc.close()

    if not title:
        title = pdf_path.stem.replace("_", " ").title()

    # Clean the data before returning
    title = clean_text(title)
    toc = [{"level": t["level"], "text": clean_text(t["text"]), "page": t["page"]} for t in toc]
    outline = [clean_section_data(section) for section in outline]
    
    return {
        "title": title,
        "toc": toc,
        "outline": outline
    }
