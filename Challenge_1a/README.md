# Challenge 1a: PDF Processing Solution

## Overview
This is a **sample solution** for Challenge 1a of the Adobe India Hackathon 2025. The challenge requires implementing a PDF processing solution that extracts structured data from PDF documents and outputs JSON files. The solution must be containerized using Docker and meet specific performance and resource constraints.

## Official Challenge Guidelines

### Submission Requirements
- **GitHub Project**: Complete code repository with working solution
- **Dockerfile**: Must be present in the root directory and functional
- **README.md**:  Documentation explaining the solution, models, and libraries used

### Build Command
```bash
docker build --platform linux/amd64 -t <reponame.someidentifier> .
```

### Run Command
```bash
docker run --rm -v $(pwd)/input:/app/input:ro -v $(pwd)/output/repoidentifier/:/app/output --network none <reponame.someidentifier>
```

### Critical Constraints
- **Execution Time**: ≤ 10 seconds for a 50-page PDF
- **Model Size**: ≤ 200MB (if using ML models)
- **Network**: No internet access allowed during runtime execution
- **Runtime**: Must run on CPU (amd64) with 8 CPUs and 16 GB RAM
- **Architecture**: Must work on AMD64, not ARM-specific

### Key Requirements
- **Automatic Processing**: Process all PDFs from `/app/input` directory
- **Output Format**: Generate `filename.json` for each `filename.pdf`
- **Input Directory**: Read-only access only
- **Open Source**: All libraries, models, and tools must be open source
- **Cross-Platform**: Test on both simple and complex PDFs

## Sample Solution Structure
```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # JSON files provided as outputs.
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile           # Docker container configuration
├── process_pdfs.py      # Sample processing script
└── README.md           # This file
```

## Sample Implementation

### Current Sample Solution
The provided `process_pdfs.py` is a **basic sample** that demonstrates:
- PDF file scanning from input directory
- Dummy JSON data generation
- Output file creation in the specified format

**Note**: This is a placeholder implementation using dummy data. A real solution would need to:
- Implement actual PDF text extraction
- Parse document structure and hierarchy
- Generate meaningful JSON output based on content analysis

### Sample Processing Script (`process_pdfs.py`)
```python
# Current sample implementation
def process_pdfs():
    input_dir = Path("/app/input")
    output_dir = Path("/app/output")
    
    # Process all PDF files
    for pdf_file in input_dir.glob("*.pdf"):
        # Generate structured JSON output
        # (Current implementation uses dummy data)
        output_file = output_dir / f"{pdf_file.stem}.json"
        # Save JSON output
```

### Sample Docker Configuration
```dockerfile
FROM --platform=linux/amd64 python:3.10
WORKDIR /app
COPY process_pdfs.py .
CMD ["python", "process_pdfs.py"]
```

## Expected Output Format

### Required JSON Structure
Each PDF should generate a corresponding JSON file that **must conform to the schema** defined in `sample_dataset/schema/output_schema.json`.


## Implementation Guidelines

### Performance Considerations
- **Memory Management**: Efficient handling of large PDFs
- **Processing Speed**: Optimize for sub-10-second execution
- **Resource Usage**: Stay within 16GB RAM constraint
- **CPU Utilization**: Efficient use of 8 CPU cores

### Testing Strategy
- **Simple PDFs**: Test with basic PDF documents
- **Complex PDFs**: Test with multi-column layouts, images, tables
- **Large PDFs**: Verify 50-page processing within time limit


## Testing Your Solution

### Local Testing
```bash
# Build the Docker image
docker build --platform linux/amd64 -t pdf-processor .

# Test with sample data
docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/input:ro -v $(pwd)/sample_dataset/outputs:/app/output --network none pdf-processor
```

### Validation Checklist
- [ ] All PDFs in input directory are processed
- [ ] JSON output files are generated for each PDF
- [ ] Output format matches required structure
- [ ] **Output conforms to schema** in `sample_dataset/schema/output_schema.json`
- [ ] Processing completes within 10 seconds for 50-page PDFs
- [ ] Solution works without internet access
- [ ] Memory usage stays within 16GB limit
- [ ] Compatible with AMD64 architecture

---

**Important**: This is a sample implementation. Participants should develop their own solutions that meet all the official challenge requirements and constraints. 

---

## 🧠 Step-by-Step Pipeline for Semantic PDF Structuring

### **Step 1: Load and Structure PDF**

* **Goal**: Extract text blocks with layout info (font size, page number, position).
* **Tools**: \[`PyMuPDF (fitz)`]
* **Output**: Raw blocks grouped into title, H1–H3 based on font + position.

---

### **Step 2: Identify Headings + Outline**

* **Goal**: Use font size, position, and regex to determine hierarchy (H1/H2/H3).
* **Tools**: Your current heuristics + regex
* **Output**: JSON outline with `level`, `text`, `page`, `section_type`, `persona_focus`

---

### **Step 3: Language Detection**

* **Goal**: Detect the language of each heading or block.
* **Tools**:

  * \[`langdetect`] (very fast, multilingual)
  * or \[`fastText`] (slightly heavier, more accurate)
* **Output**: Add `"language": "fr"` (or `"en"`, etc.)

---

### **Step 4: Translate (if needed)**

* **Goal**: Translate to English for semantic and persona tagging.
* **Tools**:

  * \[`argos-translate`] (offline, lightweight)
  * or \[`transformers` + `Helsinki-NLP/opus-mt-*`] if running locally
* **Output**: Add `"translated_text"` field only if language ≠ English

---

### **Step 5: Semantic Labeling (Updated)**

* **Goal**: Match heading text to broader section types.
* **Tools**: Keyword + fuzzy match rules (your current improved `label_semantics()` function)
* **Output**: `"section_type": "Execution"` etc.

---

### **Step 6: Persona Mapping**

* **Goal**: Suggest target audience for the section.
* **Tools**: Your current `tag_persona_focus()` logic
* **Output**: `"persona_focus": ["Engineer", "Event Planner"]`

---

### **Step 7: Content Type Detection**

* **Goal**: Classify document or section as a `report`, `email`, `notice`, `invite`, etc.
* **Tools**:

  * \[`scikit-learn`] or \[`transformers` with small models] trained on document genres
  * or use rule-based if only section headings are available
* **Output**: `"content_type": "invitation"`

---

### **Step 8: Keyword Extraction**

* **Goal**: Pull 3–5 key terms for each heading block.
* **Tools**:

  * \[`KeyBERT`] (uses BERT embeddings, works even in small batches)
  * or \[`RAKE`], `YAKE`, or `TF-IDF` (classic methods, fast)
* **Output**: `"keywords": ["deadline", "submission", "formatting"]`

---

### **Step 9: Sentiment Analysis (Optional)**

* **Goal**: Determine tone (neutral, urgent, positive, etc.).
* **Tools**:

  * \[`VADER`] for English (super light)
  * or multilingual sentiment from Hugging Face if translated
* **Output**: `"sentiment": "neutral"`

---

## 🔧 Tool Stack Summary

| Purpose            | Tool / Library                | Offline? | Notes                 |
| ------------------ | ----------------------------- | -------- | --------------------- |
| PDF Parsing        | PyMuPDF                       | ✅        | Already used          |
| Language Detection | langdetect / fastText         | ✅        | Fast & reliable       |
| Translation        | argos-translate / Helsinki MT | ✅        | Offline model-based   |
| Keywords           | KeyBERT / YAKE / RAKE         | ✅        | Very lightweight      |
| Sentiment          | VADER (for English)           | ✅        | Optional              |
| Content Typing     | Rule-based or scikit-learn    | ✅        | Train on your doc set |

---

Would you like a base directory + file structure now to start implementing each of these modularly (starting with translation + language detection)?




Absolutely! Here's the **complete, updated PDF processing workflow** including all enhanced JSON fields, semantic tagging, persona mapping, and intelligent Table of Contents (TOC) integration:

---

## 🧠 **Semantic PDF Structuring — Final Workflow Summary**

### 🔁 Step-by-Step Pipeline

---

### **📥 Step 1: Load and Structure PDF**

**Goal:** Extract structured content from the PDF, preferring semantic outlines (TOC) where available.

#### 🔹 What We Do:

* Open the PDF using `PyMuPDF`.
* Attempt to extract built-in Table of Contents via `doc.get_toc()`.
* If `TOC` exists:

  * Use it directly to extract heading text + page number + level.
* If `TOC` doesn't exist:

  * Use **font-size heuristics** to detect headings (as done already).

---

### **🧠 Step 2: Enhance Each Section with Semantics**

For each detected heading/section (from TOC or heuristics):

| Enhancement Task        | Output Field in JSON   | Tool / Logic                                             |
| ----------------------- | ---------------------- | -------------------------------------------------------- |
| Semantic category       | `section_type`         | Rule-based (e.g., “Introduction”, “Results”)             |
| Persona focus           | `persona_focus`        | Rule-based (e.g., \["Manager", "Student"])               |
| Language detection      | `language`             | `langdetect` or `langid`                                 |
| Translation (if needed) | `translated_text`      | `translate` using `transformers`, `argostranslate`, etc. |
| Extract keywords        | `keywords`             | `RAKE`, `keyBERT`, or noun-phrase filter                 |
| Content type            | `content_type`         | Rule-based or lightweight classifier                     |
| Sentiment analysis      | `sentiment` (optional) | `VADER` / `textblob` / multilingual model                |
| Topic tagging           | `topics` (optional)    | `keyBERT`, `zero-shot classification`, etc.              |

---

### **💾 Step 3: Output Enhanced JSON**

Here's what the **final JSON per section** looks like:

```json
{
  "title": "Understanding Transformers",
  "outline": [
    {
      "level": "H1",
      "text": "1. Introduction to Transformers",
      "page": 1,
      "section_type": "Introduction",
      "persona_focus": ["Everyone"],
      "language": "en",
      "translated_text": "1. Introduction to Transformers",
      "keywords": ["transformers", "architecture", "introduction"],
      "content_type": "report",
      "sentiment": "neutral",
      "topics": ["Machine Learning", "AI"]
    },
    ...
  ]
}
```

---

### 🧱 **Tech Stack Overview**

| Task                      | Library / Tool                    |
| ------------------------- | --------------------------------- |
| PDF Parsing               | `PyMuPDF (fitz)`                  |
| TOC / Fallback Heuristics | `doc.get_toc()` + font size logic |
| Language Detection        | `langdetect`, `langid`            |
| Translation               | `argostranslate`, `transformers`  |
| Keywords                  | `keyBERT`, `RAKE`, `spaCy`        |
| Content Type Tagging      | Rule-based / zero-shot HF         |
| Sentiment                 | `VADER`, `TextBlob`               |
| Topics                    | `keyBERT`, `ZeroShotClassifier`   |

---

### ✅ Key Benefits

* 💡 Understands structure using both TOC and visual cues
* 🌐 Works for multilingual documents with translation fallback
* 🧠 Adds rich metadata like semantics, personas, and keywords
* 📦 Output is machine-friendly and index-ready

---

Let me know if you’d like:

* The updated code file implementing this,
* Dockerfile suggestions to install extra dependencies,
* Or a lightweight HF pipeline suggestion using HuggingFace Inference API or `transformers` CPU inference only.
