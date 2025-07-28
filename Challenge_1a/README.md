
# 📄 Challenge 1A – PDF Outline Extractor

## 🔍 Task Outline

Build a containerized PDF processor that extracts structured content from PDFs, including:

* Document title
* Table of contents (if available)
* Section hierarchy (H1, H2, H3)
* Paragraph-level text
* Keywords and semantic information (nouns, verbs, lemmas)

## 🧠 Project Outline

This system processes PDFs to generate clean and semantically rich structured output in JSON format. It uses layout-based heuristics and natural language processing to:

* Identify headings from font and formatting styles
* Segment text into meaningful paragraphs
* Extract keywords using YAKE
* Perform semantic annotation using spaCy

Each section in the output contains both visual and semantic metadata useful for downstream tasks.

## 💡 Explanation of the Idea

Unlike naive TOC-based approaches, this pipeline scans all PDF content to infer document structure using:

* Font size thresholds and styling (bold, italic, underline)
* Visual layout cues like vertical spacing and indentation
* Rule-based keyword filtering
* NLP tools for tokenization, lemmatization, and syntactic parsing

This makes the system robust to unstructured, scanned, or inconsistent PDF formatting.

### Example Output Structure:

```json
{
  "title": "Sample PDF Title",
  "toc": [...],
  "outline": [
    {
      "level": "H1",
      "text": "Getting Started",
      "page": 2,
      "paragraphs": ["Welcome to the guide...", "..."],
      "keywords": ["getting started", "setup"],
      "sentences": ["Welcome to the guide..."],
      "semantic": {
        "tokens": ["Welcome", "guide"],
        "nouns": ["guide"],
        "verbs": [],
        "lemmas": ["welcome", "guide"]
      }
    }
  ]
}
```

## 🐳 How to Build and Run

### 🔧 Build Docker Image

```bash
docker build --platform linux/amd64 -t pdf-outline-extractor .
```
or
```bash
DOCKER_BUILDKIT=0 docker build --platform linux/amd64 -t pdf-outline-extractor .
```

### 🚀 Run PDF Processor

```bash
docker run --rm -v $(pwd)/sample_dataset/pdfs:/app/input:ro -v $(pwd)/sample_dataset/outputs:/app/output --network none pdf-outline-extractor
```

## 📁 Directory Structure

```
Challenge_1a/
├── sample_dataset/
│   ├── outputs/         # Extracted JSON outputs
│   ├── pdfs/            # Input PDF files
│   └── schema/          # Output schema definition
│       └── output_schema.json
├── Dockerfile                   # Docker container configuration
├── process_pdfs.py             # Entrypoint script for batch processing
├── pdf_processor_pipeline.py   # PDF parsing and layout analysis logic
├── nlp_utils.py                # NLP functions for tokenization, keyword extraction
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation (this file)
```

---

## 🧠 Why Extract Keywords, Semantics, Lemmas, Verbs & Nouns?

Challenge 1A is designed to extract structured outlines from PDFs, but this project goes further by **embedding semantic intelligence** into the extracted sections.

### 🔍 Why Extract This Information?

| Feature      | Purpose                                                                                                          |
| ------------ | ---------------------------------------------------------------------------------------------------------------- |
| **Keywords** | Capture the essence of each section using unsupervised techniques like YAKE. Helps in identifying core concepts. |
| **Nouns**    | Reveal entities, topics, and subjects being discussed—crucial for content classification.                        |
| **Verbs**    | Indicate actions and processes, helping understand what is being done in a section.                              |
| **Lemmas**   | Provide normalized root forms of words for better semantic comparison and search.                                |
| **Tokens**   | Help in building embeddings, scoring similarity, and enabling NLP-powered ranking tasks.                         |

---

### 🚀 How This Supports Challenge 1B

Challenge 1B focuses on **semantic matching between task descriptions and relevant PDF sections**.

To enable this:

* The output of Challenge 1A becomes the **input for semantic matching** in 1B.
* Richly annotated sections (with nouns, verbs, lemmas, and keywords) allow us to:

  * Perform **task-aware matching** using semantic similarity.
  * Compute **document relevance scores**.
  * Generate **summaries** aligned with a user's task or persona.
  * Enable fine-grained filters like “only find procedural sections” or “extract places or food names.”

This design turns the system into not just a structural extractor, but a **semantic foundation** for downstream NLP applications in document understanding.

---

## 👥 Authors

* **Anagh Verma**
* **Ashutosh Bhardwaj**
* **Rishit Mohan**
  Adobe India Hackathon 2025 – Challenge 1A Contributors


---
