# 🧠 Challenge 1B – Semantic Matcher for PDFs

## 🔍 Task Outline

Given a task description and a set of PDF documents, build a system that:

* Extracts structured semantic content from PDFs
* Matches the task against document content using semantic similarity
* Returns top-ranked, summarized sections with metadata in a specified JSON schema

## 🧱 Project Outline

This system processes multiple collections of PDFs to semantically extract and rank relevant sections based on a business task. It includes:

* Document parsing with heading/paragraph detection
* NLP-based semantic token extraction
* Embedding generation and similarity scoring
* Summarization of matched content
* Final output formatting in JSON

## 💡 Explanation of the Idea

1. **PDF Parsing**: Each PDF is processed using layout heuristics and font-based heading detection to extract clean, semantically annotated sections.
2. **Task Matching**: A `job_to_be_done` task is compared with all document sections using Sentence-BERT embeddings (`MiniLM`).
3. **Summarization**: Top-matched sections are summarized using `Flan-T5-small` to provide a concise business-friendly overview.
4. **Output**: The results are saved in the required output schema, including metadata, matched sections, and their summaries.

---

## 🐳 How to Build and Run

### 🔧 Step 1: Build Docker Image

```bash
docker build -t semantic-matcher .
```

or

```bash
DOCKER_BUILDKIT=0 docker build -t semantic-matcher .
```

### 🚀 Step 2: Run the Matcher

```bash
docker run --rm -v $(pwd)/collections:/app/collections \
           -v $(pwd)/outputs:/app/outputs \
           semantic-matcher
```

Make sure you have the `collections/` folder structured with:

```
collections/
├── Collection 1/
│   ├── challenge1b_input.json
│   ├── PDFs/
│   └── json_output/    # Will be auto-created
├── Collection 2/
├── Collection 3/
```

---

## 📁 Directory Structure

```
Challenge_1b/
├── collections/                 # Input collections of PDFs
│   ├── Collection 1/
│   │   ├── challenge1b_input.json
│   │   ├── PDFs/
│   │   └── json_output/        # Output JSONs per PDF (auto-filled)
│   ├── Collection 2/
│   └── Collection 3/
├── outputs/                    # Final results saved here
├── Dockerfile                 # Docker container setup
├── nlp_utils.py               # NLP utilities (shared with 1A)
├── pdf_processor_pipeline.py  # PDF parsing and semantic extraction (shared with 1A)
├── semantic_matcher.py        # Main semantic matching pipeline
├── requirements.txt           # All Python dependencies
└── README.md                  # This file
```

---

## 🔍 Output Schema (Summary)

Each output JSON has:

* `metadata`: input files, persona, job task, timestamp
* `extracted_sections`: ranked sections with heading and page
* `subsection_analysis`: summaries per matched section

---

## 👥 Authors

* **Anagh Verma**
* **Ashutosh Bhardwaj**
* **Rishit Mohan**
  Adobe India Hackathon 2025 – Challenge 1B Contributors

---

