

You should build a semantic matcher on top of this output. That logic would:

    Take in your challenge1b_input.json (with job_to_be_done and pdfs).

    Load corresponding processed PDFs from challenge1b_output.json-style files.

    For each section, do:

        Compute embedding of the section text (paragraphs or sentences)

        Compute embedding of the task description

        Compute cosine similarity

    Return top-N matching chunks and structure them into the output format.

This is fast, works without a database, and needs no change to the core extraction logic.


```
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, <100MB

# Embed section texts
section_texts = [" ".join(section["sentences"]) for section in outline]
section_embeddings = model.encode(section_texts)

# Embed the job string
query_embedding = model.encode(job_string)

# Compute cosine similarity
scores = util.cos_sim(query_embedding, section_embeddings)
top_indices = scores.argsort(descending=True)[0][:3]
```
---

✅ Goal Recap

Input: A JSON file with a natural language task and references to processed PDFs (like challenge1b_input.json).
Output: A JSON file containing relevant information from those PDFs, structured and filtered to help fulfill the task (like challenge1b_output.json).

Constraints:

    Max processing time: 60 seconds

    Max model/method memory: 1GB

    No external APIs/LLMs unless explicitly allowed

    Must run inside a Docker container like before

🔧 What We’ll Build

We'll create a lightweight semantic retriever that:

    Loads processed PDF outlines (JSONs from your pipeline)

    Embeds the task + sections using fast sentence embeddings

    Computes cosine similarity between task and PDF sections

    Ranks sections and returns the top ones in the expected output schema

✅ Plan to Begin

Here’s the breakdown for the first part:

    Install a fast embedding model like sentence-transformers (using paraphrase-MiniLM-L6-v2)

    Build a Python module that:

        Loads input & PDF metadata

        Embeds the task and paragraphs

        Scores similarity and selects top-k relevant segments

    Return a JSON file in the format of challenge1b_output.json

---
🔥 How We Can Leverage All Your Compute (8 cores, 16 GB RAM)
1. Parallel Processing with multiprocessing

    Python’s multiprocessing library spawns separate processes (not threads), avoiding the GIL entirely.

    Each PDF section or each document can be processed independently.

    We'll process multiple PDFs or paragraphs in parallel across cores.

✅ Advantage: Direct use of 8 cores
✅ Use case: Embed all paragraphs simultaneously
2. Batch Embedding with Sentence Transformers

    The embedding model (paraphrase-MiniLM-L6-v2) supports batching.

    We can group ~64–256 paragraphs per batch to accelerate encoding.

    It uses NumPy or PyTorch under the hood, and will use BLAS multithreaded libraries like OpenBLAS or Intel MKL if installed.

✅ Advantage: Efficient memory + CPU use
✅ Use case: Speed up large input batches

---

2. 🛠 Build Embedding Space (Offline)

You can use TfidfVectorizer from scikit-learn to:

    Convert paragraphs from all extracted PDF sections into vectors.

    Do the same for the job_to_be_done task.

Then compute cosine similarity between:

    The task vector and

    All section vectors

Use this to rank most relevant sections from all PDFs.

This is fast, lightweight, and works offline within your constraints.
3. ✂️ Refine Top Sections (Snippets)

From top-5 ranked sections:

    Extract 1–2 key sentences per section using:

        nltk.sent_tokenize()

        Rule-based filters (e.g., sentence contains “form”, “fill”, “sign”, etc.)

    These form your refined_text.

4. 🧠 JSON Construction

Construct the final challenge1b_output.json with:

    metadata from the input

    extracted_sections: top-5 based on cosine similarity

    subsection_analysis: list of important 1–2 sentence extracts from each of those sections

---

Great — now that **Challenge 1A** is complete and modularized, let’s outline a **clean, well-documented plan of action for Challenge 1B**, showing:

* ✅ How we **reuse the pipeline from 1A**
* ✅ What the **semantic matcher does**
* ✅ How the **overall architecture fits together**
* ✅ What happens when you get a **new job description**
* ✅ All tools used and how they contribute

---

## ✅ Challenge 1B — Plan of Action & Architecture

---

### 🎯 **Goal (As per Challenge Statement)**

> Given a `challenge1b_input.json` file describing a **user persona** and a **task to perform** (`job_to_do`), scan pre-processed documents and return the most **semantically relevant text blocks** in `challenge1b_output.json`.

---

## 🧩 1. System Architecture Overview

```text
                ┌────────────────────────────────────┐
                │     challenge1b_input.json         │
                │  { "persona", "task", "pdfs" }     │
                └────────────────────────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │  Pre-processed PDF JSONs     │  ◄── from Challenge 1A
                └──────────────────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │     semantic_matcher.py      │
                └──────────────────────────────┘
                             │
                             ▼
                ┌──────────────────────────────┐
                │  challenge1b_output.json     │
                │ { task, top-matching blocks }│
                └──────────────────────────────┘
```

---

## 🔁 2. Reusing Challenge 1A in Challenge 1B

### 📦 Modularization Achieved:

* We moved core logic from `process_pdfs.py` into a reusable module: `pdf_processor.py`
* `extract_document_outline()` is the main callable function
* The `process_pdfs.py` batch pipeline remains for initial PDF -> JSON conversion

### 📂 Directory Reuse:

For each `Collection` inside Challenge 1B, we:

* Set the `input_dir` to point to `CollectionX/PDFs`
* Run `process_pdfs.py` to generate JSONs
* These JSONs are consumed by the 1B matcher script

---

## 🔍 3. The Semantic Matcher Logic (`semantic_matcher.py`)

### 🔧 Components:

| Component                       | Role                                                           |
| ------------------------------- | -------------------------------------------------------------- |
| **SentenceTransformer**         | Embeds both the user’s task and PDF sections into vector space |
| **Cosine Similarity**           | Ranks similarity between task and section                      |
| **Top-K Ranking**               | Returns highest scoring section blocks                         |
| **Keyword Extraction**          | Adds useful metadata using YAKE                                |
| **Semantic Summary (Optional)** | Future extension: LLM to generate summaries                    |

### 🧠 Process:

1. **Input**: `challenge1b_input.json`

   ```json
   {
     "persona": "Travel Blogger",
     "task": "Find budget-friendly activities in South of France",
     "pdfs": ["south_guide_1.json", "nice_beach_tips.json"]
   }
   ```

2. **Pre-Processed PDFs**: Already parsed from 1A pipeline (or reprocessed using `process_pdfs.py`)

3. **Chunking**:

   * Each document is broken into **sections**
   * Each section has: `heading`, `page`, `paragraphs`, `keywords`

4. **Embedding + Ranking**:

   * The task string is embedded using **`paraphrase-MiniLM-L6-v2`**
   * All sections are embedded
   * Cosine similarity between task and each section determines relevance

5. **Output**: Top-k results in `challenge1b_output.json`

   ```json
   {
     "task": "...",
     "matches": [
       {
         "pdf_name": "south_guide_1.json",
         "page": 3,
         "section_heading": "Budget Activities in Nice",
         "matched_content": "...",
         "keywords": ["local food", "socca", "promenade"],
         "score": 0.87,
         "semantic_summary": ""
       }
     ]
   }
   ```

---

## 🔨 4. Tools & Tech Stack

| Tool                       | Role                                                            |
| -------------------------- | --------------------------------------------------------------- |
| **PyMuPDF (`fitz`)**       | PDF parsing, layout, headings, paragraph segmentation           |
| **YAKE**                   | Lightweight keyword extractor                                   |
| **spaCy**                  | Text cleaning, sentence segmentation                            |
| **SentenceTransformers**   | Semantic embeddings and matching                                |
| **Docker**                 | Containerization (limited to ≤ 16GB RAM, 8 vCPUs, ≤ 1GB models) |
| **JSON Schema (optional)** | Used to structure outputs consistently                          |

---

## 🛠️ 5. Developer Actions

### ➤ PDF → JSON Extraction (run once per collection)

```bash
docker run --rm \
  -v $(pwd)/Challenge_1b/Collection\ 1/PDFs:/app/input \
  -v $(pwd)/Challenge_1b/Collection\ 1/PDF_JSONS:/app/output \
  pdf-outline-extractor
```

### ➤ Semantic Matching (per input file)

```bash
python semantic_matcher.py \
  --input Challenge_1b/Collection\ 1/challenge1b_input.json \
  --output Challenge_1b/Collection\ 1/challenge1b_output.json \
  --pdf_dir Challenge_1b/Collection\ 1/PDF_JSONS
```

---

## 🧩 6. Optional Enhancements

| Feature                | Description                                                              |
| ---------------------- | ------------------------------------------------------------------------ |
| `semantic_summary`     | Call an LLM to summarize the match                                       |
| Keyword Matching Boost | Combine cosine + keyword overlap for better relevance                    |
| Confidence Thresholds  | Add `min_score` filters for matches                                      |
| Ranking by Persona     | Pre-weight sections based on persona focus (e.g. budget, family, luxury) |

---

## ✅ TL;DR Summary

| Step | Module                | Role                                          |
| ---- | --------------------- | --------------------------------------------- |
| 1    | `process_pdfs.py`     | Converts PDFs to structured JSON              |
| 2    | `pdf_processor.py`    | Core logic for PDF segmentation               |
| 3    | `semantic_matcher.py` | Accepts a task, returns top matching sections |
| 4    | Output                | Top-k results in standardized JSON format     |

---

✅ Immediate Plan:

We'll write a single Python script that:

    Loads challenge1b_input.json (with task + PDF paths).

    Loads each corresponding .json output (produced from process_pdfs.py).

    Embeds:

        The task string

        All paragraphs + headings from each PDF

    Computes cosine similarity between task and all paragraph/section embeddings.

    Selects top-k most relevant sections.

    Returns the matched sections with their headings, paragraphs, keywords, and scores in the required challenge1b_output.json format.

🧩 Assumptions for the first version:

    Model used: sentence-transformers/paraphrase-MiniLM-L6-v2 (~85MB RAM)

    Max results: top 10 relevant blocks

    Keep output format matching challenge1b_output.json you provided

    We’ll add batching + parallelization after you test this version