# nlp_utils.py

import re
import spacy
from typing import List, Dict

# Load English model once
nlp = spacy.load("en_core_web_sm")

# 1. Clean raw text
def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\u2022\u00a9\u202f\u200b\uf0b7]", " ", text)  # Remove bullets/specials
    text = re.sub(r"\d{1,2}/\d{1,2}/\d{2,4}", "", text)  # remove dates
    return text.strip()

# 2. Tokenize + POS tag + Lemmatize
def analyze_text(text: str) -> Dict:
    doc = nlp(text)

    tokens = []
    nouns = []
    verbs = []
    lemmas = []

    for token in doc:
        if not token.is_stop and not token.is_punct and not token.is_space:
            tokens.append(token.text)
            lemmas.append(token.lemma_)
            if token.pos_ == "NOUN":
                nouns.append(token.lemma_)
            elif token.pos_ == "VERB":
                verbs.append(token.lemma_)

    return {
        "tokens": tokens,
        "nouns": list(set(nouns)),
        "verbs": list(set(verbs)),
        "lemmas": list(set(lemmas))
    }

# 3. Segment text into sentences
def get_sentences(text: str) -> List[str]:
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
