# analyzer.py
# Core analysis engine. Five checks + full text and sentence storage for NLP pipeline.

import re

from .readability_metrics import coleman_liau_index, flesch_reading_ease, count_words
from .keywords import (
    DATA_CATEGORIES, OPT_IN_PHRASES, OPT_OUT_PHRASES,
    THIRD_PARTY_PHRASES, RETENTION_PHRASES
)


# ── Date extraction patterns ───────────────────────────────────────────────────

DATE_PATTERNS = [
    re.compile(r"(?:effective|updated|last\s+(?:updated|modified|revised))\s*(?:on|:)?\s*(\w+\s+\d{1,2},?\s+\d{4})", re.I),
    re.compile(r"(?:effective|updated)\s*(?:on|:)?\s*(\d{1,2}/\d{1,2}/\d{4})", re.I),
    re.compile(r"(?:effective|updated)\s*(?:on|:)?\s*(\d{4}-\d{2}-\d{2})", re.I),
    re.compile(r"(?:effective|updated)\s*(?:on|:)?\s*(\d{1,2}\s+\w+\s+\d{4})", re.I),
]

# ── Section heading patterns ──────────────────────────────────────────────────

HEADING_PATTERNS = [
    re.compile(r"^(\d+\.?\s+[A-Z][A-Za-z\s]{3,60})$"),
    re.compile(r"^([A-Z][A-Z\s]{4,60})$"),
    re.compile(r"^([A-Z][A-Za-z\s]{3,60}):$"),
    re.compile(r"^(?:Section\s+)?\d+\.?\d*\.?\s*[-\u2013]?\s*(.{5,60})$", re.I),
]


# ── Text utilities ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list:
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if len(s.strip()) > 30]


def find_example_sentence(keyword: str, sentences: list) -> str | None:
    kw = keyword.lower()
    for sent in sentences:
        if kw in sent.lower():
            return sent[:220]
    return None


# ── Policy date extraction ─────────────────────────────────────────────────────

def extract_policy_date(text: str):
    """Extract effective/updated date from first 3000 chars of policy text."""
    for pattern in DATE_PATTERNS:
        match = pattern.search(text[:3000])
        if match:
            return match.group(1).strip()
    return None


# ── Section extraction ────────────────────────────────────────────────────────

def detect_heading(sentence: str):
    """Return heading text if sentence looks like a section heading, else None."""
    s = sentence.strip()
    if len(s) > 80 or len(s) < 4:
        return None
    for pattern in HEADING_PATTERNS:
        m = pattern.match(s)
        if m:
            return m.group(1).strip() if m.lastindex else s
    if len(s) < 60 and not s.endswith('.') and s[0].isupper():
        words = s.split()
        upper_ratio = sum(1 for w in words if w[0].isupper()) / max(len(words), 1)
        if upper_ratio >= 0.7 and len(words) <= 8:
            return s
    return None


def extract_sections_from_raw(raw_text: str) -> list:
    """
    Split RAW text (before clean_text) into sections by detecting headings
    from line breaks. Must run on original text with newlines preserved.
    Returns list of dicts: {heading, text, sentence_count}.
    """
    lines = raw_text.split("\n")
    sections = []
    current_heading = "Introduction"
    current_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        heading = detect_heading(line)
        if heading:
            # Flush previous section
            if current_lines:
                body = " ".join(current_lines)
                body = re.sub(r"\s+", " ", body).strip()
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)
                         if len(s.strip()) > 30]
                sections.append({
                    "heading": current_heading,
                    "text": body,
                    "sentences": sents,
                    "sentence_count": len(sents)
                })
            current_heading = heading
            current_lines = []
        else:
            current_lines.append(line)

    # Final section
    if current_lines:
        body = " ".join(current_lines)
        body = re.sub(r"\s+", " ", body).strip()
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body)
                 if len(s.strip()) > 30]
        sections.append({
            "heading": current_heading,
            "text": body,
            "sentences": sents,
            "sentence_count": len(sents)
        })

    return sections


# ── Individual checks ──────────────────────────────────────────────────────────

def check_readability(text: str) -> dict:
    grade = coleman_liau_index(text)
    ease = flesch_reading_ease(text)
    words = count_words(text)

    if grade <= 8:
        description = "Easy — readable by most people"
    elif grade <= 12:
        description = "High school level — manageable for most adults"
    elif grade <= 16:
        description = "College level — challenging for average user"
    else:
        description = "Graduate level — likely inaccessible to most users"

    return {
        "grade_level": round(grade, 1),
        "reading_ease": round(ease, 1),
        "word_count": words,
        "description": description
    }


def check_data_categories(text: str, sentences: list) -> dict:
    """
    For each category, returns:
      - keyword_hits: total count of individual keyword matches (ML feature)
      - matched_keywords: which specific keywords were found
      - example_sentences: up to 2 supporting sentences (human-readable)
    """
    text_lower = text.lower()
    found = {}

    for category, keywords in DATA_CATEGORIES.items():
        matched = []
        hit_count = 0
        examples = []

        for kw in keywords:
            # Count ALL occurrences for frequency feature
            occurrences = text_lower.count(kw)
            if occurrences > 0:
                hit_count += occurrences
                matched.append(kw)
                if len(examples) < 2:
                    ex = find_example_sentence(kw, sentences)
                    if ex:
                        examples.append(ex)

        if matched:
            found[category] = {
                "keyword_hits": hit_count,          # quantitative ML feature
                "matched_keywords": matched,         # which keywords triggered
                "example_sentences": examples        # for human inspection
            }

    return found


def check_consent_model(text: str, sentences: list) -> dict:
    text_lower = text.lower()

    opt_in_hits, opt_out_hits = [], []

    for phrase in OPT_IN_PHRASES:
        if phrase in text_lower:
            ex = find_example_sentence(phrase, sentences)
            if ex:
                opt_in_hits.append(ex)

    for phrase in OPT_OUT_PHRASES:
        if phrase in text_lower:
            ex = find_example_sentence(phrase, sentences)
            if ex:
                opt_out_hits.append(ex)

    # Numeric scores for ML
    opt_in_score = len(opt_in_hits)
    opt_out_score = len(opt_out_hits)

    if opt_in_hits and not opt_out_hits:
        model = "Primarily Opt-In"
    elif opt_out_hits and not opt_in_hits:
        model = "Primarily Opt-Out"
    elif opt_in_hits and opt_out_hits:
        model = "Mixed"
    else:
        model = "Unclear"

    return {
        "model": model,
        "opt_in_phrase_count": opt_in_score,    # ML feature
        "opt_out_phrase_count": opt_out_score,  # ML feature
        "opt_in_examples": opt_in_hits[:2],
        "opt_out_examples": opt_out_hits[:2]
    }


def check_third_party_sharing(text: str, sentences: list) -> dict:
    text_lower = text.lower()
    examples = []
    total_count = 0

    for phrase in THIRD_PARTY_PHRASES:
        occurrences = text_lower.count(phrase)
        if occurrences > 0:
            total_count += occurrences
            ex = find_example_sentence(phrase, sentences)
            if ex and ex not in examples:
                examples.append(ex)

    return {
        "detected": total_count > 0,
        "total_mentions": total_count,       # ML feature: raw frequency
        "examples": examples[:3]
    }


def check_retention_policy(text: str, sentences: list) -> dict:
    text_lower = text.lower()
    examples = []
    total_count = 0

    for phrase in RETENTION_PHRASES:
        occurrences = text_lower.count(phrase)
        if occurrences > 0:
            total_count += occurrences
            ex = find_example_sentence(phrase, sentences)
            if ex and ex not in examples:
                examples.append(ex)

    return {
        "detected": total_count > 0,
        "total_mentions": total_count,
        "examples": examples[:2]
    }


# ── Master function ────────────────────────────────────────────────────────────

def analyze(text: str) -> dict:
    """
    Run all checks. Returns structured features AND the full text + sentences
    so the JSON output is directly usable for downstream NLP without re-scraping.
    """
    # Extract policy date and sections from RAW text (newlines preserved)
    policy_date = extract_policy_date(text)
    sections = extract_sections_from_raw(text)

    # Now clean for all other analysis
    text = clean_text(text)
    sentences = split_sentences(text)

    return {
        # ── Raw text (required for NLP: TF-IDF, embeddings, topic modeling) ──
        "full_text": text,
        "sentences": sentences,

        # ── Policy metadata ────────────────────────────────────────────────
        "policy_date": policy_date,
        "sections": sections,
        "section_count": len(sections),
        "section_headings": [s["heading"] for s in sections],

        # ── Structured features (ready for ML tabular models) ──────────────
        "readability": check_readability(text),
        "data_categories": check_data_categories(text, sentences),
        "consent_model": check_consent_model(text, sentences),
        "third_party_sharing": check_third_party_sharing(text, sentences),
        "retention_policy": check_retention_policy(text, sentences),
        "sentence_count": len(sentences)
    }
