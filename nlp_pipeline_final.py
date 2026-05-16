"""
nlp_pipeline_final.py
Privacy Policy NLP Pipeline — Final Production Version

Architecture:
  Phase 1: Ingest cleaned JSON → filter noise sections → chunk → DataFrame
  Phase 2:
    Module A: Intent classification (HuggingFace bart-large-mnli API)
    Module B: Supply chain (READ from JSON — Nemotron already extracted)
    Module C: Dark pattern detection (spaCy, parallel, CPU cores - 1)
  Phase 3: Company aggregation → CSV + 3 Plotly figures

Data Cleaning:
  Sections with 0 sentences are noise (nav items, addresses, state names).
  Sections with 1+ sentences are kept — including numbered headings.
  Additional filters: emails, postal addresses, standalone state/country names.
  All cleaning is inline during Phase 1, no separate step needed.

Usage:
    python nlp_pipeline_final.py --reports-dir prepared_reports_final --skip-intent
    python nlp_pipeline_final.py --reports-dir prepared_reports_final --hf-token hf_xxx
"""
#

# ══════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ══════════════════════════════════════════════════════════════════════════════

import os
import sys
import re
import json
import time
import argparse
import warnings
import multiprocessing
from pathlib import Path

# Encoding safety net: this repo runs on a GBK (Windows China) console that
# raises UnicodeEncodeError on the ✓/✗/⚠/═ glyphs used throughout this
# pipeline's progress + figure logging. Force UTF-8 so a full corpus run
# cannot die on a print() call.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

import requests as req
import pandas as pd
import spacy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

REPORTS_DIR      = "prepared_reports_final"
OUTPUT_DIR       = "nlp_outputs"
MAX_TOKENS       = 500
MIN_CHUNK_TOKENS = 30
N_WORKERS        = max(1, (os.cpu_count() or 2) - 1)

HF_API_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"

INTENT_LABELS = [
    "Primary App Functionality",
    "Targeted Advertising",
    "Third-Party Data Sale",
    "Security and Fraud Prevention",
    "AI Model Training",
]

PASSIVE_DEPS = {"nsubj:pass", "auxpass", "nsubjpass"}

VAGUE_PATTERNS = re.compile(
    r"\b(may|might|could|sometimes|certain|some|appropriate|"
    r"necessary|relevant|reasonable|various|other parties|"
    r"from time to time|as needed|at our discretion)\b",
    re.IGNORECASE
)

CAT_COLORS = {
    "Big Tech": "#4C72B0", "AI": "#DD8452", "Social": "#55A868",
    "Privacy-First": "#C44E52", "Data Broker": "#8172B2", "Platform": "#937860",
}

# ── Section noise filters ─────────────────────────────────────────────────────
# Built from analysis of all 23 cleaned JSONs (145 noise headings categorized).
# Primary rule: 0-sentence sections are always noise.
# Secondary rules catch the few 0-sentence sections that slip through
# when sentence_count is missing from the JSON.

# Standalone US state names (from legal jurisdiction lists in Palantir, Equifax)
US_STATES = {
    "virginia", "colorado", "connecticut", "delaware", "texas", "california",
    "montana", "oregon", "utah", "iowa", "indiana", "tennessee", "nebraska",
    "new hampshire", "new jersey", "maryland", "minnesota", "rhode island",
}

# Standalone country names (from geographic scope lists)
COUNTRIES = {
    "united states", "united kingdom", "canada", "australia", "germany",
    "france", "japan", "china", "india", "brazil", "singapore",
}


def is_noise_section(heading: str, sentence_count: int) -> bool:
    """
    Determine if a section is noise that should be excluded from NLP analysis.

    Decision tree (derived from 23-company, 145-noise-heading analysis):
      1. 0 sentences → always noise (nav items, labels, empty sections)
      2. Email address → noise (contact info, not policy content)
      3. Postal address (contains 5-digit zip) → noise
      4. Standalone state/country name → noise (jurisdiction list items)
      5. Pure technical fragment (hex, IP) → noise (Google log examples)
      6. Everything else with 1+ sentences → KEEP (including numbered headings)
    """
    h = heading.strip()
    h_lower = h.lower().rstrip(".,;:")

    # Rule 1: no content = not a real section
    if sentence_count <= 0:
        return True

    # Rule 2: email address
    if "@" in h:
        return True

    # Rule 3: postal address (5-digit zip code)
    if re.search(r"\d{5}", h):
        return True

    # Rule 4: standalone geographic name
    if h_lower in US_STATES or h_lower in COUNTRIES:
        return True

    # Rule 5: pure technical fragments (hex hashes, IP addresses)
    if re.match(r"^[a-f0-9]{6,}$", h_lower):
        return True
    if re.match(r"^[\d\.]+$", h):  # pure numbers like "67.89"
        return True

    # Everything else is a real section — keep it
    return False


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: INGESTION & CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

def simple_token_count(text: str) -> int:
    """Approximate token count: words × 1.3 for English legal text."""
    return int(len(text.split()) * 1.3)


def chunk_sentences_with_sections(sentences: list, clean_sections: list,
                                  max_tokens: int) -> list:
    """
    Group sentences into chunks of max_tokens, each labeled with its section.
    Uses first-80-char matching to map sentences to their parent section.
    """
    # Build sentence → heading lookup from cleaned sections
    sent_to_heading = {}
    for section in clean_sections:
        for s in section.get("sentences", []):
            key = s[:80].strip().lower()
            sent_to_heading[key] = section["heading"]

    default_heading = clean_sections[0]["heading"] if clean_sections else "General"
    chunks = []
    current_sents = []
    current_count = 0
    current_heading = default_heading

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Look up section heading
        key = sent[:80].strip().lower()
        if key in sent_to_heading:
            current_heading = sent_to_heading[key]

        n = simple_token_count(sent)
        if n > max_tokens:
            if current_sents:
                chunks.append({"text": " ".join(current_sents),
                               "section_heading": current_heading})
                current_sents, current_count = [], 0
            chunks.append({"text": sent, "section_heading": current_heading})
        elif current_count + n > max_tokens:
            chunks.append({"text": " ".join(current_sents),
                           "section_heading": current_heading})
            current_sents, current_count = [sent], n
        else:
            current_sents.append(sent)
            current_count += n

    if current_sents:
        chunks.append({"text": " ".join(current_sents),
                       "section_heading": current_heading})
    return chunks


def load_json_reports(reports_dir: str) -> tuple:
    """
    Load all JSON reports. Returns (DataFrame, supply_chain_data).

    For each company:
      1. Reads metadata (company, category, policy_date)
      2. Filters noise sections via is_noise_section()
      3. Reads semantic_supply_chain entities (Nemotron, already extracted)
      4. Chunks sentences with section labels
      5. Stores in DataFrame rows
    """
    path = Path(reports_dir)
    files = sorted(path.glob("*.json"))
    files = [f for f in files if "collection_log" not in f.name]
    if not files:
        raise FileNotFoundError(f"No JSON files in: {reports_dir}")

    print(f"\n  Found {len(files)} JSON files in '{reports_dir}'")
    rows = []
    supply_chain_data = {}

    for fp in files:
        with open(fp, encoding="utf-8") as f:
            data = json.load(f)

        company  = data.get("company", fp.stem)
        category = data.get("category", "Unknown")
        url      = data.get("source_url", "")
        analysis = data.get("analysis", {})

        # ── Metadata ───────────────────────────────────────────
        policy_date  = analysis.get("policy_date") or "Unknown"
        grade_level  = analysis.get("readability", {}).get("grade_level")
        reading_ease = analysis.get("readability", {}).get("reading_ease")
        consent_model = analysis.get("consent_model", {}).get("model", "Unknown")

        # ── Supply chain (Nemotron, already extracted) ─────────
        sc = analysis.get("semantic_supply_chain", {})
        entities = sc.get("entities", []) if isinstance(sc, dict) else []
        supply_chain_data[company] = entities

        # ── Clean sections ─────────────────────────────────────
        raw_sections = analysis.get("sections", [])
        clean_sections = [s for s in raw_sections
                          if not is_noise_section(
                              s.get("heading", ""),
                              s.get("sentence_count", 0))]
        noise_removed = len(raw_sections) - len(clean_sections)

        # ── Get sentences ──────────────────────────────────────
        sentences = analysis.get("sentences", [])
        if not sentences:
            full = analysis.get("full_text", "")
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full)
                         if len(s.strip()) > 20]
        if not sentences:
            print(f"  ⚠  {company}: no text, skipping")
            continue

        # ── Chunk with section labels ──────────────────────────
        chunk_dicts = chunk_sentences_with_sections(
            sentences, clean_sections, MAX_TOKENS)
        chunk_dicts = [c for c in chunk_dicts
                       if simple_token_count(c["text"]) >= MIN_CHUNK_TOKENS
                       and re.search(r"[a-zA-Z]{4,}", c["text"])]

        for idx, cd in enumerate(chunk_dicts):
            rows.append({
                "company":         company,
                "category":        category,
                "url":             url,
                "policy_date":     policy_date,
                "chunk_id":        f"{company}_{idx:04d}",
                "chunk_index":     idx,
                "total_chunks":    len(chunk_dicts),
                "section_heading": cd["section_heading"],
                "text":            cd["text"],
                "token_count":     simple_token_count(cd["text"]),
                "grade_level":     grade_level,
                "reading_ease":    reading_ease,
                "consent_model":   consent_model,
                "section_count":   len(clean_sections),
            })

        sc_count = len(entities)
        print(f"  ✓  {company:20} {len(chunk_dicts):3} chunks  "
              f"{len(clean_sections):2} sects (-{noise_removed} noise)  "
              f"{sc_count} 3rd parties")

    df = pd.DataFrame(rows)
    print(f"\n  Total: {len(df)} chunks, {df['company'].nunique()} companies")
    return df, supply_chain_data


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2A: MODULE A — INTENT CLASSIFICATION (HuggingFace API)
# ══════════════════════════════════════════════════════════════════════════════
# bart-large-mnli: dedicated NLI model with calibrated probability scores.
# Nemotron is better for extraction (Module B), HF is better for classification.

def load_intent_classifier(hf_token=None):
    """Verify HuggingFace API is reachable. Returns config dict or None."""
    token = hf_token or os.environ.get("HF_TOKEN", "")
    if not token:
        print("  No HuggingFace token — intent classification skipped.")
        print("  Free token: https://huggingface.co/settings/tokens")
        return None
    print("  Testing HuggingFace API...")
    try:
        r = req.post(HF_API_URL,
                     headers={"Authorization": f"Bearer {token}",
                              "x-wait-for-model": "true"},
                     json={"inputs": "test", "parameters": {"candidate_labels": ["test"]}},
                     timeout=60)
        if r.status_code in (200, 503):
            print("  ✓ HuggingFace API connected (bart-large-mnli)")
            return {"token": token, "url": HF_API_URL}
        print(f"  ⚠ API returned {r.status_code}")
        return None
    except Exception as e:
        print(f"  ⚠ Unreachable: {e}")
        return None


def classify_intent(text: str, classifier: dict) -> tuple:
    """
    Returns (top_label, confidence). Retries on 503/429.

    Fix 1: Unwrap list BEFORE checking keys — HF router wraps response in
           a list: [{...}], which caused all chunks to return Parse Error.
    Fix 2: Add x-wait-for-model header so cold-start returns result, not 503.
    """
    payload = {"inputs": text[:1500], "parameters": {"candidate_labels": INTENT_LABELS}}
    headers = {
        "Authorization": f"Bearer {classifier['token']}",
        "x-wait-for-model": "true",   # Fix 2: wait for cold start
    }

    for attempt in range(6):
        try:
            r = req.post(classifier["url"], headers=headers, json=payload, timeout=60)

            if r.status_code == 200:
                data = r.json()

                # Fix 1: Unwrap list FIRST before any key checks
                # HF router returns: [{"labels": [...], "scores": [...]}]
                if isinstance(data, list) and len(data) > 0:
                    data = data[0]

                # Handle both response formats
                if isinstance(data, dict):
                    if "labels" in data and "scores" in data:
                        # Standard zero-shot format
                        return data["labels"][0], round(data["scores"][0], 4)
                    elif "label" in data and "score" in data:
                        # Single-label format from some router versions
                        return data["label"], round(data["score"], 4)

                print(f"      [!] Unrecognised API format: {str(data)[:120]}")
                return "Parse Error", 0.0

            elif r.status_code == 503:
                wait = 20
                try:
                    err_data = r.json()
                    if isinstance(err_data, dict) and "estimated_time" in err_data:
                        wait = min(int(err_data["estimated_time"]), 60)
                except Exception:
                    pass
                print(f"      [!] Model warming up, waiting {wait}s... ({attempt+1}/6)")
                time.sleep(wait)

            elif r.status_code == 429:
                print(f"      [!] Rate limited. Sleeping 60s... ({attempt+1}/6)")
                time.sleep(60)

            else:
                print(f"      [!] API Error {r.status_code}: {r.text[:80]}")
                return "API Error", 0.0

        except Exception as e:
            print(f"      [!] Network error: {str(e)[:60]}. Retrying in 5s.")
            time.sleep(5)

    return "Timeout", 0.0


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2B: MODULE B — SUPPLY CHAIN (READ FROM JSON)
# ══════════════════════════════════════════════════════════════════════════════
# Nemotron already extracted entities during batch_collect.py.
# This module just reads them — no API calls, no re-extraction.


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2C: MODULE C — DARK PATTERN DETECTION (spaCy, parallel)
# ══════════════════════════════════════════════════════════════════════════════
# Three signals: passive_ratio, vague_density, avg_sent_depth → dark_score.

def dark_pattern_score(text: str, nlp) -> dict:
    """Compute dark-pattern signals from syntactic analysis."""
    doc = nlp(text)
    tokens = [t for t in doc if not t.is_space]
    n_tokens = max(len(tokens), 1)

    # Passive voice
    passive_count = sum(1 for t in tokens if t.dep_ in PASSIVE_DEPS)
    passive_ratio = round(passive_count / n_tokens, 4)

    # Vague language per 100 words
    vague_hits = len(VAGUE_PATTERNS.findall(text))
    vague_density = round(vague_hits / (n_tokens / 100), 4)

    # Syntax tree depth
    def _tree_depth(token, visited=None):
        """Recursive tree depth — defined once, reused across all sentences."""
        if visited is None: visited = set()
        if token.i in visited: return 0
        visited.add(token.i)
        ch = list(token.children)
        return 0 if not ch else 1 + max(_tree_depth(c, visited) for c in ch)

    depths = []
    for sent in doc.sents:
        roots = [t for t in sent if t.dep_ == "ROOT"]
        if not roots:
            continue
        depths.append(_tree_depth(roots[0]))

    avg_depth = round(sum(depths) / max(len(depths), 1), 2)

    return {
        "passive_ratio": passive_ratio,
        "vague_density": vague_density,
        "avg_sent_depth": avg_depth,
        "dark_score": round(
            (passive_ratio * 40 + min(vague_density, 10) * 4
             + min(avg_depth, 10) * 4) / 10, 3)
    }


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: ORCHESTRATION (Module A sequential, Module C parallel)
# ══════════════════════════════════════════════════════════════════════════════

# No multiprocessing workers needed — spaCy's nlp.pipe() handles batching
# internally and is both thread-safe and Windows-compatible.


def run_pipeline(df, run_intent, intent_clf, spacy_model="en_core_web_sm"):
    """Run Module A (if enabled) + Module C across all chunks."""

    # ── Module A: Intent ───────────────────────────────────────
    if run_intent and intent_clf:
        n = len(df)
        print(f"\n  Module A: Intent classification ({n} chunks via HF API)...")
        intents, scores = [], []
        for i, text in enumerate(df["text"], 1):
            label, score = classify_intent(text, intent_clf)
            intents.append(label)
            scores.append(score)
            if i % 20 == 0 or i == n:
                print(f"    {i}/{n}")
        df["intent_label"] = intents
        df["intent_score"] = scores
    else:
        df["intent_label"] = "skipped"
        df["intent_score"] = 0.0

    # ── Module C: Dark patterns via spaCy nlp.pipe() ────────────
    # nlp.pipe() is spaCy's built-in batch processor. It is:
    #   - Single-process (no Windows spawn/deadlock issues)
    #   - More memory-efficient than multiprocessing (one model instance)
    #   - Faster than sequential processing for large corpora
    #   - Thread-safe and cross-platform
    n_chunks = len(df)
    batch_size = max(1, n_chunks // N_WORKERS)   # mirror intended parallelism
    print(f"\n  Module C: Dark pattern analysis via nlp.pipe() "
          f"(batch_size={batch_size}, {n_chunks} chunks)...")

    nlp = spacy.load(spacy_model)

    # Disable unused pipeline components for speed
    texts = df["text"].tolist()
    chunk_ids = df["chunk_id"].tolist()

    results = []
    with nlp.disable_pipes("ner", "lemmatizer") if "ner" in nlp.pipe_names else nlp.select_pipes(enable=["tok2vec", "tagger", "parser", "attribute_ruler", "senter"] if "parser" in nlp.pipe_names else []):
        pass

    # Process all chunks in batches
    for i, doc in enumerate(nlp.pipe(texts, batch_size=batch_size)):
        tokens = [t for t in doc if not t.is_space]
        n_tokens = max(len(tokens), 1)

        # Passive voice
        passive_count = sum(1 for t in tokens if t.dep_ in PASSIVE_DEPS)
        passive_ratio = round(passive_count / n_tokens, 4)

        # Vague language density
        vague_hits = len(VAGUE_PATTERNS.findall(texts[i]))
        vague_density = round(vague_hits / (n_tokens / 100), 4)

        # Syntax tree depth
        def _tree_depth(token, visited=None):
            if visited is None: visited = set()
            if token.i in visited: return 0
            visited.add(token.i)
            ch = list(token.children)
            return 0 if not ch else 1 + max(_tree_depth(c, visited) for c in ch)

        depths = []
        for sent in doc.sents:
            roots = [t for t in sent if t.dep_ == "ROOT"]
            if roots:
                depths.append(_tree_depth(roots[0]))

        avg_depth = round(sum(depths) / max(len(depths), 1), 2)
        dark = round(
            (passive_ratio * 40 + min(vague_density, 10) * 4
             + min(avg_depth, 10) * 4) / 10, 3)

        results.append({
            "chunk_id":      chunk_ids[i],
            "passive_ratio": passive_ratio,
            "vague_density": vague_density,
            "avg_sent_depth":avg_depth,
            "dark_score":    dark,
        })

        if (i + 1) % 50 == 0 or (i + 1) == n_chunks:
            print(f"    {i+1}/{n_chunks} chunks processed")

    rdf = pd.DataFrame(results).set_index("chunk_id")
    df = df.set_index("chunk_id")
    for col in ["passive_ratio", "vague_density", "avg_sent_depth", "dark_score"]:
        df[col] = rdf[col]
    df = df.reset_index()
    print(f"    Done: {len(df)} chunks")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4: AGGREGATION
# ══════════════════════════════════════════════════════════════════════════════
# Output columns are mapped to essay sections:
#   grade_level, reading_ease          → Finding 1 (readability)
#   consent_model                      → Finding 2 (opt-in/opt-out)
#   n_buyers, n_processors             → Finding 3 (supply chain)
#   avg_dark_score, avg_vague_density   → Finding 4 (dark patterns)

# ── Supply-chain interpretation helpers (deep schema) ─────────────────────────
# The Nemotron entities now carry: third_party_name, relationship_type
# (buyer|processor|ai_trainer|joint_controller|unknown), data_categories_shared
# (list[str]), stated_purpose (str), cross_border_transfer (bool|None).

# A recipient is "generic" (a disclosed *category*, not a named org) if its
# surface form is a common-noun phrase with no proper-noun/brand signal.
_GENERIC_HEAD = re.compile(
    r"\b(partners?|service providers?|sub[\s-]?processors?|processors?|"
    r"vendors?|suppliers?|affiliates?|subsidiar(?:y|ies)|advertisers?|"
    r"ad networks?|third part(?:y|ies)|other companies|other parties|"
    r"data brokers?|analytics providers?|contractors?|agents?|"
    r"consultants?|auditors?|lawyers?|authorities|recipients?)\b",
    re.IGNORECASE,
)


def _is_named_entity(name: str) -> bool:
    """True if the recipient is a specific named org, not a generic category."""
    n = (name or "").strip()
    if not n:
        return False
    low = n.lower()
    # Generic if the whole phrase is a common-noun category and carries no
    # internal capitalized brand token (e.g. "advertising partners" → generic,
    # "Google Analytics" → named, "Stripe, Inc." → named).
    has_brand_token = any(
        tok[:1].isupper() and tok.lower() not in _STOP_CAPS and len(tok) > 1
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9&.\-]+", n)
    )
    if _GENERIC_HEAD.search(low) and not has_brand_token:
        return False
    return has_brand_token


_STOP_CAPS = {
    "the", "a", "an", "our", "your", "their", "its", "we", "us", "other",
    "such", "certain", "various", "third", "party", "parties", "and", "or",
}

# Map free-text data-category strings → high-signal ML buckets.
_CAT_BUCKETS = {
    "shares_location": re.compile(
        r"\b(location|gps|geo[\s-]?location|geographic|ip address|whereabouts)\b", re.I),
    "shares_financial": re.compile(
        r"\b(payment|credit card|debit card|billing|financial|bank|"
        r"purchase|transaction|salary|income)\b", re.I),
    "shares_contact_identity": re.compile(
        r"\b(name|email|phone|address|contact|identifier|account|"
        r"username|date of birth|ssn|social security|passport|government id)\b", re.I),
    "shares_behavioral": re.compile(
        r"\b(browsing|behavior|behaviour|activity|usage|interest|preference|"
        r"cookie|clickstream|history|interaction|search quer|viewing)\b", re.I),
    "shares_sensitive": re.compile(
        r"\b(biometric|health|medical|genetic|race|ethnic|religio|"
        r"sexual|precise location|political|union|children|minor)\b", re.I),
}

_VAGUE_PURPOSE = re.compile(
    r"\b(not stated|n/?a|unknown|business purpose|legitimate interest|"
    r"as described|as needed|various|other purpose|improve(?:ment)? "
    r"(?:of )?(?:our )?services?|operational)\b", re.I)


def _purpose_is_vague(purpose: str) -> bool:
    p = (purpose or "").strip()
    if not p or len(p) < 4:
        return True
    return bool(_VAGUE_PURPOSE.search(p) or VAGUE_PATTERNS.search(p))


def _entity_rows(company, category, entities):
    """Long-form: one row per (company, third party) for ML training."""
    rows = []
    for e in entities:
        name = (e.get("third_party_name") or "").strip()
        if not name:
            continue
        cats = [c for c in (e.get("data_categories_shared") or []) if c]
        rows.append({
            "company":               company,
            "category":              category,
            "third_party_name":      name,
            "is_named_entity":       _is_named_entity(name),
            "relationship_type":     e.get("relationship_type", "unknown"),
            "n_data_categories":     len(cats),
            "data_categories_shared": " | ".join(cats),
            "stated_purpose":        (e.get("stated_purpose") or "not stated").strip(),
            "purpose_is_vague":      _purpose_is_vague(e.get("stated_purpose", "")),
            "cross_border_transfer": e.get("cross_border_transfer"),
        })
    return rows


def aggregate(df, supply_chain_data):
    """Collapse chunks to one row per company with the deep supply-chain block."""
    records = []
    for company, grp in df.groupby("company"):
        ic = grp["intent_label"].value_counts(normalize=True)
        category = grp["category"].iloc[0]

        entities = supply_chain_data.get(company, []) or []
        total = len(entities)

        def _rel(t):
            return [e for e in entities if e.get("relationship_type") == t]
        buyers   = _rel("buyer")
        procs    = _rel("processor")
        trainers = _rel("ai_trainer")
        jointc   = _rel("joint_controller")
        unkn     = _rel("unknown")

        named   = [e for e in entities if _is_named_entity(e.get("third_party_name", ""))]
        n_named = len(named)
        n_generic = total - n_named

        # Distinct data categories + high-signal exposure flags
        all_cats, bucket_hit = [], {k: False for k in _CAT_BUCKETS}
        for e in entities:
            for c in (e.get("data_categories_shared") or []):
                if c and c.strip():
                    all_cats.append(c.strip().lower())
                    for b, rx in _CAT_BUCKETS.items():
                        if rx.search(c):
                            bucket_hit[b] = True
        distinct_cats = sorted(set(all_cats))

        # Purpose transparency
        n_vague = sum(1 for e in entities
                      if _purpose_is_vague(e.get("stated_purpose", "")))
        pct_vague = round(100 * n_vague / total, 1) if total else 0.0

        # Cross-border
        n_xborder = sum(1 for e in entities
                        if e.get("cross_border_transfer") is True)
        pct_xborder = round(100 * n_xborder / total, 1) if total else 0.0

        # No-data-detail ratio (only generic "personal information" or empty)
        n_nodata = sum(
            1 for e in entities
            if not [c for c in (e.get("data_categories_shared") or [])
                    if c and c.strip().lower() not in
                    ("personal information", "personal data", "data", "information")]
        )

        # Composite headline: 0 = fully auditable, 100 = fully opaque.
        if total == 0:
            opacity = 100.0
        else:
            generic_r = n_generic / total
            vague_r   = n_vague / total
            unknown_r = len(unkn) / total
            nodata_r  = n_nodata / total
            opacity = round(100 * (0.35 * generic_r + 0.30 * vague_r
                                   + 0.20 * unknown_r + 0.15 * nodata_r), 1)

        monetization = bool(buyers or trainers)
        named_ratio  = round(n_named / total, 3) if total else 0.0

        sections = sorted(grp["section_heading"].unique())

        records.append({
            "company":              company,
            "category":             category,
            "consent_model":        grp["consent_model"].iloc[0],
            "policy_date":          grp["policy_date"].iloc[0],
            "grade_level":          grp["grade_level"].dropna().iloc[0] if not grp["grade_level"].dropna().empty else None,
            "reading_ease":         grp["reading_ease"].dropna().iloc[0] if not grp["reading_ease"].dropna().empty else None,
            "n_chunks":             len(grp),
            "n_sections":           grp["section_count"].iloc[0],
            "top_sections":         "; ".join(sections[:5]),
            "pct_advertising":      round(ic.get("Targeted Advertising", 0)*100, 1),
            "pct_functionality":    round(ic.get("Primary App Functionality", 0)*100, 1),
            "pct_ai_training":      round(ic.get("AI Model Training", 0)*100, 1),
            "pct_third_party_sale": round(ic.get("Third-Party Data Sale", 0)*100, 1),
            "pct_security":         round(ic.get("Security and Fraud Prevention", 0)*100, 1),
            "avg_dark_score":       round(grp["dark_score"].mean(), 3),
            "avg_passive_ratio":    round(grp["passive_ratio"].mean(), 4),
            "avg_vague_density":    round(grp["vague_density"].mean(), 3),
            "avg_sent_depth":       round(grp["avg_sent_depth"].mean(), 2),
            # ── Deep supply-chain block ──────────────────────────
            "total_third_parties":      total,
            "n_buyers":                 len(buyers),
            "n_processors":             len(procs),
            "n_ai_trainers":            len(trainers),
            "n_joint_controllers":      len(jointc),
            "n_unknown_rel":            len(unkn),
            "n_named_entities":         n_named,
            "n_generic_entities":       n_generic,
            "named_entity_ratio":       named_ratio,
            "distinct_data_categories": len(distinct_cats),
            "shares_location":          bucket_hit["shares_location"],
            "shares_financial":         bucket_hit["shares_financial"],
            "shares_contact_identity":  bucket_hit["shares_contact_identity"],
            "shares_behavioral":        bucket_hit["shares_behavioral"],
            "shares_sensitive":         bucket_hit["shares_sensitive"],
            "pct_vague_purpose":        pct_vague,
            "monetization_intent":      monetization,
            "n_cross_border":           n_xborder,
            "pct_cross_border":         pct_xborder,
            "supply_chain_opacity_score": opacity,
            "data_categories_top":      "; ".join(distinct_cats[:12]),
            "third_party_names":        "; ".join(
                sorted({e["third_party_name"] for e in named})[:15]),
        })

    return pd.DataFrame(records).sort_values("category")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES (3 total — one per essay finding)
# ══════════════════════════════════════════════════════════════════════════════

def make_figures(agg, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    n = len(agg)

    # ── Fig 1: Readability ─────────────────────────────────────
    cm = (agg.groupby("category")[["grade_level", "reading_ease"]]
          .mean().round(1).reset_index()
          .sort_values("grade_level", ascending=True))
    colors = [CAT_COLORS.get(c, "#888") for c in cm["category"]]

    fig1 = make_subplots(rows=1, cols=2,
        subplot_titles=("Grade Level", "Reading Ease"),
        horizontal_spacing=0.12)
    fig1.add_trace(go.Bar(x=cm["grade_level"], y=cm["category"],
        orientation="h", marker_color=colors, showlegend=False,
        text=cm["grade_level"], textposition="outside"), row=1, col=1)
    # Fix 4: add_vline does not support row/col — use shapes on xref/yref instead
    fig1.add_shape(type="line", x0=8, x1=8, y0=-0.5, y1=len(cm)-0.5,
        line=dict(dash="dash", color="red", width=1.5),
        xref="x1", yref="y1")
    fig1.add_annotation(x=8, y=len(cm)-0.5, text="Avg adult (grade 8)",
        showarrow=False, xref="x1", yref="y1",
        font=dict(color="red", size=10), xanchor="left")
    fig1.add_trace(go.Bar(x=cm["reading_ease"], y=cm["category"],
        orientation="h", marker_color=colors, showlegend=False,
        text=cm["reading_ease"], textposition="outside"), row=1, col=2)
    fig1.add_shape(type="line", x0=60, x1=60, y0=-0.5, y1=len(cm)-0.5,
        line=dict(dash="dash", color="green", width=1.5),
        xref="x2", yref="y2")
    fig1.add_annotation(x=60, y=len(cm)-0.5, text="Plain English (60)",
        showarrow=False, xref="x2", yref="y2",
        font=dict(color="green", size=10), xanchor="left")
    fig1.update_layout(
        title_text=f"Privacy Policy Readability ({n} Organizations)",
        height=400, width=900, plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=100, r=60, t=80, b=40))
    fig1.write_html(os.path.join(output_dir, "fig1_readability.html"))
    fig1.write_image(os.path.join(output_dir, "fig1_readability.png"), scale=2)
    print("  ✓ fig1_readability")

    # ── Fig 2: Consent Model ───────────────────────────────────
    order = ["Primarily Opt-Out", "Mixed", "Unclear", "Primarily Opt-In"]
    cc = agg.groupby(["category", "consent_model"]).size().reset_index(name="count")
    cp = (cc.pivot(index="category", columns="consent_model", values="count")
          .fillna(0).reindex(columns=order, fill_value=0))
    cmap = {"Primarily Opt-Out": "#d62728", "Mixed": "#ff7f0e",
            "Unclear": "#aec7e8", "Primarily Opt-In": "#2ca02c"}
    fig2 = go.Figure()
    for label in order:
        if label in cp.columns:
            fig2.add_trace(go.Bar(name=label, x=cp.index, y=cp[label],
                marker_color=cmap[label]))
    fig2.update_layout(barmode="stack",
        title="Consent Model Distribution by Category",
        xaxis_title="Category", yaxis_title="Companies",
        height=420, width=750, plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(title="Consent Model"),
        margin=dict(l=60, r=40, t=80, b=60))
    fig2.write_html(os.path.join(output_dir, "fig2_consent.html"))
    fig2.write_image(os.path.join(output_dir, "fig2_consent.png"), scale=2)
    print("  ✓ fig2_consent")

    # ── Fig 3: Dark Pattern vs Complexity ──────────────────────
    fig3 = px.scatter(agg,
        x="grade_level", y="avg_dark_score", color="category",
        color_discrete_map=CAT_COLORS,
        size="total_third_parties", size_max=30,
        hover_name="company",
        hover_data={"consent_model": True, "reading_ease": True,
                    "n_buyers": True, "n_processors": True,
                    "supply_chain_opacity_score": True,
                    "pct_vague_purpose": True},
        labels={"grade_level": "Reading Grade Level",
                "avg_dark_score": "Dark Pattern Score"},
        title=f"Complexity vs Dark Patterns ({n} companies)"
              "<br><sup>Bubble size = third parties (Nemotron deep extraction)</sup>")
    fig3.update_layout(height=480, width=800,
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=60, r=40, t=100, b=60))
    fig3.write_html(os.path.join(output_dir, "fig3_darkpattern.html"))
    fig3.write_image(os.path.join(output_dir, "fig3_darkpattern.png"), scale=2)
    print("  ✓ fig3_darkpattern")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Privacy Policy NLP Pipeline")
    parser.add_argument("--reports-dir", default=REPORTS_DIR)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR)
    parser.add_argument("--skip-intent", action="store_true")
    parser.add_argument("--hf-token",    default=None,
        help="HuggingFace token. Or set HF_TOKEN env var.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print("=" * 65)
    print("  Privacy Policy NLP Pipeline (Final)")
    print(f"  Workers  : {N_WORKERS}")
    print(f"  Reports  : {args.reports_dir}")
    print(f"  Module A : {'SKIP' if args.skip_intent else 'HF bart-large-mnli'}")
    print(f"  Module B : Read from JSON (Nemotron)")
    print(f"  Module C : spaCy dark patterns")
    print("=" * 65)

    # Phase 1
    print("\n[Phase 1] Ingesting + cleaning...")
    df, sc_data = load_json_reports(args.reports_dir)
    df.to_csv(os.path.join(args.output_dir, "chunks.csv"),
              index=False, encoding="utf-8-sig")

    # Phase 2
    print("\n[Phase 2] Analysis...")
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        raise OSError("Run: python -m spacy download en_core_web_sm")

    intent_clf = None
    if not args.skip_intent:
        intent_clf = load_intent_classifier(args.hf_token)

    df = run_pipeline(df, not args.skip_intent, intent_clf)
    df.drop(columns=["text"]).to_csv(
        os.path.join(args.output_dir, "chunks_analyzed.csv"),
        index=False, encoding="utf-8-sig")

    # Phase 3
    print("\n[Phase 3] Aggregating...")
    agg = aggregate(df, sc_data)
    agg.to_csv(os.path.join(args.output_dir, "company_aggregated.csv"),
               index=False, encoding="utf-8-sig")

    # Entity-level long-form sidecar — one row per (company, third party).
    # This is the ML-training-friendly grain: relationship, data categories,
    # purpose vagueness and cross-border flag are preserved per recipient
    # instead of being collapsed into company means.
    cat_by_company = (df.groupby("company")["category"].first().to_dict())
    detail_rows = []
    for company, ents in sc_data.items():
        detail_rows.extend(
            _entity_rows(company, cat_by_company.get(company, "Unknown"),
                         ents or []))
    detail_df = pd.DataFrame(detail_rows)
    detail_df.to_csv(os.path.join(args.output_dir, "supply_chain_detail.csv"),
                     index=False, encoding="utf-8-sig")
    print(f"  ✓ supply_chain_detail.csv ({len(detail_df)} third-party rows)")

    # Key findings for essay
    print("\n" + "=" * 65)
    print("  KEY FINDINGS FOR ESSAY")
    print("=" * 65)

    print("\n  [1] Readability:")
    print(agg.groupby("category")[["grade_level","reading_ease"]].mean().round(1).to_string())

    print("\n  [2] Consent:")
    print(agg["consent_model"].value_counts().to_string())

    print("\n  [3] Supply Chain (Nemotron deep extraction):")
    print(agg.groupby("category")[[
        "total_third_parties", "n_buyers", "n_processors", "n_ai_trainers",
        "named_entity_ratio", "pct_vague_purpose", "supply_chain_opacity_score"
    ]].mean().round(1).to_string())

    print("\n  [4] Dark Patterns:")
    print(agg.groupby("category")[["avg_dark_score","avg_vague_density"]].mean().round(2).to_string())

    # Figures
    print(f"\n[Figures] → {args.output_dir}/")
    try:
        make_figures(agg, args.output_dir)
    except Exception as e:
        print(f"  ⚠ {e}")

    print("\n" + "=" * 65)
    print("  Done. ")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()