# semantic_extractor.py
#
# Module B — NVIDIA Nemotron-120B structured supply-chain extraction.
#
# Reconstruction (2026-05-16):
#   * Deep auditor-grade Pydantic schema (data categories, stated purpose,
#     cross-border transfer, relationship typing incl. joint_controller).
#   * Whole-policy extraction: extract_supply_chain() now accepts the FULL
#     policy text, chunks it internally under Nemotron's context budget,
#     pre-filters chunks to third-party-relevant passages (token economy),
#     extracts per chunk, then de-duplicates and merges entities across
#     chunks into a single coherent supply chain.
#
#   The previous design was fed only analyzer.py's truncated examples[:3]
#   (~600 chars of a multi-thousand-word policy) by batch_collect.py, which
#   starved the 120B model and produced 0-1 entities per company. The feed
#   is now the full policy; this module does the heavy lifting safely.

import os
import re
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ── TEMP telemetry switch ─────────────────────────────────────────────────────
# Set SC_TRACE=1 to print pre-filter / chunking diagnostics. Off by default so
# production runs stay quiet. Remove this block + the _trace() calls once the
# pre-filter behaviour is confirmed.
SC_TRACE = os.environ.get("SC_TRACE", "") == "1"


def _trace(msg: str) -> None:
    if SC_TRACE:
        # ASCII-only: this repo runs on a GBK (Windows China) console that
        # raises UnicodeEncodeError on box-drawing / check glyphs.
        print(f"  [SC-TRACE] {msg}", flush=True)

# ==============================================================================
# 1. DEEP PYDANTIC SCHEMA (auditor-grade)
# ==============================================================================
# 'unknown' is an explicit member of relationship_type so the model always has
# a valid escape hatch — this prevents instructor validation retries/crashes on
# genuinely ambiguous passages instead of forcing a wrong 'processor' default.


class ThirdPartyEntity(BaseModel):
    third_party_name: str = Field(
        ...,
        description=(
            "The exact named entity or specific category "
            "(e.g., 'Google Analytics', 'advertising partners'). "
            "Do not extract the company itself."
        ),
    )
    relationship_type: Literal[
        "buyer", "processor", "ai_trainer", "joint_controller", "unknown"
    ] = Field(..., description="The legal/functional relationship.")
    data_categories_shared: List[str] = Field(
        ...,
        description=(
            "List of specific data types shared with this entity "
            "(e.g., 'IP address', 'location', 'browsing history')."
        ),
    )
    stated_purpose: str = Field(
        ...,
        description=(
            "The reason provided for sharing this data "
            "(e.g., 'cross-context behavioral advertising', 'cloud hosting')."
        ),
    )
    cross_border_transfer: Optional[bool] = Field(
        None,
        description=(
            "True if the policy explicitly mentions transferring data to this "
            "entity across international borders."
        ),
    )


class DataSupplyChain(BaseModel):
    entities: List[ThirdPartyEntity]


# ==============================================================================
# 2. LLM CLIENT
# ==============================================================================

MODEL_NAME = "nvidia/nemotron-3-super-120b-a12b"

client = instructor.from_openai(
    OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ.get(
            "NVIDIA_API_KEY",
            "***REMOVED***",
        ),
        # CRITICAL: without an explicit timeout the hosted Nemotron endpoint
        # can leave a request open indefinitely (observed: a single chunk
        # silently deadlocked the whole 33-company sweep with no exit code).
        # A per-request timeout converts that hang into a raised exception,
        # which _extract_one_chunk already degrades to [] — so one bad chunk
        # costs ~its-data, not the entire run. 3 transport retries with
        # backoff absorb transient 5xx/network blips loudly, not silently.
        # Timeout tuned from measured latency. Legitimate Nemotron-120B
        # structured extraction on a ~1100-word chunk runs ~50-120s (Equifax
        # 3-chunk traces took 139-360s total). The earlier 75s value was BELOW
        # legitimate latency and silently truncated 27 chunks corpus-wide
        # (Equifax 8→0 entities). 300s sits well above real latency yet still
        # catches a true infinite hang (the original deadlock was ~1 hour).
        # max_retries=1 keeps a genuinely-bad chunk bounded (~10-12 min) so it
        # degrades to [] without freezing the sweep.
        timeout=300.0,
        max_retries=1,
    )
)


# ==============================================================================
# 3. INTERNAL CHUNKING + RELEVANCE PRE-FILTER
# ==============================================================================
# Nemotron-3-super-120b has a large context window, but structured-output
# fidelity degrades and cost rises with very long inputs. We therefore window
# the policy into ~chunks and only spend tokens on chunks that actually contain
# third-party-sharing language.

# Words per chunk. ~1100 words ≈ ~1500 tokens of legal English — comfortably
# inside the model budget while keeping each extraction focused and accurate.
CHUNK_WORDS = 1100
# Sentence overlap between consecutive chunks so an entity introduced at a
# chunk boundary is not split away from its purpose/data clause.
CHUNK_OVERLAP_SENTS = 2

# A chunk is only sent to the LLM if it contains at least one of these signals.
RELEVANCE_PATTERN = re.compile(
    r"\b(shar(e|ed|ing)|disclos(e|ed|ing|ure)|provide[sd]?\s+to|"
    r"transfer(red|s|ring)?|sell|sold|sale|third[\s-]?part(y|ies)|"
    r"service provider|sub[\s-]?processor|processor|vendor|supplier|"
    r"partner|affiliate|subsidiar(y|ies)|advertis|analytics|"
    r"on our behalf|recipient|data broker|train(ing)?\s+(our|its|the)?\s*"
    r"(model|ai|llm)|process(es|ed|ing)?\s+(your|personal)\s+data)\b",
    re.IGNORECASE,
)

_SENT_SPLIT = re.compile(r"(?<=[.!?;:])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text or "") if s.strip()]
    return [p for p in parts if len(p) > 2]


def _chunk_policy(full_text: str) -> List[str]:
    """Window the full policy into overlapping, relevance-filtered chunks."""
    sentences = _split_sentences(full_text)
    if not sentences:
        return []

    chunks: List[str] = []
    cur: List[str] = []
    cur_words = 0

    for sent in sentences:
        w = len(sent.split())
        if cur and cur_words + w > CHUNK_WORDS:
            chunks.append(" ".join(cur))
            # carry an overlap tail into the next window
            cur = cur[-CHUNK_OVERLAP_SENTS:] if CHUNK_OVERLAP_SENTS else []
            cur_words = sum(len(s.split()) for s in cur)
        cur.append(sent)
        cur_words += w

    if cur:
        chunks.append(" ".join(cur))

    total = len(chunks)
    # Token economy: only keep chunks that actually discuss third parties.
    relevant, dropped = [], []
    for c in chunks:
        (relevant if RELEVANCE_PATTERN.search(c) else dropped).append(c)

    # Safety net: if the filter is too aggressive (e.g. unusual phrasing),
    # fall back to the full set rather than extracting nothing.
    fellback = not relevant
    sent = chunks if fellback else relevant

    _trace(f"policy words={len(full_text.split())}  "
           f"sentences={len(sentences)}")
    _trace(f"chunks generated   : {total}")
    _trace(f"dropped by prefilter: {len(dropped)}")
    _trace(f"sent to Nemotron   : {len(sent)}"
           + ("  (SAFETY-NET fallback: filter matched 0, sending ALL)"
              if fellback else ""))
    for i, c in enumerate(dropped):
        _trace(f"  - DROPPED chunk[{i}] ({len(c.split())}w): "
               f"{c[:160].strip()!r}")
    return sent


# ==============================================================================
# 4. CROSS-CHUNK MERGE / DE-DUPLICATION
# ==============================================================================

_GENERIC_NAME_NORMALIZE = re.compile(r"[^a-z0-9 ]")


def _norm_name(name: str) -> str:
    n = _GENERIC_NAME_NORMALIZE.sub("", (name or "").lower()).strip()
    n = re.sub(r"\s+", " ", n)
    # collapse trivial plural / article noise so "partners" == "partner"
    n = re.sub(r"\b(the|a|an|our|its|their)\b", "", n).strip()
    return re.sub(r"s\b", "", n)  # crude singularization for dedupe keys only


# relationship precedence when the same entity is classified differently in
# different chunks — the more privacy-significant role wins.
_REL_PRECEDENCE = {
    "ai_trainer": 5,
    "buyer": 4,
    "joint_controller": 3,
    "processor": 2,
    "unknown": 1,
}


def _merge_entities(raw: List[dict]) -> List[dict]:
    """Collapse duplicate entities seen across multiple chunks."""
    merged: dict = {}
    for e in raw:
        name = (e.get("third_party_name") or "").strip()
        if not name:
            continue
        key = (_norm_name(name), )
        rel = e.get("relationship_type", "unknown") or "unknown"
        cats = [c.strip() for c in (e.get("data_categories_shared") or []) if c and c.strip()]
        purpose = (e.get("stated_purpose") or "").strip()
        xborder = e.get("cross_border_transfer", None)

        if key not in merged:
            merged[key] = {
                "third_party_name": name,
                "relationship_type": rel,
                "data_categories_shared": [],
                "stated_purpose": purpose,
                "cross_border_transfer": xborder,
            }
        m = merged[key]

        # Prefer the longer / more specific surface name.
        if len(name) > len(m["third_party_name"]):
            m["third_party_name"] = name

        # Highest-precedence relationship wins.
        if _REL_PRECEDENCE.get(rel, 0) > _REL_PRECEDENCE.get(m["relationship_type"], 0):
            m["relationship_type"] = rel

        # Union of data categories (case-insensitive, order-preserving).
        seen = {c.lower() for c in m["data_categories_shared"]}
        for c in cats:
            if c.lower() not in seen:
                m["data_categories_shared"].append(c)
                seen.add(c.lower())

        # Keep the most informative stated purpose.
        if len(purpose) > len(m["stated_purpose"]):
            m["stated_purpose"] = purpose

        # cross_border_transfer: any explicit True wins over None/False.
        if xborder is True:
            m["cross_border_transfer"] = True
        elif xborder is False and m["cross_border_transfer"] is None:
            m["cross_border_transfer"] = False

    return list(merged.values())


# ==============================================================================
# 5. SYSTEM PROMPT (auditor-grade)
# ==============================================================================

def _system_prompt(company_name: str) -> str:
    return (
        "You are a senior data-protection auditor performing a forensic review "
        f"of {company_name}'s privacy policy to map its COMPLETE data supply "
        "chain. You parse deliberately vague legal language with rigor and you "
        "never let hedging phrasing ('may', 'such as', 'including but not "
        "limited to', 'trusted partners') cause you to omit a disclosed "
        "recipient.\n\n"
        "TASK\n"
        "From the passage below, extract EVERY distinct external recipient of "
        "personal data — whether named (e.g. 'Google Analytics', 'Stripe') or "
        "described only as a category (e.g. 'advertising partners', 'service "
        "providers', 'affiliates'). Extract each one separately; do not collapse "
        "distinct recipients into a single generic blob.\n\n"
        "RELATIONSHIP CLASSIFICATION — DEFINITIONAL TEST\n"
        "Apply this decision test to EACH recipient. The pivotal question is: "
        "WHO controls the data once the recipient has it?\n"
        "- 'processor': a vendor that may use the data ONLY to perform a "
        "service FOR the company, under the company's instructions, and may "
        "NOT use it for its own goals. Examples: AWS/cloud hosting, payment "
        "processing, email/SMS delivery, customer-support tooling, basic "
        "first-party analytics, security scanning. Litmus: if the recipient "
        "would have to delete the data when the contract ends and cannot "
        "exploit it independently, it is a processor.\n"
        "- 'buyer': receives the data to make ITS OWN independent decisions, "
        "build or enrich ITS OWN profiles/models, monetize it, or sell/share "
        "it onward. THIS EXPLICITLY INCLUDES: financial institutions or "
        "lenders evaluating creditworthiness, insurers underwriting risk, "
        "employers screening applicants, data brokers / data resellers / "
        "'third-party data providers', advertisers, ad networks, marketers, "
        "and any recipient performing credit, risk, fraud-scoring, or "
        "background-check decisions for THEIR OWN business. If the recipient "
        "uses the data for its own determinations, it is a 'buyer', NOT a "
        "'processor' — even if the policy calls them a 'customer', 'client', "
        "or 'partner'.\n"
        "- 'ai_trainer': receives data specifically to train, fine-tune, or "
        "improve machine-learning / foundation / generative-AI models.\n"
        "- 'joint_controller': the recipient and the company TOGETHER decide "
        "the purposes/means of processing — co-branded services, ad "
        "co-controllership, shared-decision data exchanges (e.g. a credit "
        "bureau and a furnisher jointly determining reporting).\n"
        "- 'unknown': a recipient is clearly disclosed but the passage gives "
        "genuinely insufficient context to assign any role above. Use this "
        "instead of guessing. NEVER invent a relationship. Do NOT use "
        "'processor' as a lazy default — 'unknown' is the correct fallback "
        "when the role is unclear.\n\n"
        "CONTEXTUAL EXCEPTION (DATA-BROKER / BUREAU BUSINESS MODEL)\n"
        f"If {company_name}'s core business involves selling data, profiles, "
        "scores, or reports (e.g. credit bureaus, consumer-reporting agencies, "
        "data brokers, identity/marketing-data vendors) AND the recipient is "
        "described as a 'customer', 'client', 'subscriber', 'financial "
        "institution', or an entity receiving 'reports', 'scores', 'data "
        "products', or 'services', you MUST classify them as 'buyer'. In this "
        "business model such recipients are the DESTINATION node of the data — "
        "they receive it for their OWN independent business use (lending, "
        "underwriting, screening, marketing). Do NOT downgrade these to "
        "'unknown' merely because the passage is terse: the business model "
        "itself supplies the missing context. This exception OVERRIDES the "
        "'unknown' fallback, but never overrides a clearer 'processor' / "
        "'joint_controller' / 'ai_trainer' signal in the passage.\n\n"
        "STRICT LOCAL ATTRIBUTION (ANTI-LAZINESS — CRITICAL)\n"
        "Only extract data categories that are EXPLICITLY linked to the "
        "SPECIFIC recipient in the CURRENT text passage. If the text says "
        "'we share data with X' but does not specify WHAT data goes to X, you "
        "MUST set data_categories_shared to an empty list []. If the passage "
        "only loosely implies a type, you may use ['unspecified']. Do NOT "
        "copy a global enumeration (e.g. a CCPA 'categories of personal "
        "information we collect' list, or a master data-types section) onto "
        "every recipient. A recipient gets a category ONLY if the passage "
        "ties that category to THAT recipient. Identical long category lists "
        "repeated across multiple recipients is a forensic error and will be "
        "rejected.\n\n"
        "FIELD RULES\n"
        "- data_categories_shared: SPECIFIC data types the passage attributes "
        "to THIS recipient, in the policy's own terms (e.g. 'IP address', "
        "'precise location', 'credit history', 'SSN'). Empty list [] if the "
        "passage does not say what this recipient receives.\n"
        "- stated_purpose: the passage's stated reason for THIS sharing, "
        "tightly paraphrased (e.g. 'creditworthiness evaluation', 'cloud "
        "hosting', 'fraud prevention'). If none stated, use 'not stated'.\n"
        "- cross_border_transfer: true ONLY if the passage explicitly ties "
        "this recipient to an international/cross-border transfer (outside "
        "EEA, SCCs, Data Privacy Framework, transfer to the US). Else null.\n\n"
        "HARD CONSTRAINTS\n"
        f"1. NEVER extract '{company_name}', its own brands/products, or 'you'/"
        "'the user' as a third party. Only OUTSIDE organizations.\n"
        "2. If the passage names no external recipient, return an empty list. "
        "Do not fabricate entities to fill the schema.\n"
        "3. Output every name and value in ENGLISH; translate/standardize "
        "non-English text.\n"
        "4. Be exhaustive on RECIPIENTS but conservative on ATTRIBUTION: list "
        "every distinct recipient, but only the data/purpose the passage "
        "actually ties to each one."
    )


# ==============================================================================
# 6. PUBLIC API
# ==============================================================================

def _extract_one_chunk(chunk: str, company_name: str) -> List[dict]:
    """Run Nemotron on a single chunk. Failures degrade to [] (never raise)."""
    try:
        result = client.chat.completions.create(
            model=MODEL_NAME,
            response_model=DataSupplyChain,
            max_retries=1,
            # Deterministic: identical policy text must yield identical
            # extraction across runs, or the research dataset is not
            # reproducible. temperature=0 also tightens instruction-following
            # for the classification rules in the system prompt.
            temperature=0,
            messages=[
                {"role": "system", "content": _system_prompt(company_name)},
                {"role": "user", "content": chunk},
            ],
        )
        return [e.model_dump() for e in result.entities]
    except Exception as e:  # noqa: BLE001 — one bad chunk must not sink the company
        print(f"      [!] chunk extraction failed: {str(e)[:140]}")
        return []


def extract_supply_chain(text: str, company_name: str) -> dict:
    """
    Extract the full data supply chain from an ENTIRE privacy policy.

    Accepts the complete policy text (not truncated snippets). Internally:
      1. windows the policy into overlapping ~1100-word chunks,
      2. drops chunks with no third-party-sharing language (token economy),
      3. runs Nemotron-120B structured extraction per chunk,
      4. merges + de-duplicates entities across chunks.

    Returns {"entities": [...], "_meta": {...}} — _meta is diagnostic and is
    ignored by the downstream pipeline's .get("entities", []).
    """
    _trace("=" * 60)
    _trace(f"EXTRACT SUPPLY CHAIN :: {company_name}")
    _trace("=" * 60)
    text = (text or "").strip()
    if not text:
        _trace("ABORT: empty text passed to extractor")
        return {"entities": []}

    chunks = _chunk_policy(text)
    if not chunks:
        _trace("ABORT: 0 chunks after splitting (no sentences found)")
        return {"entities": []}

    raw: List[dict] = []
    for i, ch in enumerate(chunks, 1):
        print(f"      - supply-chain chunk {i}/{len(chunks)} "
              f"({len(ch.split())} words)")
        raw.extend(_extract_one_chunk(ch, company_name))

    entities = _merge_entities(raw)
    _trace(f"raw entities={len(raw)}  ->  merged={len(entities)}")
    for e in entities:
        _trace(f"  * {e['third_party_name']!r} [{e['relationship_type']}] "
               f"cats={e.get('data_categories_shared')}")
    _trace("=" * 60)
    return {
        "entities": entities,
        "_meta": {
            "chunks_sent": len(chunks),
            "raw_entities": len(raw),
            "merged_entities": len(entities),
        },
    }
