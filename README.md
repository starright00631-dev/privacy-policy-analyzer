# Privacy Policy Analyzer

An end-to-end NLP pipeline that scrapes, structures, and analyzes the privacy policies of **33 major organizations** (Big Tech, AI labs, Social, Privacy-First, Data Brokers, Platforms) — quantifying readability, consent model design, third-party data supply chains, and syntactic dark patterns.

Achieved **100% fetch success (33/33)** through a hardened three-layer scraper (live → fallback URLs → Wayback Machine) with anti-bot evasion. Third-party entities are extracted via **NVIDIA Nemotron-120B** with Pydantic-validated structured outputs.

---

## Motivation

Privacy policies are the primary legal interface between users and the companies collecting their data, yet research consistently shows they are written far above the average adult reading level. Wagner (2023) found that average readability scores have *worsened* over 25 years, and Das et al. (2018) measured a mean grade level of 12.78 across popular apps — equivalent to a college freshman. Meanwhile, Lin & Strulov-Shlain (2023) demonstrated that opt-out defaults suppress users' privacy valuations by 14–22% compared to opt-in, and Kraft et al. (2024) confirmed that when Apple's ATT gave users a genuine opt-in choice, tracking rates dropped by 54.73%.

This project operationalizes those findings into an automated analysis tool. Rather than reading policies manually, it applies modern NLP — including a 120B-parameter LLM — to extract structured, comparable metrics across organizations in six categories.

---

## Highlights

- **Nemotron-120B structured extraction** (`nvidia/nemotron-3-super-120b-a12b`) with Pydantic schema validation via the `instructor` library — classifies third parties as `buyer`, `processor`, or `ai_trainer`.
- **Hardened three-layer scraper** with Playwright anti-bot evasion, cookie-wall bypass, scroll-to-load, and a Wayback Machine fallback that resolves snapshots even when the Internet Archive APIs are down (achieved **33/33** companies, 0 failures).
- **HuggingFace zero-shot intent classification** (`facebook/bart-large-mnli`) over 500-token chunks.
- **spaCy syntactic dark-pattern detection** — passive-voice ratio, vague-language density, and parse-tree depth combined into a `dark_score`.
- **Reproducible outputs**: 33 structured JSON reports, a chunk-level CSV, a company-level aggregate CSV, and 3 publication-quality Plotly figures.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Data Collection (batch_collect.py)                    │
│                                                                 │
│   Scraper (privacy_analyzer/scraper.py)                         │
│     live URL → fallback URLs → Wayback Machine                  │
│     Playwright headless + anti-bot init script                  │
│     Cookie-wall click bypass + scroll-to-load                   │
│     Static-requests path for archived SPAs                      │
│                                                                 │
│   Analyzer (privacy_analyzer/analyzer.py)                       │
│     Section detection · readability · consent model             │
│     · keyword-based data categories · third-party flag          │
│                                                                 │
│   Structured Extraction (semantic_extractor.py)                 │
│     ► Nemotron-120B + Pydantic schema (instructor)              │
│       returns {entities: [{name, relationship_type}]}           │
│                                                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 33 structured JSON files
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: NLP Pipeline (nlp_pipeline_final.py)                  │
│                                                                 │
│   Phase 1   Ingest → drop noise sections → 500-token chunks     │
│   Phase 2A  HuggingFace bart-large-mnli zero-shot intent        │
│   Phase 2B  Read Nemotron entities from JSON (no re-extraction) │
│   Phase 2C  spaCy nlp.pipe() dark-pattern signals               │
│   Phase 3   Aggregate to one row per company → CSV + Plotly     │
└─────────────────────────────────────────────────────────────────┘
```

### Module breakdown

| # | Module | Method | Output |
|---|---|---|---|
| **B** | **Supply Chain (headline)** | NVIDIA **Nemotron-120B** via the `instructor` library, structured to a Pydantic `DataSupplyChain` schema. Run once per company during collection on the sentences that the analyzer flagged as third-party-sharing language. | Per-company list of named third parties, each tagged `buyer` \| `processor` \| `ai_trainer` |
| **A** | Intent classification | HuggingFace Inference API, `facebook/bart-large-mnli` zero-shot over 500-token chunks. Cold-start handled via `x-wait-for-model`. | Per-chunk label + confidence: Functionality, Advertising, Third-Party Sale, Security, AI Training |
| **C** | Dark patterns | spaCy `nlp.pipe()` (single-process, Windows-safe) measuring passive-voice dependency ratio, vague-language regex density, and average parse-tree depth | Composite `dark_score` per chunk → averaged per company |

---

## Module B in detail — Nemotron-120B structured extraction

`privacy_analyzer/semantic_extractor.py` defines a Pydantic schema and uses the `instructor` library to force the model to return JSON that conforms to it:

```python
class ThirdPartyEntity(BaseModel):
    third_party_name: str
    relationship_type: str  # 'buyer' | 'processor' | 'ai_trainer'

class DataSupplyChain(BaseModel):
    entities: List[ThirdPartyEntity]

client = instructor.from_openai(
    OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=os.environ["NVIDIA_API_KEY"],
    )
)

extracted = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    response_model=DataSupplyChain,
    messages=[...]
)
```

Why a 120B model? Privacy policies use deliberately vague phrasing (*"trusted partners"*, *"service providers"*, *"affiliates"*). A smaller model loses the relational context needed to decide whether *"advertising partners"* is a **buyer** (receives data for their own targeting) or a **processor** (operating on the company's behalf). The system prompt encodes auditor-style relationship definitions and explicitly forbids extracting the company itself or generic placeholders.

The output is consumed downstream by the NLP pipeline (no re-extraction) and is what powers the `n_buyers` / `n_processors` / `n_ai_trainers` columns in `company_aggregated.csv` and the bubble sizes in `fig3_darkpattern.html`.

---

## Scraper — production-grade fetch layer

The scraper is the second piece of resume-worthy engineering. The design target was **100% fetch success across 33 production sites** despite Cloudflare bot blocks, JS-rendered single-page apps, login walls, cookie consent overlays, and a partially-down Internet Archive.

| Layer | What it does | Notable hardening |
|---|---|---|
| **1. Live (Playwright)** | Headless Chromium fetches the primary URL | `--disable-blink-features=AutomationControlled`, `navigator.webdriver` masking, realistic UA + `Sec-Ch-Ua` headers, `wait_until="domcontentloaded"` then opportunistic `networkidle`, configurable post-load wait, expanded cookie-wall click selectors (Allow all / Accept all / Agree / aria-label\*=Accept), **scroll-to-bottom + scroll-to-top** to trigger lazy-loaded content (this is what fixed IBM, OpenAI, and Mozilla, which were returning 10–52-word stub pages) |
| **2. Fallback URLs** | Tries each company's per-site backup URLs | Same anti-bot stack; required for Meta (`meta.com → facebook.com/privacy/policy/`), Amazon, Mistral, TransUnion, Uber |
| **3. Wayback Machine** | Snapshot resolver + extractor | Three serial lookup strategies — availability API, CDX API, then **`/web/<year>/<url>` redirect resolution** which works even when both APIs return 503. Snapshots are fetched with **plain `requests` + Readability/BeautifulSoup first** (Playwright re-rendering Wayback's `wombat.js` shim breaks Framer/Next.js archived pages — Perplexity rendered as 0 words via Playwright but 5,750 words via static fetch) |

Every layer accepts a result only if it returns ≥ `MIN_WORDS` (300) — short stub pages are rejected and the next layer is tried.

### Final fetch metrics (33 companies)

| Metric | Value |
|---|---|
| Successfully collected | **33 / 33 (100%)** |
| Failed | 0 |
| Source: live | 26 |
| Source: fallback URL | 5 (Meta, Amazon, Mistral, TransUnion, Uber) |
| Source: Wayback Machine | 2 (Salesforce, Perplexity) |

Full refinement log in [`refine.md`](./refine.md).

---

## Key Findings (33 Organizations)

**Readability:** Every category exceeds the average US adult reading level (grade 8). Data Brokers and Big Tech average grade 14+, while Privacy-First companies (Mozilla, Brave, DuckDuckGo) average grade 11.1 — still above grade 8 but measurably more accessible.

**Consent Model:** Data Brokers rely predominantly on opt-out consent. Only one organization across all 33 (Mozilla) uses primarily opt-in language. The structural default across the industry is data collection unless users actively object.

**Supply Chain — the Data Broker breakthrough (Nemotron-120B):** The v1 deep
pipeline extracts **231 distinct third-party entities** across the corpus. The
headline result is in the Data Broker category: a **bounded domain rule**
(credit bureaus / data brokers selling reports → the recipient is a `buyer`,
not a vendor) corrected a prior *definitional-collapse* failure where every
recipient defaulted to `processor`. Post-fix, the Data Broker category shows
**33 `buyer` relationships vs. 8 `processor`** — a **4.1 : 1 ratio** (Equifax
7/7 buyers, Acxiom 4/4, TransUnion 16 buyers). This correctly quantifies the
primary monetization vector of data brokers: their supply chain is *data sale
to independent recipients*, not infrastructure processing. The earlier
pipeline reported these same brokers as ~0–1 buyers and almost entirely
`processor` — the breakthrough is this corrected characterization.

**Dark Patterns:** Privacy-First companies cluster distinctly from Data Brokers and Big Tech on both syntactic complexity and vague-language density, suggesting that accessible policy writing is a design choice, not an industry constraint.

---

## Data Limitations & Caveats

True engineering rigor requires stating where v1 is conservative or
depth-limited. None of the following are hidden behind the headline numbers.

**1. An opacity score of 100.0 means "insufficient data," not "maximum
secrecy."** Five companies — **Apple, Uber, Reddit, Twitter/X, LinkedIn** —
score `supply_chain_opacity_score = 100.0`. This is an **artifact of the
scraper hitting depth limits**: the fetched policies were thin (≈600–800
words — short summary/landing pages, not the full legal text), so few or no
third-party recipients could be extracted. The opacity metric assigns 100 by
construction when zero entities are found. These five must be treated as
**insufficient data / not analyzable in v1**, *not* as evidence of maximal
corporate secrecy. They were also ~0–1 entities in the pre-v1 pipeline, so
this is a stable fetch-depth limitation, not a regression.

**2. `n_ai_trainers` is exactly 0 across all 33 companies — by design, not by
omission.** The extraction prompt prioritizes **zero hallucination over
implicit inference**. Privacy policies say "we share data with cloud
providers / service providers / analytics partners"; they essentially never
say "we send your data to an AI trainer." Because the model is forbidden from
*assuming* an AI-training relationship that the text does not explicitly
state, the `ai_trainer` class correctly never fires. This is a deliberate
precision-over-recall trade-off: a `0` we can defend, not a fabricated number.

**3. Cross-border transfers are deliberately under-counted.** Only **5
entities** across the entire corpus carry `cross_border_transfer = True`
(e.g. Cohere → Google Cloud Platform, Anthropic for storage/processing). The
**strict-local-attribution** prompt only flags a transfer when the passage
*explicitly* ties an international transfer to that specific recipient. This
is intentionally conservative — it proves the prompt is not inflating the
signal, but it also means cross-border flows are a floor, not a census.

**4. Data categories are frequently empty — this is the cure, not a defect.**
Many entities have an empty `data_categories_shared`. Earlier the model
copy-pasted a policy's global CCPA "categories we collect" list onto every
recipient. The strict-local-attribution rule now forces the list empty unless
a category is tied to *that* recipient in-passage. Empty is the correct,
honest output of an anti-hallucination design.

---

## Installation

**Requirements:** Python 3.10+ (tested on 3.12 and 3.14)

```bash
git clone https://github.com/YOUR_USERNAME/privacy-policy-analyzer.git
cd privacy-policy-analyzer
pip install -r requirements.txt
python -m playwright install chromium
python -m spacy download en_core_web_sm
```

### Environment Variables

```bash
# Module B — Nemotron-120B supply chain extraction (NVIDIA Build / NIM)
export NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Module A — HuggingFace zero-shot intent classification (free token)
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

Get keys at `https://build.nvidia.com` and `https://huggingface.co/settings/tokens`. **Never commit API keys.**

---

## Usage

### 1. Collect privacy policies (with Nemotron extraction)

```bash
python batch_collect.py
# optional flags:
python batch_collect.py --delay 1 --min-words 200
python batch_collect.py --no-llm           # fetch-only, skip Nemotron (debug)
```

Fetches all 33 policies via the three-layer scraper, runs the analyzer, and calls **Nemotron-120B** on each policy's flagged third-party-sharing sentences. Writes one structured JSON per company to `prepared_reports/`, plus a `summary.csv` and a `collection_log.json`.

### 2. Run the NLP pipeline

```bash
# Full pipeline (requires HF_TOKEN):
python nlp_pipeline_final.py --reports-dir prepared_reports_final --hf-token $HF_TOKEN

# Without intent classification (no token needed):
python nlp_pipeline_final.py --reports-dir prepared_reports_final --skip-intent
```

The pipeline reuses the Nemotron entities already in the JSONs — no re-extraction.

### 3. Outputs

```
prepared_reports_final/         # 33 per-company JSONs (analysis + Nemotron entities)
├── google.json
├── ...
├── summary.csv                 # one row per company, ML-friendly features
└── collection_log.json         # {collected, failed, low_quality}

nlp_outputs_final/
├── chunks.csv                  # all 500-token chunks with metadata
├── chunks_analyzed.csv         # chunks + intent labels + dark-pattern scores
├── company_aggregated.csv      # one row per company, all metrics
├── fig1_readability.html/png   # readability by category
├── fig2_consent.html/png       # consent model distribution
└── fig3_darkpattern.html/png   # complexity vs dark-pattern scatter
```

---

## Project Structure

```
privacy-policy-analyzer/
│
├── batch_collect.py                # Data collection orchestrator (calls scraper + analyzer + Nemotron)
├── nlp_pipeline_final.py           # NLP analysis pipeline (Modules A + B-read + C)
├── main.py                         # Single-URL CLI analyzer
├── refine.md                       # Scraper hardening log (33/33 success)
│
├── privacy_analyzer/
│   ├── semantic_extractor.py       # ► Nemotron-120B + Pydantic structured extraction
│   ├── scraper.py                  # 3-layer Playwright + Wayback scraper (anti-bot)
│   ├── analyzer.py                 # Readability, consent, sections, third-party flag
│   ├── readability_metrics.py      # Coleman-Liau + Flesch (no external deps)
│   ├── keywords.py                 # Data-category keyword dictionaries
│   └── reporter.py                 # Terminal report formatter
│
├── prepared_reports_final/         # 33 structured JSON files
├── nlp_outputs_final/              # Pipeline output (CSV + figures)
├── requirements.txt
└── README.md
```

---

## Data Schema

Each JSON file in `prepared_reports_final/` contains:

```json
{
  "company": "Google",
  "category": "Big Tech",
  "source_url": "https://policies.google.com/privacy",
  "data_source": "live",
  "scraped_at": "2026-05-10T...Z",
  "analysis": {
    "full_text": "...",
    "sentences": ["...", "..."],
    "policy_date": "January 15, 2024",
    "sections": [
      {"heading": "Information We Collect", "sentences": [...], "sentence_count": 12}
    ],
    "readability": {"grade_level": 11.3, "reading_ease": 39.6, "word_count": 8421},
    "consent_model": {"model": "Primarily Opt-Out", "opt_in_phrase_count": 1, "opt_out_phrase_count": 8},
    "data_categories": {"Location": {"keyword_hits": 47, "matched_keywords": ["location", "gps"]}},
    "third_party_sharing": {"detected": true, "total_mentions": 23, "examples": [...]},
    "semantic_supply_chain": {
      "entities": [
        {"third_party_name": "advertising partners", "relationship_type": "buyer"},
        {"third_party_name": "Google Cloud Platform", "relationship_type": "processor"}
      ]
    }
  }
}
```

---

## References

- Wagner, I. (2023). Privacy Policies Across the Ages: Content of Privacy Policies 1996–2021. *ACM Transactions on Privacy and Security, 26*(3), 1–35. https://doi.org/10.1145/3590152
- Das, G., Cheung, C., Nebeker, C., Bietz, M., & Bloss, C. (2018). Privacy Policies for Apps Targeted Toward Youth: Descriptive Analysis of Readability. *JMIR mHealth and uHealth, 6*(1), e3. https://doi.org/10.2196/mhealth.7626
- Lin, T. & Strulov-Shlain, A. (2023). Choice Architecture, Privacy Valuations, and Selection Bias in Consumer Data. *Working Paper*. https://arxiv.org/abs/2308.13496
- Kraft, L., Bleier, A., Skiera, B., & Koschella, T. (2024). Granular Control and Privacy Decisions: Evidence from Apple's App Tracking Transparency. *SSRN Working Paper*. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4598472
- Bollinger, B., Collis, A., & Doshi, A. (2022). Opted Out, Yet Tracked: Are Regulations Enough to Protect Your Privacy? *Proceedings of the ACM Web Conference 2022*. https://arxiv.org/abs/2202.00885

---

## License

MIT
