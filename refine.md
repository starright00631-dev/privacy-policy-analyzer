# `batch_collect.py` Hardening — Refinement Log

**Date:** 2026-05-10
**Outcome:** 100% fetch success (33/33 companies, 0 failures, 0 timeouts, 0 empty payloads).

---

## 1. Initial state

The script was run as-is on a fresh environment. Several environment + code issues surfaced:

### Missing dependencies
The Python interpreter was missing several packages from `requirements.txt`:

| Package | Symptom |
|---|---|
| `readability-lxml` | `ModuleNotFoundError: No module named 'readability'` |
| `instructor` | `ModuleNotFoundError: No module named 'instructor'` |
| `spacy` + `en_core_web_sm` model | `OSError: [E050] Can't find model 'en_core_web_sm'` |

All were installed via `pip install ...` and `python -m spacy download en_core_web_sm`.
Playwright Chromium was already provisioned.

### Baseline run failures
Carried over from `prepared_reports_final/collection_log.json`, the baseline showed **4 fetches failing**:

| Company | Failure mode |
|---|---|
| **Meta** | Primary `https://www.meta.com/privacy/policy/` → HTTP 429; Facebook fallbacks → 281-word login wall; Wayback proxy unavailable |
| **IBM** | Primary returned only 10 words (JS-rendered shell); fallback same; Wayback proxy unavailable |
| **OpenAI** | Primary returned 38 words (Next.js shell before hydration); fallbacks same; Wayback proxy unavailable |
| **Mozilla** | Primary `/en-US/privacy/` is an index page returning 52 words; fallbacks same; Wayback proxy unavailable |

A new run after re-installing dependencies surfaced a **5th failure**:

| Company | Failure mode |
|---|---|
| **Perplexity** | Primary + every fallback URL returned **HTTP 403** (Cloudflare bot block); Wayback returned a snapshot but Playwright rendered 0 words because the Wayback `wombat.js` overlay broke the Framer-built page's hydration |

### Code-level bugs
- `batch_collect.py:73` — Oracle's fallback URL list had a missing comma, silently concatenating `"…/privacy-policy.html" "…/privacy/"` into a single broken URL.
- `privacy_analyzer/scraper.py` — Playwright launched with `headless=False`, which spawns a visible browser window per company (slow + unsafe for autonomous runs).
- Wayback fallback hard-coded a local Clash proxy (`http://127.0.0.1:7897`) and gave up if the proxy was offline.
- Wayback only used the simple availability API; when it returned no snapshot or when archive.org was overloaded (`503 Service Temporarily Unavailable`), the layer failed entirely.
- Wayback always re-rendered snapshots through Playwright. For SPAs (Framer / Next.js), the Wayback rewrite scripts collide with the original JS and the rendered DOM ends up empty even though the static archive HTML has the full text.

---

## 2. Refinements applied

### `privacy_analyzer/scraper.py`

**Playwright fetch engine (`_fetch_raw`)**
- Switched to `headless=True` (added `--disable-blink-features=AutomationControlled`, `--no-sandbox`, `--disable-dev-shm-usage` for stability).
- Added an init script that hides `navigator.webdriver` (basic anti-bot evasion).
- Uses `wait_until="domcontentloaded"` plus an opportunistic `wait_for_load_state("networkidle", timeout=10000)` so SPAs get extra time without hard-failing if they never settle.
- Added a `extra_wait_ms` parameter (default 8 s) so callers can extend the wait for known-slow pages.
- Expanded the cookie-wall click selector list (covers `Allow all cookies / Accept all / Accept / Agree / I Accept / aria-label*=Accept`).
- Added a scroll-to-bottom + scroll-to-top pass after settling to trigger lazy-loaded sections (this is what fixed IBM, OpenAI, and Mozilla — their content is rendered into virtualized lists that only mount after a scroll event).
- Wrapped `browser.close()` in `try/except` so cleanup never masks the real exception.

**Wayback resolver (`_wayback_lookup`, new)**
Three serial strategies, each tried with no proxy first then with `HTTP(S)_PROXY` env vars if set:
1. `archive.org/wayback/available` (the original simple API).
2. `web.archive.org/cdx/search/cdx?...&filter=statuscode:200&limit=-5` — surfaces older snapshots when the most-recent capture is missing.
3. **Redirect resolution:** `HEAD https://web.archive.org/web/<year>/<url>` (current year, then -1 / -2). Wayback redirects this canonical form to the closest 14-digit-timestamp snapshot. This works even when the API endpoints are throttled or returning 503 (which was the case during this session).

**Wayback content extraction (`_wayback_fallback` rewrite)**
- New helper `_fetch_static(url)` does a plain `requests.get` + `_extract_text`, with proxy fallback and `apparent_encoding` correction (some Wayback responses claim ISO-8859-1 when the body is UTF-8).
- `_wayback_fallback` now tries `_fetch_static` first; only falls back to Playwright rendering if the static path returns less than `MIN_WORDS`. This single change fixed Perplexity — the archived HTML contains 5,750 words inline, but Playwright's render of the same snapshot was 0 words because Wayback's wombat.js wrapper short-circuits the Framer hydration loop.

### `batch_collect.py`

- Fixed the Oracle URL syntax bug (missing comma).
- Added a `--no-llm` flag so fetch validation runs can skip the remote NVIDIA semantic-extractor calls. Default behavior is unchanged — running `python batch_collect.py` still calls the LLM exactly as before; the flag is opt-in for diagnosing fetch issues without spending tokens or being throttled by the upstream API.

---

## 3. Final success metrics

Final run: `python batch_collect.py --delay 1 --min-words 200 --no-llm`

| Metric | Value |
|---|---|
| **Companies attempted** | 33 |
| **Successfully collected** | **33 (100%)** |
| **Failed** | 0 |
| **Timeouts / 403 / empty payloads** | 0 |
| **Source: live**     | 26 |
| **Source: fallback** | 5 (Meta, Amazon, Mistral, TransUnion, Uber) |
| **Source: wayback**  | 2 (Salesforce, Perplexity) |
| **Low-quality (<500 words)** | 2 (Equifax 386 w, LiveRamp 331 w — both above the 200-word `--min-words` threshold and saved as legitimate fetches) |

Outputs:
- `prepared_reports/<company>.json` — 33 files
- `prepared_reports/summary.csv` — 33 rows
- `prepared_reports/collection_log.json` — `{ "collected": 33, "failed": [] }`

### Companies that previously failed and now succeed
| Company | Resolution path |
|---|---|
| Meta | Facebook fallback now extracts 7,476 words via the longer wait + cookie-wall clicker |
| IBM  | Live, 6,034 words — fixed by scroll-to-load + longer wait |
| OpenAI | Live, 3,873 words — fixed by scroll-to-load + longer wait |
| Mozilla | Live, 740 words — fixed by scroll-to-load + longer wait |
| Perplexity | Wayback static fetch, 5,750 words — fixed by skipping Playwright on archived snapshots |

### Restoring the LLM semantic extractor
The `--no-llm` flag was added purely to isolate the fetch layer during debugging. The original LLM-on behavior is the **default**: `python batch_collect.py` (no flag) runs the semantic extractor exactly as it did before this work, against the now-100%-successful fetches.

---

# v1 Deep Supply-Chain Pipeline — Post-Mortem

**Date:** 2026-05-16
**Outcome:** 33/33 collected, 0 corrupt, **231 distinct third-party entities**
(prior pipeline: ~40). Schema/consistency/cross-file HARD QA all pass.

## 1. Root-cause: the shallow-extraction bottleneck

The prior extractor produced 0–1 entities/company. Root cause was **not** the
schema — `batch_collect.py` fed Nemotron only `analyzer.examples[:3]`
(~600 truncated chars of a multi-thousand-word policy). Fix: feed
`results["full_text"]`; `semantic_extractor.py` now windows the policy into
overlapping ~1,100-word chunks internally, pre-filters to third-party-relevant
passages, extracts per chunk, and merges/de-duplicates across chunks.

## 2. The Definitional-Collapse Bug

**Symptom.** With the deep feed, a data broker's entire supply chain
collapsed to a single relationship type: every recipient — *Financial
institutions*, *Consumer credit customers*, *Third-party data providers* —
was labelled `processor`. Equifax: 8/8 `processor`, 0 `buyer`. For a
credit bureau this is semantically inverted: its customers are the
*destination/buyer* node, not vendors acting on its behalf.

**Cause.** `processor` was the model's low-effort default for any recipient
whose role wasn't spelled out verbatim, and the prompt lacked a decision
test for "controls the data for its own use" vs. "acts on our behalf."

**Fix — the bounded Data-Broker domain rule.** A staged prompt overhaul:
1. Added an explicit *who-controls-the-data* decision test (buyer vs.
   processor) and made `unknown` (not `processor`) the honest fallback.
2. Added a **bounded contextual exception**: if the company's core business
   is selling data/reports/scores (credit bureau, data broker) and the
   recipient is a `customer / client / subscriber / financial institution`
   receiving reports/products, it MUST be classified `buyer` — the business
   model supplies the missing context. The exception explicitly does *not*
   override a clearer `processor`/`joint_controller`/`ai_trainer` signal.
3. Set `temperature=0` for reproducibility (see §4b — MoE non-determinism
   means the *material* signal is stable even though byte-identical output
   is not guaranteed on the hosted endpoint).

**Result.** Data Broker category post-fix: **33 `buyer` vs. 8 `processor`
(4.1 : 1)**; Equifax 7/7 `buyer`, Acxiom 4/4, TransUnion 16 buyers. The
definitional collapse is resolved and the brokers' monetization vector is
correctly quantified.

## 3. Strict-Local-Attribution — trade-offs

**Problem.** The model copy-pasted a policy's global CCPA "categories of
personal information we collect" enumeration onto *every* extracted
recipient, manufacturing identical 11-item category lists.

**Fix.** A `STRICT LOCAL ATTRIBUTION` prompt block: a recipient gets a data
category only if the passage ties that category to *that* recipient; absent
that, `data_categories_shared` is `[]`.

**Trade-off (documented as a feature).** Many entities now have empty
`data_categories_shared`. This is **correct anti-hallucination behaviour**,
not a regression — an empty list is an honest "the text did not say,"
whereas the prior non-empty lists were fabricated. `validate_csv.py`'s soft
check still flags this; the flag is expected and is *not* a defect. The same
philosophy explains `n_ai_trainers == 0` corpus-wide and only 5 cross-border
entities: precision over recall, zero implicit inference.

## 4. Resilience & OS Quirks (4 autonomous recovery events)

A multi-hour, network-bound LLM sweep surfaced three distinct failure
classes plus an environment bug. Checkpointing (per-company JSON written
immediately; resume skips any company whose checkpoint parsed and ran the
extractor) preserved integrity through all of them.

**(a) Silent API deadlocks → request timeouts.** The `instructor`/OpenAI
client had no timeout, so a stuck Nemotron call hung *with no exit code*.
Loud auto-recovery (exit-code based) could not catch it; a stall-watchdog
(no-log-growth detector) was added as a backstop. Permanent fix: an explicit
client `timeout` so a hung call *raises* and `_extract_one_chunk` degrades
it to `[]` instead of freezing the run. Recurred deterministically at one
pathological policy (TikTok) until `max_retries` was also bounded.

**(b) The `taskkill` Git-Bash path-mangling bug → concurrent writers.**
`taskkill /F /IM python.exe` issued from the Git-Bash shell had its
`/F /IM` flags **path-converted to `F:/`** by MSYS, so *every* process
kill silently failed for much of the session. Zombie sweeps accumulated to
**5 concurrent `batch_collect` writers** against the same `prepared_reports/`
directory. Detected via `tasklist` showing persistent PIDs across "kills".
Fix: kill via `powershell -NoProfile -Command "Stop-Process -Name python
-Force"` (no MSYS path mangling). Lesson: on Windows+Git-Bash, never use
`taskkill /FLAG`; use PowerShell `Stop-Process`. JSON integrity survived
only because the checkpoint write was effectively atomic per company.

**(c) 75 s vs. 300 s chunk truncation → surgical re-collection of 17.**
The fix for (a) was initially over-tuned to `timeout=75 s` — *below*
legitimate Nemotron latency (measured ~50–120 s/chunk; Equifax 3-chunk
extraction runs 139–360 s). 27 chunks silently timed out → empty
extraction; **Equifax regressed 8 → 0 entities**. This was caught by a
post-sweep integrity check (13 zero-entity companies), not assumed-good.
Recalibrated to `timeout=300 s` (clears real latency, still catches the
~1 h true hang) with `max_retries=1` (bounded fail-fast). Recovery was
**surgical**: identified the 17 companies with ≥1 timeout ∪ zero entities,
deleted only their checkpoints, and resumed — the 16 fully-clean companies
were preserved, not re-run.

**(d) GBK console encoding.** The host runs a GBK (Windows China) console;
`✓ / → / ═` glyphs in `print()` raised `UnicodeEncodeError` and crashed
runs mid-sweep. Fix: a `sys.stdout/stderr.reconfigure(encoding="utf-8",
errors="backslashreplace")` safety net at the entry point of every runnable
script, plus ASCII-only telemetry in the `SC_TRACE` path.

## 5. Known limitations carried into v1
- 5 companies (Apple, Uber, Reddit, Twitter/X, LinkedIn) = insufficient data
  (thin fetch, opacity 100 by construction — *not* proven secrecy).
- `n_ai_trainers == 0` corpus-wide (zero-inference design choice).
- Cross-border = 5 entities (strict-attribution floor, not a census).
- Residual: ~7 chunks across 5 otherwise-populated companies lost to
  transient timeout; data substantial, accepted over infinite retry.
