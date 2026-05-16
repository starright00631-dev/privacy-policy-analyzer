"""
batch_collect.py
Fetches privacy policies from 33 companies with three-layer fallback:
  1. Primary URL
  2. Alternative URLs (tried in order)
  3. Wayback Machine snapshot

Output:
  prepared_reports/<company>.json   — full analysis + raw text + sentences
  prepared_reports/summary.csv      — tabular ML features, one row per company
  prepared_reports/collection_log.json

Usage:
    python batch_collect.py
    python batch_collect.py --delay 3 --min-words 200
"""

import os, sys, json, csv, time, argparse, re
from datetime import datetime, timezone

# Encoding safety net: this repo runs on a GBK (Windows China) console that
# raises UnicodeEncodeError on the → / ✓ / ⚠ glyphs used in this file's logs
# and on any SC-TRACE / library output. Force UTF-8 so a multi-hour 33-company
# sweep can never die on a print() call.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="backslashreplace")
    except Exception:
        pass

from privacy_analyzer.scraper import fetch_policy
from privacy_analyzer.analyzer import analyze
from privacy_analyzer.semantic_extractor import extract_supply_chain

# ── Company list ──────────────────────────────────────────────────────────────
# Format: (display_name, primary_url, [fallback_urls], category)
# Fallbacks are tried in order before Wayback Machine.

COMPANIES = [
    # ── Big Tech ──────────────────────────────────────────────────────────────
    ("Google",
        "https://policies.google.com/privacy",
        [],
        "Big Tech"),

    ("Microsoft",
        "https://privacy.microsoft.com/en-us/privacystatement",
        ["https://www.microsoft.com/en-us/privacy/privacystatement"],
        "Big Tech"),

    ("Apple",
        "https://www.apple.com/legal/privacy/en-ww/",
        [
            "https://www.apple.com/legal/privacy/en/",
            "https://www.apple.com/privacy/",
        ],
        "Big Tech"),

    ("Meta",
        "https://www.meta.com/privacy/policy/",
        [
            "https://www.facebook.com/privacy/policy/",
            "https://www.facebook.com/about/privacy/",
            "https://www.facebook.com/privacy/policy/?entry_point=data_policy_redirect&entry=0",
        ],
        "Big Tech"),

    ("Amazon",
        "https://www.amazon.com/gp/help/customer/display.html?nodeId=GX7NJQ4ZB8MHFRNJ",
        [
            "https://aws.amazon.com/privacy/",
            "https://www.amazon.co.uk/gp/help/customer/display.html?nodeId=GX7NJQ4ZB8MHFRNJ",
        ],
        "Big Tech"),

    ("IBM",
        "https://www.ibm.com/privacy",
        ["https://www.ibm.com/us-en/privacy",],
        "Big Tech"),

    ("Oracle",
        "https://www.oracle.com/legal/privacy/customer-data-research-development-privacy-policy/",
        [
            "https://www.oracle.com/legal/privacy/privacy-policy.html",
            "https://www.oracle.com/legal/privacy/",
        ],
        "Big Tech"),

    ("Salesforce",
        "https://www.salesforce.com/company/privacy/",
        ["https://www.salesforce.com/company/privacy/full_privacy.jsp"],
        "Big Tech"),

    # ── AI Companies ──────────────────────────────────────────────────────────
    ("OpenAI",
        "https://openai.com/policies/privacy-policy",
        [
            "https://openai.com/privacy/",
            "https://openai.com/policies/row-privacy-policy/",
         ],
        "AI"),

    ("Anthropic",
        "https://www.anthropic.com/privacy",
        ["https://www.anthropic.com/legal/privacy"],
        "AI"),

    ("xAI",
        "https://x.ai/privacy-policy",
        ["https://x.ai/legal/privacy-policy"],
        "AI"),

    ("Cohere",
        "https://cohere.com/privacy",
        ["https://cohere.ai/privacy"],
        "AI"),

    ("Mistral",
     "https://mistral.ai/privacy-policy",
     [
        "https://legal.mistral.ai/terms/data-processing-addendum",
         "https://mistral.ai/privacy/",
         "https://mistral.ai/fr/privacy-policy/",
         "https://docs.mistral.ai/getting-started/compliance-and-privacy/",
     ],
     "AI"),

    ("HuggingFace",
        "https://huggingface.co/privacy",
        [],
        "AI"),

    ("Perplexity",
        "https://www.perplexity.ai/hub/legal/privacy-policy",
        [
            "https://www.perplexity.ai/privacy",
            "https://www.perplexity.ai/hub/legal/privacy-policy",
            "https://www.perplexity.ai/hub/legal",
        ],
        "AI"),

    ("Palantir",
        "https://www.palantir.com/privacy-and-security/",
        [
             "https://www.palantirventures.com/legal/privacy-and-cookie-policy/",
             "https://www.palantir.com/privacy/",
        ],
        "AI"),

    # ── Social Media ──────────────────────────────────────────────────────────
    ("Twitter/X",
        "https://twitter.com/en/privacy",
        ["https://x.com/en/privacy"],
        "Social"),

    ("TikTok",
        "https://www.tiktok.com/legal/page/us/privacy-policy/en",
        [
            "https://newsroom.tiktok.com/en-us/privacy-policy",
            "https://www.tiktok.com/legal/page/us/privacy-policy/en",
            "https://www.tiktok.com/legal/page/us/privacy-policy/en",
         ],
        "Social"),

    ("Snapchat",
        "https://www.snap.com/en-US/privacy/privacy-policy",
        [],
        "Social"),

    ("Reddit",
        "https://www.reddit.com/policies/privacy-policy",
        ["https://old.reddit.com/policies/privacy-policy"],
        "Social"),

    ("Pinterest",
        "https://policy.pinterest.com/en/privacy-policy",
        [],
        "Social"),

    ("LinkedIn",
        "https://cn.linkedin.com/legal/privacy-policy?",
        [
            "https://www.linkedin.com/legal/privacy-policy",
            "https://cn.linkedin.com/legal/privacy-policy?",

         ],
        "Social"),

    # ── Privacy-First ──────────────────────────────────────────────────────────
    ("Mozilla",
        "https://www.mozilla.org/en-US/privacy/",
        ["https://www.mozilla.org/privacy/"],
        "Privacy-First"),

    ("Brave",
        "https://brave.com/privacy/browser/",
        ["https://brave.com/privacy-policy/"],
        "Privacy-First"),

    ("DuckDuckGo",
        "https://duckduckgo.com/privacy",
        [
            "https://duckduckgo.com/privacy-policy",
            "https://duckduckgo.com/privacy",
        ],
        "Privacy-First"),

    # ── Data Brokers ──────────────────────────────────────────────────────────
    ("Acxiom",
        "https://www.acxiom.com/privacy/international-privacy-notice/#rights",
        [
            "https://www.acxiom.com/about-us/privacy/",
            "https://www.acxiom.com/privacy/privacy-policy/",
         ],
        "Data Broker"),

    ("Experian",
        "https://www.experian.com/privacy/us-consumer-data-privacy-policy",
        [
            "https://www.experian.com/privacy/center.html",
            "https://www.experian.com/privacy/",
            "https://www.experian.com/corporate/privacy-policy",
            "https://www.experian.com/privacy/us-consumer-data-privacy-policy",
        ],
        "Data Broker"),

    # Equifax fix (2026-05-16): the old primary /privacy/ is a 386-word HUB
    # page ("we've consolidated ... into our new, comprehensive Equifax Privacy
    # Statement — Read the Notice"). 386 > MIN_WORDS(300) so the scraper wrongly
    # accepted the stub as 'live' and never reached the real policy. The actual
    # comprehensive statement is /privacy/privacy-statement/ (2,373 words,
    # verified). The old hub is kept ONLY as a last-resort fallback; it is a
    # stub so it would still be a poor result, but primary now resolves first.
    ("Equifax",
        "https://www.equifax.com/privacy/privacy-statement/",
        ["https://www.equifax.com/privacy/"],
        "Data Broker"),

    ("TransUnion",
        "https://www.transunion.com/privacy/consumer-privacy-policy",
        [
            "https://www.transunion.com/privacy",
            "https://www.transunion.com/privacy/privacy-policy",
            "https://www.transunion.com/legal/privacy-policy",
        ],
        "Data Broker"),

    ("LiveRamp",
        "https://liveramp.com/privacy/",
        ["https://liveramp.com/privacy-policy/"],
        "Data Broker"),

    ("The Trade Desk",
        "https://www.thetradedesk.com/us/privacy",
        ["https://www.thetradedesk.com/us/privacy-policy"],
        "Data Broker"),

    # ── Platforms ─────────────────────────────────────────────────────────────
    ("Uber",
        "https://www.uber.com/legal/en/document/?name=privacy-policy",
        [
            "https://www.uber.com/us/en/privacy/",
            "https://www.uber.com/global/en/privacy/overview/",
            "https://www.uber.com/en-GB/legal/privacy/users/",
        ],
        "Platform"),
    ("Airbnb",
        "https://www.airbnb.com.sg/help/article/3175?_set_bev_on_new_domain=1777688869_EAOGRiMjE3YTZlMT&set_everest_cookie_on_new_domain=1777688869.EAYzk2YjU3YjJlZDRkZm.X3yA4Ka8Hgna7fzUGDRxrHYXugABNcTpi4pc9B7b5i0",
        [
            "https://www.airbnb.com/help/article/2855?locale=en",
            "https://www.airbnb.com/help/article/2855/airbnb-privacy-policy",
            "https://www.airbnb.com/help/article/2855",
            "https://www.airbnb.com/privacy-policy",
        ],
        "Platform"),

]

OUTPUT_DIR = "prepared_reports"
MIN_WORDS_QUALITY = 500   # flag low-quality scrapes in log but still save them


def safe_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def flatten_for_csv(name, url, category, source, results):
    r = results
    cats = r["data_categories"]
    cm = r["consent_model"]
    return {
        "company":             name,
        "category":            category,
        "url":                 url,
        "source":              source,
        "policy_date":         r.get("policy_date") or "Unknown",
        "scraped_at":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "word_count":          r["readability"]["word_count"],
        "sentence_count":      r["sentence_count"],
        "section_count":       r.get("section_count", 0),
        "section_headings":    "|".join(r.get("section_headings", [])),
        "grade_level":         r["readability"]["grade_level"],
        "reading_ease":        r["readability"]["reading_ease"],
        "readability_label":   r["readability"]["description"],
        "consent_model":       cm["model"],
        "opt_in_phrase_count": cm.get("opt_in_phrase_count", 0),
        "opt_out_phrase_count":cm.get("opt_out_phrase_count", 0),
        "third_party_detected":r["third_party_sharing"]["detected"],
        "third_party_mentions":r["third_party_sharing"].get("total_mentions", 0),
        "retention_stated":    r["retention_policy"]["detected"],
        "retention_mentions":  r["retention_policy"].get("total_mentions", 0),
        "cat_location":        "Location" in cats,
        "cat_financial":       "Financial" in cats,
        "cat_biometric":       "Biometric" in cats,
        "cat_browsing_device": "Browsing & Device" in cats,
        "cat_contact_id":      "Contact & Identity" in cats,
        "cat_behavioral":      "Behavioral & Inferred" in cats,
        "cat_communications":  "Communications" in cats,
        "total_categories":    len(cats),
        "hits_location":       cats.get("Location",             {}).get("keyword_hits", 0),
        "hits_financial":      cats.get("Financial",            {}).get("keyword_hits", 0),
        "hits_biometric":      cats.get("Biometric",            {}).get("keyword_hits", 0),
        "hits_browsing_device":cats.get("Browsing & Device",    {}).get("keyword_hits", 0),
        "hits_contact_id":     cats.get("Contact & Identity",   {}).get("keyword_hits", 0),
        "hits_behavioral":     cats.get("Behavioral & Inferred",{}).get("keyword_hits", 0),
        "hits_communications": cats.get("Communications",       {}).get("keyword_hits", 0),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay",     type=float, default=2.0)
    parser.add_argument("--min-words", type=int,   default=200)
    parser.add_argument("--no-llm",    action="store_true",
                        help="Skip the remote semantic extractor (use for fetch-only validation runs)")
    parser.add_argument("--fresh",     action="store_true",
                        help="Ignore existing checkpoints and re-collect every company")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_rows, failed, low_quality = [], [], []

    def _checkpoint_ok(fp: str) -> bool:
        """A checkpoint is valid only if it parsed and actually ran the
        semantic extractor (so a crash mid-extraction is NOT skipped)."""
        try:
            with open(fp, encoding="utf-8") as cf:
                d = json.load(cf)
            sc = d.get("analysis", {}).get("semantic_supply_chain")
            # Must have run extraction unless it was an intentional --no-llm
            # skip; a missing/None sc means the run died before extraction.
            return isinstance(sc, dict)
        except Exception:
            return False

    print(f"\n{'='*65}")
    print(f"  Batch Privacy Policy Collector")
    print(f"  Companies : {len(COMPANIES)}  |  Delay: {args.delay}s")
    print(f"  Strategy  : live → fallback URLs → Wayback Machine")
    print(f"{'='*65}\n")

    for i, (name, primary_url, fallbacks, category) in enumerate(COMPANIES, start=1):
        print(f"[{i:02d}/{len(COMPANIES)}] {name} ({category})")
        print(f"           {primary_url}")

        # ── Checkpoint / resume ────────────────────────────────
        # JSON is written immediately after each company below, so a crash
        # at company N loses only company N. On restart we skip every
        # company whose checkpoint already parsed and ran the extractor.
        ck_path = os.path.join(OUTPUT_DIR, safe_filename(name) + ".json")
        if not args.fresh and _checkpoint_ok(ck_path):
            with open(ck_path, encoding="utf-8") as cf:
                done = json.load(cf)
            src = (done.get("data_source", "") or "").split(":")[0] or "checkpoint"
            csv_rows.append(flatten_for_csv(
                name, primary_url, category, src, done["analysis"]))
            print(f"  [SKIP] checkpoint exists - resuming past it. "
                  f"Use --fresh to force re-collect.\n")
            continue

        # ── Fetch with fallback chain ──────────────────────────
        try:
            text, source = fetch_policy(
                primary_url,
                fallback_urls=fallbacks,
                use_wayback=True,
                retry_on_429=2
            )
        except RuntimeError as e:
            print(f"  ✗ ALL STRATEGIES FAILED: {e}\n")
            failed.append({"company": name, "category": category,
                           "url": primary_url, "reason": str(e)})
            time.sleep(args.delay)
            continue

        wc = len(text.split())
        source_label = source.split(":")[0]   # live / fallback / wayback
        print(f"  Source    : {source_label}  |  {wc:,} words")

        if wc < args.min_words:
            print(f"  ⚠  Only {wc} words — skipping (JS-rendered or index page)\n")
            failed.append({"company": name, "category": category,
                           "url": primary_url,
                           "reason": f"Too few words ({wc}) after all fallbacks"})
            time.sleep(args.delay)
            continue

        # ── Analyze ────────────────────────────────────────────
        results = analyze(text)
        if args.no_llm:
            results["semantic_supply_chain"] = {"entities": [], "skipped": "no-llm flag"}
        elif results.get("third_party_sharing", {}).get("detected", False):
            print("    Sending to LLM for Semantic Extraction...")

            # Feed the FULL policy text to the extractor. The previous
            # truncated examples[:3] feed (~600 chars) starved Nemotron and
            # produced 0-1 entities/company. semantic_extractor.py now chunks
            # the full text internally, under the model's context budget.
            policy_text = results.get("full_text", "") or " ".join(
                results.get("third_party_sharing", {}).get("examples", []))

            # Run the LLM and add the data to your results
            results["semantic_supply_chain"] = extract_supply_chain(policy_text, name)
        else:
            results["semantic_supply_chain"] = {"entities": []}

        print(f"  Grade     : {results['readability']['grade_level']}  "
              f"|  Ease: {results['readability']['reading_ease']}  "
              f"|  Cats: {len(results['data_categories'])}")
        print(f"  Consent   : {results['consent_model']['model']}")
        print(f"  Sections  : {results.get('section_count', 0)}  "
              f"|  Date: {results.get('policy_date') or 'not found'}")

        # Flag low-quality scrapes (saved but noted)
        if wc < MIN_WORDS_QUALITY:
            print(f"  ⚑ Low word count ({wc}) — saved but flagged as low quality")
            low_quality.append({"company": name, "word_count": wc, "source": source})

        # ── Save JSON ──────────────────────────────────────────
        out = {
            "company":    name,
            "category":   category,
            "source_url": primary_url,
            "data_source": source,
            "scraped_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "analysis":   results
        }
        filepath = os.path.join(OUTPUT_DIR, safe_filename(name) + ".json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved   → {filepath}\n")

        csv_rows.append(flatten_for_csv(name, primary_url, category, source_label, results))

        if i < len(COMPANIES):
            time.sleep(args.delay)

    # ── Outputs ────────────────────────────────────────────────
    if csv_rows:
        csv_path = os.path.join(OUTPUT_DIR, "summary.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Summary CSV   → {csv_path}  ({len(csv_rows)} companies)")

    log = {"collected": len(csv_rows), "failed": failed, "low_quality": low_quality}
    log_path = os.path.join(OUTPUT_DIR, "collection_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    print(f"Log           → {log_path}")

    print(f"\n{'='*65}")
    print(f"  Collected : {len(csv_rows)}  |  Failed: {len(failed)}  "
          f"|  Low quality: {len(low_quality)}")
    if failed:
        print(f"\n  Still failing after all strategies:")
        for f_ in failed:
            print(f"    ✗  {f_['company']}: {f_['reason'][:70]}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
