# reporter.py
# Formats analysis results for terminal output and optional JSON export.

import json


def _section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def print_report(url: str, results: dict) -> None:
    """Print a structured, readable report to the terminal."""

    print("\n" + "=" * 60)
    print("  PRIVACY POLICY ANALYZER")
    print("=" * 60)
    print(f"  Source : {url}")
    print(f"  Length : {results['readability']['word_count']:,} words  |  "
          f"{results['sentence_count']} sentences")

    # ── Readability ────────────────────────────────────────────
    _section("📖  READABILITY")
    r = results["readability"]
    print(f"  Grade Level   : {r['grade_level']}  ({r['description']})")
    print(f"  Reading Ease  : {r['reading_ease']} / 100  "
          f"({'easier' if r['reading_ease'] >= 50 else 'harder'} than average)")
    print()
    print("  ⚑ For reference: average US adult reads at ~8th grade level.")
    print("    Most privacy policies are written at 12–16 grade level.")

    # ── Data categories ────────────────────────────────────────
    _section("📦  DATA CATEGORIES DETECTED")
    cats = results["data_categories"]
    if cats:
        for cat, examples in cats.items():
            kw = examples[0]["keyword"] if examples else "—"
            ex = examples[0]["sentence"] if examples else ""
            print(f"\n  ✓  {cat}  (trigger: '{kw}')")
            if ex:
                print(f"     └─ \"{ex}\"")
    else:
        print("  No specific data categories detected.")

    # ── Consent model ──────────────────────────────────────────
    _section("🔐  CONSENT MODEL")
    c = results["consent_model"]
    print(f"  Classification: {c['model']}")
    if c["opt_in_examples"]:
        print(f"\n  Opt-In language found:")
        print(f"  └─ \"{c['opt_in_examples'][0]}\"")
    if c["opt_out_examples"]:
        print(f"\n  Opt-Out language found:")
        print(f"  └─ \"{c['opt_out_examples'][0]}\"")

    # ── Third-party ────────────────────────────────────────────
    _section("🔗  THIRD-PARTY DATA SHARING")
    tp = results["third_party_sharing"]
    status = "⚠  YES — sharing language detected" if tp["detected"] else "✓  Not explicitly mentioned"
    print(f"  Status: {status}")
    for ex in tp["examples"][:2]:
        print(f"  └─ \"{ex}\"")

    # ── Retention ──────────────────────────────────────────────
    _section("🗓   DATA RETENTION")
    ret = results["retention_policy"]
    if ret["detected"]:
        print("  Status: Retention period mentioned")
        for ex in ret["examples"]:
            print(f"  └─ \"{ex}\"")
    else:
        print("  Status: ⚠  No explicit retention period stated")
        print("  Note  : Absence of retention language means data may be kept indefinitely.")

    print("\n" + "=" * 60 + "\n")


def save_json(url: str, results: dict, filepath: str = "report.json") -> None:
    """Serialize results to a JSON file for downstream use."""
    output = {"source_url": url, "analysis": results}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  JSON report saved → {filepath}")
