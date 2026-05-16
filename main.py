#!/usr/bin/env python3
# main.py
# Entry point for the Privacy Policy Analyzer.
#
# Usage:
#   python main.py <URL>
#   python main.py <URL> --json report.json
#
# Examples:
#   python main.py https://policies.google.com/privacy
#   python main.py https://www.apple.com/legal/privacy/ --json apple_report.json

import argparse
import sys

from privacy_analyzer.scraper import fetch_policy
from privacy_analyzer.analyzer import analyze
from privacy_analyzer.reporter import print_report, save_json


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="privacy-analyzer",
        description=(
            "Privacy Policy Analyzer\n"
            "Fetches a privacy policy from a URL and analyzes it for:\n"
            "  • Readability (how hard it is for users to understand)\n"
            "  • Data categories collected\n"
            "  • Consent model (opt-in vs opt-out)\n"
            "  • Third-party data sharing\n"
            "  • Data retention language\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "url",
        help="Full URL of the privacy policy page to analyze"
    )
    parser.add_argument(
        "--json",
        metavar="FILEPATH",
        help="Also save results as a JSON file (e.g. --json report.json)"
    )

    args = parser.parse_args()

    # ── Fetch ──────────────────────────────────────────────────
    print(f"\nFetching: {args.url}")
    try:
        text = fetch_policy(args.url)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    if len(text.split()) < 100:
        print(
            "Warning: Retrieved text is very short. "
            "The page may require JavaScript to render, "
            "or the URL may not point directly to a privacy policy.",
            file=sys.stderr
        )

    # ── Analyze ────────────────────────────────────────────────
    print("Analyzing...")
    results = analyze(text)

    # ── Report ─────────────────────────────────────────────────
    print_report(args.url, results)

    if args.json:
        save_json(args.url, results, args.json)


if __name__ == "__main__":
    main()
