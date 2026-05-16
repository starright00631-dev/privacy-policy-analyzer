# scraper.py
# Three-layer fetch: live → fallback URLs → Wayback Machine.
# Each layer is only accepted if it returns >= MIN_WORDS words.
# If a layer returns text but it's too short, the next layer is tried.

from playwright.sync_api import sync_playwright
import os
import re
import time
import requests
from bs4 import BeautifulSoup
from readability import Document
"""
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
    
}
"""

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Sec-Ch-Ua": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"'
}

NOISE_TAGS = ["nav", "footer", "script", "style", "header", "aside"]

# Any layer returning fewer than this many words is rejected and the next tried
MIN_WORDS = 300


def _extract_text(html: str) -> str:
    """Dual-layer extraction: Readability first, BeautifulSoup fallback."""

    # --- LAYER 1: READABILITY (Clean Text) ---
    try:
        doc = Document(html)
        clean_html = doc.summary()
        soup = BeautifulSoup(clean_html, "html.parser")

        # Strip Wayback Machine toolbar
        for tag in soup.find_all(id=re.compile(r"wm-ipp|wm_tb", re.I)):
            tag.decompose()

        content = soup.get_text(separator="\n", strip=True)

        # If Readability successfully found the core article, return it!
        if len(content.split()) > 300:
            return content
    except Exception as e:
        pass  # If Readability fails, quietly move to Layer 2

    # --- LAYER 2: BEAUTIFULSOUP FALLBACK (For IBM, Meta, etc.) ---
    # If we get here, Readability stripped too much (e.g., IBM 10 words, Meta 281 words)
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(id=re.compile(r"wm-ipp|wm_tb", re.I)):
        tag.decompose()

    for tag in soup(NOISE_TAGS):
        tag.decompose()

    fallback_content = (
            soup.find("main")
            or soup.find("article")
            or soup.find(id="content")
            or soup.find(id="privacy-policy")
            or soup.find(class_="privacy-policy")
            or soup.body
    )

    if fallback_content is None:
        return ""

    return fallback_content.get_text(separator="\n", strip=True)

# Minimal headers for servers that reject complex Accept headers (fixes 406)
HEADERS_MINIMAL = {
    "User-Agent": HEADERS["User-Agent"],
    "Accept": "text/html",
    "Accept-Language": "en-US,en;q=0.9",
}


def _fetch_raw(url: str, timeout: int = 30, extra_wait_ms: int = 8000, headless: bool = True) -> str:
    """
    JS-Rendering Fetch Engine using Playwright.
    Headless by default. extra_wait_ms can be tuned for slow JS-heavy pages.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 900},
            locale="en-US",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        # Hide webdriver flag (basic anti-bot evasion for Cloudflare/Akamai)
        context.add_init_script(
            "Object.defineProperty(navigator, 'webdriver', { get: () => undefined });"
        )
        page = context.new_page()

        try:
            response = page.goto(url, wait_until="domcontentloaded", timeout=timeout * 1000)

            # Try networkidle for SPAs but don't fail if it never settles
            try:
                page.wait_for_load_state("networkidle", timeout=10000)
            except Exception:
                pass

            page.wait_for_timeout(extra_wait_ms)

            # Cookie wall bypass for Meta / Google / others
            try:
                page.locator(
                    "button:has-text('Allow all cookies'), "
                    "button:has-text('Accept All'), "
                    "button:has-text('Accept all'), "
                    "button:has-text('Accept'), "
                    "button:has-text('I Accept'), "
                    "button:has-text('Agree'), "
                    "button[aria-label*='Accept']"
                ).first.click(timeout=2500)
                page.wait_for_timeout(1500)
            except Exception:
                pass

            if response is None:
                raise RuntimeError("Failed to get a response")

            if response.status == 429:
                raise RuntimeError("HTTP 429 Rate Limit")
            elif response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}")

            # Try scrolling to trigger lazy-loaded content
            try:
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1500)
                page.evaluate("window.scrollTo(0, 0)")
            except Exception:
                pass

            html_content = page.content()
            return _extract_text(html_content)

        except Exception as e:
            raise RuntimeError(f"Playwright Error: {str(e)}")
        finally:
            try:
                browser.close()
            except Exception:
                pass


def _wayback_lookup(url: str) -> str:
    """
    Find the URL of a recent Wayback Machine snapshot for `url`.
    Tries the availability API first (no proxy), then falls back to the CDX API
    which can surface older snapshots when the most-recent one isn't available.
    """
    proxy_envs = [None]
    if os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"):
        proxy_envs.append({
            "http": os.environ.get("HTTP_PROXY"),
            "https": os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
        })

    last_err = None

    # Strategy A: simple availability API
    avail = f"https://archive.org/wayback/available?url={url}"
    for proxies in proxy_envs:
        for attempt in range(2):
            try:
                r = requests.get(
                    avail,
                    headers={"User-Agent": HEADERS["User-Agent"]},
                    proxies=proxies,
                    timeout=15,
                )
                r.raise_for_status()
                snap = r.json().get("archived_snapshots", {}).get("closest", {}) or {}
                if snap.get("url"):
                    return snap["url"]
                break  # 200 OK but no snapshot → try CDX next
            except Exception as e:
                last_err = e
                time.sleep(1.5)

    # Strategy B: CDX API — search any captures, take the most recent OK one
    cdx = (
        "https://web.archive.org/cdx/search/cdx"
        f"?url={url}&output=json&limit=-5&filter=statuscode:200"
    )
    for proxies in proxy_envs:
        for attempt in range(2):
            try:
                r = requests.get(
                    cdx,
                    headers={"User-Agent": HEADERS["User-Agent"]},
                    proxies=proxies,
                    timeout=20,
                )
                r.raise_for_status()
                rows = r.json()
                if len(rows) > 1:
                    # rows[0] is the header; pick the latest by timestamp
                    latest = sorted(rows[1:], key=lambda row: row[1])[-1]
                    timestamp, original = latest[1], latest[2]
                    return f"https://web.archive.org/web/{timestamp}/{original}"
                break
            except Exception as e:
                last_err = e
                time.sleep(1.5)

    # Strategy C: redirect resolution. /web/<year>/<url> → latest snapshot for
    # that year. Works when both API endpoints are throttled or down.
    from datetime import datetime as _dt
    year_now = _dt.utcnow().year
    for year in (year_now, year_now - 1, year_now - 2):
        redirect_url = f"https://web.archive.org/web/{year}/{url}"
        for proxies in proxy_envs:
            try:
                r = requests.head(
                    redirect_url,
                    headers={"User-Agent": HEADERS["User-Agent"]},
                    proxies=proxies,
                    timeout=15,
                    allow_redirects=True,
                )
                final = r.url
                # A real snapshot URL contains /web/<14-digit timestamp>/
                if re.search(r"/web/\d{14}/", final):
                    return final
            except Exception as e:
                last_err = e

    raise RuntimeError(f"No Wayback snapshot found ({last_err})")


def _fetch_static(url: str, timeout: int = 25) -> str:
    """
    Plain-requests fetch + extraction. For static archives (Wayback) this is
    far more reliable than Playwright, which often executes JS that nukes the
    archived DOM (Framer/Next.js sites in particular).
    """
    proxy_envs = [None]
    if os.environ.get("HTTP_PROXY") or os.environ.get("HTTPS_PROXY"):
        proxy_envs.append({
            "http": os.environ.get("HTTP_PROXY"),
            "https": os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
        })

    last_err = None
    for proxies in proxy_envs:
        try:
            r = requests.get(url, headers=HEADERS, proxies=proxies, timeout=timeout, allow_redirects=True)
            if r.status_code == 429:
                raise RuntimeError("HTTP 429 Rate Limit")
            if r.status_code >= 400:
                raise RuntimeError(f"HTTP {r.status_code}")
            # Use apparent_encoding when the server lies about charset
            if not r.encoding or r.encoding.lower() in ("iso-8859-1", "ascii"):
                r.encoding = r.apparent_encoding or "utf-8"
            return _extract_text(r.text)
        except Exception as e:
            last_err = e
            time.sleep(1.0)
    raise RuntimeError(f"Static fetch failed: {last_err}")


def _wayback_fallback(url: str) -> tuple:
    """
    Query Wayback for a snapshot of `url`. Try a static (requests) extraction
    first because Wayback wraps pages in wombat.js, which often breaks
    Framer/Next.js sites when re-rendered via Playwright. If static fails,
    fall back to Playwright rendering.
    Returns (text, snapshot_url). Raises RuntimeError on failure.
    """
    snapshot_url = _wayback_lookup(url)
    errs = []

    # Strategy A: static requests extraction (best for archived HTML)
    try:
        text = _fetch_static(snapshot_url)
        if len(text.split()) >= MIN_WORDS:
            return text, snapshot_url
        errs.append(f"static: only {len(text.split())} words")
    except Exception as e:
        errs.append(f"static: {e}")

    # Strategy B: Playwright rendering (for snapshots that need JS)
    try:
        text = _fetch_raw(snapshot_url, extra_wait_ms=10000)
        return text, snapshot_url
    except Exception as e:
        errs.append(f"playwright: {e}")

    raise RuntimeError(f"Wayback render failed for {snapshot_url}: {' | '.join(errs)}")

'''
def _wayback_fallback(url: str) -> tuple:
    """
    Query Wayback Machine CDX API for most recent snapshot.
    Returns (text, snapshot_url). Raises RuntimeError on any failure.
    """
    cdx = f"https://archive.org/wayback/available?url={url}"
    try:
        r = requests.get(cdx, headers=HEADERS, timeout=15)
        r.raise_for_status()
        snap = r.json().get("archived_snapshots", {}).get("closest", {})
        snapshot_url = snap.get("url")
        if not snapshot_url:
            raise RuntimeError("No Wayback snapshot found")
        text = _fetch_raw(snapshot_url)
        return text, snapshot_url
    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Wayback error: {e}")
# 2nd version
def _wayback_fallback(url: str) -> tuple:
    """
    Query Wayback Machine CDX API for most recent snapshot.
    Returns (text, snapshot_url). Raises RuntimeError on any failure.
    """
    cdx = f"https://archive.org/wayback/available?url={url}"
    try:
        # STEP 1: Use 'requests' for the fast API ping
        # Note: We use a basic header here just to get the JSON
        r = requests.get(cdx, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()

        snap = r.json().get("archived_snapshots", {}).get("closest", {})
        snapshot_url = snap.get("url")

        if not snapshot_url:
            raise RuntimeError("No Wayback snapshot found")

        # STEP 2: Use your NEW Playwright fetch engine to render the heavy JS page!
        text = _fetch_raw(snapshot_url)

        return text, snapshot_url

    except RuntimeError:
        raise
    except Exception as e:
        raise RuntimeError(f"Wayback error: {e}")
'''

def fetch_policy(primary_url: str,
                 fallback_urls: list = None,
                 use_wayback: bool = True,
                 retry_on_429: int = 2) -> tuple:
    """
    Fetch a privacy policy with three-layer fallback.
    Each layer is only accepted if it returns >= MIN_WORDS words.
    Short results (index pages, stub pages) are rejected and the next layer tried.

    Returns (text, source_label) where source_label is 'live', 'fallback', or 'wayback'.
    Raises RuntimeError if all strategies fail or return too little text.
    """
    errors = []
    fallback_urls = fallback_urls or []

    # ── Layer 1: Primary URL ───────────────────────────────────
    for attempt in range(1 + retry_on_429):
        try:
            text = _fetch_raw(primary_url)
            wc = len(text.split())
            if wc >= MIN_WORDS:
                return text, "live"
            else:
                errors.append(f"primary: only {wc} words (stub/index page)")
                break   # short result — don't retry, move on
        except RuntimeError as e:
            msg = str(e)
            if "429" in msg and attempt < retry_on_429:
                wait = 10 * (attempt + 1)
                print(f"    429 rate limit. Waiting {wait}s...")
                time.sleep(wait)
            else:
                errors.append(f"primary: {msg}")
                break

    # ── Layer 2: Fallback URLs ─────────────────────────────────
    for fb in fallback_urls:
        try:
            text = _fetch_raw(fb)
            wc = len(text.split())
            if wc >= MIN_WORDS:
                return text, "fallback"
            else:
                errors.append(f"fallback({fb}): only {wc} words")
        except RuntimeError as e:
            errors.append(f"fallback({fb}): {e}")

    # ── Layer 3: Wayback Machine ───────────────────────────────
    if use_wayback:
        try:
            text, snap_url = _wayback_fallback(primary_url)
            wc = len(text.split())
            if wc >= MIN_WORDS:
                return text, "wayback"
            else:
                errors.append(f"wayback: only {wc} words from {snap_url}")
        except RuntimeError as e:
            errors.append(f"wayback: {e}")

    raise RuntimeError(" | ".join(errors))
