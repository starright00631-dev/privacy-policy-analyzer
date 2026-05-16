# readability.py
# Self-contained readability metrics — no NLTK or cmudict required.
# Implements Coleman-Liau Index and a simple Flesch Reading Ease approximation.

import re


def count_words(text: str) -> int:
    return len(text.split())


def count_sentences(text: str) -> int:
    sentences = re.split(r"[.!?]+", text)
    return max(1, len([s for s in sentences if s.strip()]))


def count_letters(text: str) -> int:
    return len(re.sub(r"[^a-zA-Z]", "", text))


def count_syllables_approx(word: str) -> int:
    """
    Approximate syllable count per word using vowel-group heuristic.
    Not perfect, but sufficient for readability estimation without a dictionary.
    """
    word = word.lower().strip(".,;:!?\"'")
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent 'e' at end
    if word.endswith("e") and len(word) > 2:
        count = max(1, count - 1)
    return max(1, count)


def coleman_liau_index(text: str) -> float:
    """
    Coleman-Liau readability index.
    Based purely on character counts — no syllable counting needed.
    Typical range: 1–17 (maps to US grade levels).
    """
    words = count_words(text)
    sentences = count_sentences(text)
    letters = count_letters(text)

    if words == 0:
        return 0.0

    L = (letters / words) * 100       # avg letters per 100 words
    S = (sentences / words) * 100     # avg sentences per 100 words

    return round(0.0588 * L - 0.296 * S - 15.8, 1)


def flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease score (0–100).
    Higher = easier. Plain English ≈ 60–70.
    Legal/academic text often scores 20–40.
    """
    words_list = text.split()
    words = max(1, len(words_list))
    sentences = count_sentences(text)
    syllables = sum(count_syllables_approx(w) for w in words_list)

    score = (
        206.835
        - 1.015 * (words / sentences)
        - 84.6 * (syllables / words)
    )
    return round(max(0.0, min(100.0, score)), 1)
