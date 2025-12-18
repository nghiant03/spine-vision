"""Fuzzy string matching utilities."""

from rapidfuzz import fuzz
from unidecode import unidecode


def fuzzy_match_score(text1: str, text2: str, normalize: bool = True) -> float:
    """Calculate fuzzy match score between two strings.

    Args:
        text1: First string to compare.
        text2: Second string to compare.
        normalize: If True, normalize strings (lowercase, strip, unidecode).

    Returns:
        Match score from 0 to 100.
    """
    if normalize:
        text1 = unidecode(text1).lower().strip()
        text2 = unidecode(text2).lower().strip()
    return fuzz.partial_ratio(text1, text2)


def fuzzy_value_extract(
    text_lines: list[str],
    field: str,
    threshold: float = 80,
    window_length: int = 2,
) -> str | None:
    """Extract a field value from OCR text lines using fuzzy matching.

    Searches for a field pattern (e.g., "Ho ten nguoi benh") and extracts
    the value that follows it.

    Args:
        text_lines: List of OCR text strings.
        field: Field name pattern to search for (e.g., "Ngay sinh").
        threshold: Minimum fuzzy match score (0-100) to consider a match.
        window_length: Number of extra words to consider when matching field.

    Returns:
        Extracted value string (uppercase), or None if not found.
    """
    field = field.lower()

    for line in text_lines:
        normalized_text = unidecode(line).lower().strip()
        score = fuzz.partial_ratio(field, normalized_text)

        if score <= threshold:
            continue

        key_word_count = len(field.split())
        words = normalized_text.split()

        if len(words) < key_word_count:
            continue

        min_len = max(1, key_word_count - 1)
        max_len = min(len(words), key_word_count + window_length)

        best_score = 0
        best_end_index = 0

        for i in range(min_len, max_len + 1):
            candidate_key = " ".join(words[:i])
            candidate_clean = candidate_key.rstrip(" :.-")
            score = fuzz.ratio(field, candidate_clean.lower())

            if score > best_score:
                best_score = score
                best_end_index = i

        if best_score >= threshold:
            value_part = "".join(words[best_end_index:])
            return value_part.lstrip(".:;").upper()

    return None


def fuzzy_find_best_match(
    query: str,
    candidates: list[str],
    threshold: float = 80,
    normalize: bool = True,
) -> tuple[str | None, float]:
    """Find the best fuzzy match from a list of candidates.

    Args:
        query: String to match.
        candidates: List of candidate strings.
        threshold: Minimum score to consider a match.
        normalize: Whether to normalize strings before matching.

    Returns:
        Tuple of (best matching candidate or None, match score).
    """
    best_match = None
    best_score = 0.0

    for candidate in candidates:
        score = fuzzy_match_score(query, candidate, normalize)
        if score > best_score:
            best_score = score
            best_match = candidate

    if best_score >= threshold:
        return best_match, best_score
    return None, best_score
