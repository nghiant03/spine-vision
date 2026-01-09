"""Fuzzy matching utilities for patient identification and folder matching."""

import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from loguru import logger
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


# Regex for matching patient image folder names
IMAGE_FOLDER_REGEX = re.compile(r"^[A-Z_]+(_\d{4})?_\d{8}( \(\d+\))?$")


class FolderInfo(TypedDict):
    """Information extracted from a patient image folder name."""

    path: Path
    name_part: str
    birth_year: str | None


def parse_image_folder_name(folder_name: str) -> tuple[str, str | None]:
    """Parse patient name and birth year from folder name.

    Expected format: PATIENT_NAME_YYYY_YYYYMMDD( (N))?
    - PATIENT_NAME: Uppercase with underscores
    - YYYY (optional): Birth year
    - YYYYMMDD: Study date
    - (N) (optional): Duplicate counter

    Args:
        folder_name: Name of the image folder.

    Returns:
        Tuple of (name_part without underscores, birth_year or None).
    """
    base_name = re.sub(r" \(\d+\)$", "", folder_name)
    parts = base_name.split("_")

    if len(parts) >= 3 and re.fullmatch(r"\d{4}", parts[-2]):
        name_part = "".join(parts[:-2])
        birth_year = parts[-2]
    else:
        name_part = "".join(parts[:-1])
        birth_year = None

    return name_part, birth_year


def build_folder_lookup(image_path: Path) -> dict[str, FolderInfo]:
    """Build a lookup dictionary of patient image folders.

    Args:
        image_path: Root directory containing patient folders.

    Returns:
        Dictionary mapping folder keys to FolderInfo.
    """
    folder_dict: dict[str, FolderInfo] = {}

    for path in image_path.rglob("*"):
        if not path.is_dir() or not IMAGE_FOLDER_REGEX.match(path.name):
            continue

        name_part, birth_year = parse_image_folder_name(path.name)

        if birth_year:
            key = f"{name_part}_{birth_year}"
        else:
            key = name_part

        folder_dict[key] = {
            "path": path,
            "name_part": name_part,
            "birth_year": birth_year,
        }

    return folder_dict


def find_matching_folder(
    patient_name: str,
    patient_birthday: str,
    folder_map: dict[str, FolderInfo],
    threshold: float = 85,
    date_format: str = "%d/%m/%Y",
) -> Path | None:
    """Find the best matching image folder for a patient.

    Matches by name similarity, with birth year as tiebreaker.

    Args:
        patient_name: Patient's name from OCR.
        patient_birthday: Patient's birthday string.
        folder_map: Dictionary from build_folder_lookup().
        threshold: Minimum fuzzy match score.
        date_format: Format of patient_birthday string.

    Returns:
        Path to matching folder, or None if no match found.
    """
    try:
        patient_birth_year = datetime.strptime(patient_birthday, date_format).year
    except ValueError:
        logger.warning(f"Could not parse birthday: {patient_birthday}")
        patient_birth_year = None

    candidates = []

    for key, data in folder_map.items():
        score = fuzz.partial_ratio(patient_name, data["name_part"])

        if score > threshold:
            candidates.append(
                {
                    "key": key,
                    "score": score,
                    "birth_year": data["birth_year"],
                    "path": data["path"],
                }
            )

    if not candidates:
        return None

    candidates.sort(key=lambda x: x["score"], reverse=True)
    best_score = candidates[0]["score"]
    top_matches = [c for c in candidates if c["score"] == best_score]

    if patient_birth_year:
        for match in top_matches:
            if match["birth_year"] == str(patient_birth_year):
                return match["path"]

    for match in top_matches:
        if match["birth_year"] is None:
            return match["path"]

    return top_matches[0]["path"] if top_matches else None


def find_matching_folder_by_name(
    patient_name: str,
    folder_map: dict[str, FolderInfo],
    threshold: float = 85,
) -> Path | None:
    """Find the best matching image folder by name only.

    Matches by name similarity without birth year tiebreaker.
    Used when birthday is not available.

    Args:
        patient_name: Patient's name.
        folder_map: Dictionary from build_folder_lookup().
        threshold: Minimum fuzzy match score.

    Returns:
        Path to matching folder, or None if no match found.
    """
    candidates = []

    for key, data in folder_map.items():
        score = fuzz.partial_ratio(patient_name, data["name_part"])

        if score > threshold:
            candidates.append(
                {
                    "key": key,
                    "score": score,
                    "path": data["path"],
                }
            )

    if not candidates:
        return None

    # Return the best match by score
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[0]["path"]


class PatientMatcher:
    """Matches patients from OCR data to image folders.

    Stateful wrapper around folder matching logic.
    """

    def __init__(
        self,
        image_path: Path,
        threshold: float = 85,
        date_format: str = "%d/%m/%Y",
    ) -> None:
        """Initialize patient matcher.

        Args:
            image_path: Root directory containing patient image folders.
            threshold: Minimum fuzzy match score for matching.
            date_format: Expected date format for birthdays.
        """
        self.threshold = threshold
        self.date_format = date_format
        self.folder_map = build_folder_lookup(image_path)
        logger.info(f"Built folder lookup with {len(self.folder_map)} entries")

    def match(self, patient_name: str, patient_birthday: str) -> Path | None:
        """Find matching folder for a patient using name and birthday.

        Args:
            patient_name: Patient's name.
            patient_birthday: Patient's birthday string.

        Returns:
            Path to matching image folder, or None.
        """
        return find_matching_folder(
            patient_name,
            patient_birthday,
            self.folder_map,
            self.threshold,
            self.date_format,
        )

    def match_by_name(self, patient_name: str) -> Path | None:
        """Find matching folder for a patient using name only.

        Used when birthday is not available (e.g., patient-named reports).

        Args:
            patient_name: Patient's name.

        Returns:
            Path to matching image folder, or None.
        """
        return find_matching_folder_by_name(
            patient_name,
            self.folder_map,
            self.threshold,
        )
