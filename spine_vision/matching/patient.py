"""Patient identification and folder matching."""

import re
from datetime import datetime
from pathlib import Path
from typing import TypedDict

from loguru import logger
from rapidfuzz import fuzz

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
