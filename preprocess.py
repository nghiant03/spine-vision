import logging
import re
import shutil
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tyro
from loguru import logger
from openpyxl.reader.excel import SUPPORTED_FORMATS
from paddleocr import TextDetection
from PIL import Image
from rapidfuzz import fuzz 
from tqdm.rich import tqdm
from unidecode import unidecode
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

from config import PreprocessConfig

EXCEL_FORMATS = (*SUPPORTED_FORMATS, ".xls")
ONE_HOT_COL = "Modic"
NAME_FIELD_PATTERN = "Ho ten nguoi benh"
BIRTHDAY_FIELD_PATTERN = "Ngay sinh"
IMAGE_FOLDER_REGEX = re.compile(r"^[A-Z_]+(_\d{4})?_\d{8}( \(\d+\))?$")


def load_tabular_data(config: PreprocessConfig) -> pd.DataFrame:
    table_path = config.table_path
    files_data = []

    valid_files = (
        p for p in table_path.rglob("*") 
        if p.is_file() and p.name not in config.exclude_files
    )

    for file_path in valid_files:
        match file_path.suffix:
            case ".csv":
                files_data.append(pd.read_csv(file_path))
            case suffix if suffix in EXCEL_FORMATS:
                files_data.append(pd.read_excel(file_path))
            case _:
                logger.warning(f"Unsupported format: {file_path}")

    if not files_data:
        return pd.DataFrame()

    df = pd.concat(files_data)
    
    initial_size = df.size
    df = df.drop_duplicates()
    logger.debug(f"Dropped {initial_size - df.size} duplicates.")
    
    na_count = df.isna().any(axis=1).sum()
    df = df.dropna()
    logger.debug(f"Dropped {na_count} rows with NA.")
    
    df = df[~df[config.id_col].isin(config.corrupted_ids)]
    
    dummies = (
        df[ONE_HOT_COL]
        .astype(str)
        .str.replace(r"\.0\b", "", regex=True)
        .str.get_dummies(sep="&")
        .add_prefix(f"{ONE_HOT_COL}_")
    )
    
    df = pd.concat([df, dummies], axis=1).drop(columns=ONE_HOT_COL)
    logger.info("Loaded tabular data.")
    return df.astype(int)

def crop_polygon(image_np: np.ndarray, points: np.ndarray) -> Image.Image:
    points = points.astype(np.float32)
    (tl, tr, br, bl) = points

    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))

    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(points, dst)
    warped = cv2.warpPerspective(image_np, M, (max_width, max_height))

    return Image.fromarray(warped)

def fuzzy_value_extract(text_lines: list[str], field: str, threshold: float, window_length: int) -> str | None:
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

def find_matching_folder(patient_name: str, patient_birthday: str, folder_map: dict[str, dict], threshold: float) -> Path | None:
    candidates = []
    patient_birth_year = datetime.strptime(patient_birthday, "%d/%m/%Y").year
    
    for key, data in folder_map.items():
        key_name = data["name_part"]
        score = fuzz.partial_ratio(patient_name, key_name)

        if score > threshold:
            candidates.append({
                "key": key,
                "score": score,
                "birth_year": data["birth_year"],
                "path": data["path"]
            })

    if not candidates:
        return None
    
    candidates.sort(key=lambda x: x["score"], reverse=True)
    
    best_score = candidates[0]["score"]
    top_matches = [candidate for candidate in candidates if candidate["score"] == best_score]

    if patient_birth_year:
        for match in top_matches:
            if match["birth_year"] == str(patient_birth_year):
                return match["path"] 

    for match in top_matches:
        if match["birth_year"] is None:
            return match["path"]

    return None

def main(config: PreprocessConfig):
    log_level = logging.DEBUG if config.verbose else logging.INFO

    logger.remove()
    logger.add(
        lambda msg: tqdm.write(msg, end=""),
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=log_level
    )
    
    if config.enable_file_log:
        config.log_path.mkdir(parents=True, exist_ok=True)
        logger.add(
            config.log_path / "preprocess.log",
            level="DEBUG",
            rotation="10 MB",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{line} | {message}",
            encoding="utf-8"
        )

    logger.debug("Started preprocessing.")
    label_data = load_tabular_data(config)
    if label_data.empty:
        logger.info(f"No valid data found at {config.table_path}")
        return

    config.output_path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Unique Patients: {label_data[config.id_col].nunique()}")

    device = "cuda:0" if config.use_gpu else "cpu"
    logger.info("Loading detection model.")
    detection_model = TextDetection(model_name="PP-OCRv5_server_det")

    recognition_config = Cfg.load_config_from_name("vgg_transformer")
    
    recognition_config["device"] = device
    recognition_config["cnn"]["pretrained"] = False
    recognition_config["predictor"]["beamsearch"] = False
    
    logger.info("Loading recognition model.")
    recognition_model = Predictor(recognition_config)

    report_lookup = {
        path.stem: path for path in config.report_path.rglob("*.png")
    }
    
    image_folder_dict = {}

    for path in config.image_path.rglob("*"):
        if path.is_dir() and IMAGE_FOLDER_REGEX.match(path.name):
            base_name = re.sub(r" \(\d+\)$", "", path.name)
            parts = base_name.split("_")
            
            if len(parts) >= 3 and re.fullmatch(r"\d{4}", parts[-2]):
                name_part = "".join(parts[:-2])
                birth_year = parts[-2]
                key = f"{name_part}_{birth_year}"
            else:
                name_part = "".join(parts[:-1])
                birth_year = None
                key = name_part
            
            image_folder_dict[key] = {
                "path": path,
                "name_part": name_part,
                "birth_year": birth_year
            }

    unique_ids = label_data[config.id_col].unique()
    
    matched_ids = []
    for unique_id in tqdm(unique_ids, desc="Processing Patients", unit="id"):        
        if not (report_path := report_lookup.get(str(unique_id))):
            continue

        if not (detection_result := detection_model.predict(report_path.as_posix())):
            continue

        boxes = detection_result[0]["dt_polys"]
        image = Image.open(report_path).convert("RGB")
        image_np = np.array(image) 

        text_lines = []
        for box in boxes:
            box = np.array(box).astype(np.int32)
            
            crop = crop_polygon(image_np, box)
            
            text = recognition_model.predict(crop)
            text_lines.append(text)

        if not (patient_name := fuzzy_value_extract(text_lines, NAME_FIELD_PATTERN, config.report_fuzzy_threshold, 3)):
            logger.warning(f"Could not extract name for ID {unique_id}")
            continue

        if not (patient_birthday := fuzzy_value_extract(text_lines, BIRTHDAY_FIELD_PATTERN, config.report_fuzzy_threshold, 2)):
            logger.warning(f"Could not extract birthday for ID {unique_id}")
            continue


        if best_folder := find_matching_folder(patient_name, patient_birthday, image_folder_dict, config.image_fuzzy_threshold):
            dest = config.output_image_path / str(unique_id)
            shutil.copytree(best_folder, dest, dirs_exist_ok=True)
            logger.info(f"Copied {best_folder.name} -> {dest}")
            matched_ids.append(unique_id)
        else:
            logger.warning(f"No matching folder found for name '{patient_name}' (ID: {unique_id})")

    label_data = label_data[label_data[config.id_col].isin(matched_ids)]
    label_data.to_csv(config.output_table_path, index=False)
    logger.info(f"Saved table to {config.output_table_path}")

if __name__ == "__main__":
    preprocess_config = tyro.cli(PreprocessConfig)
    main(preprocess_config)
