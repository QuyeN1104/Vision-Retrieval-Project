import os
import csv
import uuid
from typing import List
from .datatypes import ImageRecord

def load_dataset(data_dir: str) -> List[ImageRecord]:
    """
    Scans an image directory and returns a list of ImageRecords.
    Assumes image files have extensions like .jpg, .jpeg, .png.
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    records = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                file_path = os.path.join(root, file)
                record_id = str(uuid.uuid4())
                records.append(ImageRecord(id=record_id, path=file_path))
    return records

def load_from_manifest(csv_path: str) -> List[ImageRecord]:
    """
    Loads image records from a CSV manifest.
    Expected CSV columns: path, label, caption, split
    """
    records = []
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_id = str(uuid.uuid4())
            records.append(
                ImageRecord(
                    id=record_id,
                    path=row.get("path", ""),
                    label=row.get("label"),
                    caption=row.get("caption"),
                    split=row.get("split")
                )
            )
    return records
