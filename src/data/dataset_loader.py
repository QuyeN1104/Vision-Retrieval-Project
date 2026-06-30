import os
import csv
import uuid
import random
from typing import List, Tuple
from .datatypes import ImageRecord

def load_dataset(data_dir: str) -> List[ImageRecord]:
    """
    Recursively scans an image directory and returns a list of ImageRecords.

    Label is automatically derived from the immediate parent directory name
    of each image file (e.g. ``data/L28_V001/001.jpg`` → label ``L28_V001``).
    If the image sits directly in *data_dir* (no subdirectory), label is None.
    """
    valid_extensions = {".jpg", ".jpeg", ".png"}
    data_dir = os.path.abspath(data_dir)
    project_root = os.getcwd()
    records = []
    for root, _, files in os.walk(data_dir):
        for file in sorted(files):
            ext = os.path.splitext(file)[1].lower()
            if ext in valid_extensions:
                abs_file_path = os.path.join(root, file)
                # Store paths relative to project root for portability
                try:
                    file_path = os.path.relpath(abs_file_path, project_root)
                    # If it starts with '..', it's outside the project, keep it absolute
                    if file_path.startswith(".."):
                        file_path = abs_file_path
                except ValueError:
                    file_path = abs_file_path
                    
                record_id = str(uuid.uuid4())
                # Derive label from the immediate parent directory name.
                # If the image is directly inside data_dir, label stays None.
                parent_dir = os.path.basename(root)
                label = parent_dir if os.path.abspath(root) != data_dir else None
                records.append(ImageRecord(id=record_id, path=file_path, label=label))
    return records

def load_from_manifest(csv_path: str) -> List[ImageRecord]:
    """
    Loads image records from a CSV manifest.
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

def split_dataset(records: List[ImageRecord], test_ratio: float = 0.1) -> Tuple[List[ImageRecord], List[ImageRecord]]:
    """
    Splits the dataset into train and test sets based on the ratio.
    Groups by label (video ID) to prevent data leakage (frames from the same video stay together).
    """
    label_to_records = {}
    for record in records:
        label = record.label or "unknown"
        if label not in label_to_records:
            label_to_records[label] = []
        label_to_records[label].append(record)
        
    labels = list(label_to_records.keys())
    random.seed(42) # Fixed seed for reproducibility
    random.shuffle(labels)
    
    num_test = max(1, int(len(labels) * test_ratio)) # Ensure at least 1 test label if possible
    test_labels = set(labels[:num_test])
    
    train_records = []
    test_records = []
    
    for label, group in label_to_records.items():
        if label in test_labels:
            for record in group:
                record.split = "test"
                test_records.append(record)
        else:
            for record in group:
                record.split = "train"
                train_records.append(record)
                
    random.shuffle(train_records)
    random.shuffle(test_records)
    
    return train_records, test_records
