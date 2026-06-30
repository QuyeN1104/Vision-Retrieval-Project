import os
import csv
import uuid
import random
from typing import List, Tuple
from .datatypes import ImageRecord

def load_dataset(data_dir: str) -> List[ImageRecord]:
    """
    Scans an image directory and returns a list of ImageRecords.
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
