import json
from typing import List
from .datatypes import ImageRecord

def save_metadata(records: List[ImageRecord], path: str) -> None:
    """
    Saves a list of ImageRecords to a JSON file.
    """
    data = [
        {
            "id": r.id,
            "path": r.path,
            "label": r.label,
            "caption": r.caption,
            "split": r.split
        }
        for r in records
    ]
    with open(path, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_metadata(path: str) -> List[ImageRecord]:
    """
    Loads a list of ImageRecords from a JSON file.
    """
    records = []
    with open(path, mode="r", encoding="utf-8") as f:
        data = json.load(f)
        for item in data:
            records.append(
                ImageRecord(
                    id=item.get("id", ""),
                    path=item.get("path", ""),
                    label=item.get("label"),
                    caption=item.get("caption"),
                    split=item.get("split")
                )
            )
    return records
