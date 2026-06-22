from dataclasses import dataclass
from typing import Optional

@dataclass
class ImageRecord:
    id: str
    path: str
    label: Optional[str] = None
    caption: Optional[str] = None
    split: Optional[str] = None
