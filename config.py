"""
config.py — Centralized project configuration interface.
Owner: Tech Lead (TL-1, Sprint 1)
"""

import os
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict

from functools import lru_cache


class Config(BaseSettings):
    """
    Centralized project configuration class.
    To be populated with paths, model configurations, and search parameters.
    """
    DATA_DIR: str = "data"
    INDEX_PATH: str = "data/index/faiss.index"
    MODEL_NAME: str = "openai/clip-vit-base-patch32"
    TOP_K: int = 5
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

@lru_cache(maxsize=1)
def get_config() -> Config:
    """
    Returns the global configuration object.
    """
    return Config()