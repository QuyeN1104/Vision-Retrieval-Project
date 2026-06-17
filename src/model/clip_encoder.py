"""
src/model/clip_encoder.py — CLIP Encoder wrapper for text and images.
Owner: Model Engineer (ME-1, ME-2, ME-3, ME-4, Sprint 1)
"""
from typing import List, Union
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


class CLIPEncoder:
    """
    Wrapper around HuggingFace or OpenAI CLIP model to compute embeddings.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = None):
        """
        Initialize the CLIP model and processor.

        Args:
            model_name: HuggingFace model identifier.
            device: Device to run inference on ('cpu', 'cuda', 'mps'). If None, it auto-detects.
        """
        self.model_name = model_name
        
        # Auto-detect device if none provided
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # CLIPProcessor handles both tokenization (text) and image preprocessing
        self.processor = CLIPProcessor.from_pretrained(model_name)
        
        # Load the model and move it to the target device
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        
        # CRITICAL: Always put the model in evaluation mode for inference.
        # This disables layers like Dropout and BatchNorm, making outputs deterministic.
        self.model.eval()

    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode a single text query into a normalized L2 embedding.

        Args:
            text: Input text prompt.

        Returns:
            A 1D numpy array of shape (embedding_dim,) representing the normalized embedding.
        """
        raise NotImplementedError("To be implemented by Model Engineer (ME-2) in Sprint 1")

    def encode_image(self, image: Image.Image) -> np.ndarray:
        """
        Encode a single PIL Image into a normalized L2 embedding.

        Args:
            image: Preprocessed PIL Image.

        Returns:
            A 1D numpy array of shape (embedding_dim,) representing the normalized embedding.
        """
        raise NotImplementedError("To be implemented by Model Engineer (ME-3) in Sprint 1")

    def encode_images_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of PIL Images into normalized L2 embeddings.

        Args:
            images: List of preprocessed PIL Images.
            batch_size: Batch size for model inference.

        Returns:
            A 2D numpy array of shape (num_images, embedding_dim) representing the normalized embeddings.
        """
        raise NotImplementedError("To be implemented by Model Engineer (ME-4) in Sprint 1")

    def encode_image_from_path(self, path: str) -> np.ndarray:
        """
        Helper to load, preprocess and encode an image from its path.

        Args:
            path: Path to the image file.

        Returns:
            A 1D numpy array of shape (embedding_dim,) representing the normalized embedding.
        """
        raise NotImplementedError("To be implemented by Model Engineer (ME-3) in Sprint 2")
