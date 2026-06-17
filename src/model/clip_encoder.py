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

        # Step 1: Tokenize — convert the raw string into token IDs the model understands
        # e.g. "a cat" → {input_ids: [[49406, 320, 2368, 49407]], attention_mask: [[1,1,1,1]]}
        inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 2: Forward pass — run tokens through the text encoder
        # torch.no_grad() disables gradient computation → saves memory and speeds up inference
        with torch.no_grad():
            text_features = self.model.get_text_features(**inputs)

        # Step 3: L2 normalize — make the vector unit length so dot product == cosine similarity
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Step 4: Convert to numpy — move off GPU if needed, cast to float32 (FAISS requirement)
        return text_features.cpu().numpy().astype(np.float32).squeeze()

    def encode_image(self, image: Image.Image) -> np.ndarray:
        
        # Step 1: Preprocess — resize to 224x224, normalize pixel values (ImageNet mean/std)
        # The processor handles RGB conversion automatically (grayscale, RGBA → RGB)
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Step 2: Forward pass — run pixel tensor through the vision encoder (ViT)
        with torch.no_grad():
            image_features = self.model.get_image_features(**inputs)

        # Step 3: L2 normalize — same reason as encode_text, ensures cosine similarity via dot product
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Step 4: Convert to numpy float32, squeeze batch dim (1, 512) → (512,)
        return image_features.cpu().numpy().astype(np.float32).squeeze()

    def encode_images_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of PIL Images into normalized L2 embeddings.

        Args:
            images: List of preprocessed PIL Images.
            batch_size: Batch size for model inference.

        Returns:
            A 2D numpy array of shape (num_images, embedding_dim) representing the normalized embeddings.
        """
        all_embeddings = []

        # Process images in chunks of `batch_size` to avoid OOM
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]

            # Preprocess the entire batch at once — processor handles a list of images
            inputs = self.processor(images=batch, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)

            # Normalize each vector in the batch independently
            features = features / features.norm(dim=-1, keepdim=True)

            # Move to CPU immediately to free GPU memory before next batch
            all_embeddings.append(features.cpu().numpy().astype(np.float32))

        # Stack all chunks: list of (batch_i, 512) → single (N, 512) matrix
        return np.vstack(all_embeddings)

    def encode_image_from_path(self, path: str) -> np.ndarray:
        """
        Helper to load, preprocess and encode an image from its path.

        Args:
            path: Path to the image file.

        Returns:
            A 1D numpy array of shape (embedding_dim,) representing the normalized embedding.
        """
        raise NotImplementedError("To be implemented by Model Engineer (ME-3) in Sprint 2")
