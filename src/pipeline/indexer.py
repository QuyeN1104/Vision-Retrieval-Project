"""
src/pipeline/indexer.py — Orchestrator for building/generating index.
Owner: Tech Lead (TL-2, Sprint 2)
"""
from pathlib import Path
from src.data.dataset_loader import load_dataset
from src.data.metadata_store import save_metadata
from src.model.clip_encoder import CLIPEncoder
from src.search.faiss_index import FaissIndex
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Indexer:
    """
    Orchestrates the entire offline pipeline: Load dataset -> Batch encode -> Build Index -> Save.
    """

    def __init__(self, data_dir: Path | str, model_name: str, device: str = "cpu", index_type: str = "flat"):
        """
        Setup components for indexing.

        Args:
            data_dir: Source folder containing image database.
            model_name: Name of CLIP model to use.
            device: Computing device ('cpu', 'cuda').
            index_type: Type of index ('flat', 'ivf').
        """
        self.data_dir = Path(data_dir)
        self.model_name = model_name
        self.device = device
        self.index_type = index_type
        
        logger.info("Initializing CLIPEncoder with model %s on device %s...", model_name, device)
        self.encoder = CLIPEncoder(model_name=model_name, device=device)

    def run(self, output_dir: Path | str, batch_size: int = 32, force: bool = False) -> None:
        """
        Execute indexing flow end-to-end and write files to the output directory.

        Args:
            output_dir: Target directory to save FAISS index, metadata, and cache.
            batch_size: Encoding batch size.
            force: Force re-encoding and ignore cache.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Scanning data directory: %s", self.data_dir)
        records = load_dataset(str(self.data_dir))
        
        if not records:
            raise ValueError(f"No valid images (.jpg, .jpeg, .png) found in data directory: {self.data_dir}")
            
        logger.info("Found %d images in data directory.", len(records))

        # Check if cache exists
        cache_base = output_dir / "embeddings"
        metadata_file = output_dir / "metadata.json"
        index_file = output_dir / "faiss.index"
        
        embeddings = None
        
        # We can reuse cached embeddings if:
        # 1. force is False.
        # 2. Cache files, metadata, and index all exist.
        # 3. The set and order of image paths match the cached metadata.
        if (not force and 
            cache_base.with_suffix(".npy").exists() and 
            cache_base.with_suffix(".json").exists() and 
            metadata_file.exists() and 
            index_file.exists()):
            try:
                from src.data.metadata_store import load_metadata
                from src.model.embeddings_cache import EmbeddingsCache
                
                cached_records = load_metadata(str(metadata_file))
                # Compare scanned records paths with cached records paths
                scanned_paths = [r.path for r in records]
                cached_paths = [r.path for r in cached_records]
                
                if scanned_paths == cached_paths:
                    logger.info("Dataset matches cached metadata. Attempting to load cached embeddings...")
                    embeddings, ids = EmbeddingsCache.load(cache_base)
                    
                    # Verify cache size matches records count
                    if len(embeddings) == len(records):
                        # Re-use the cached records (keeping the same UUIDs)
                        records = cached_records
                        logger.info("Loaded %d cached embeddings successfully. Skipping encoding.", len(embeddings))
                    else:
                        logger.warning("Cache size mismatch. Re-encoding...")
                        embeddings = None
                else:
                    logger.info("Dataset changes detected (or order difference). Re-encoding...")
            except Exception as e:
                logger.warning("Failed to load embeddings from cache: %s. Re-encoding...", e)
                embeddings = None

        if embeddings is None:
            logger.info("Encoding dataset with batch size %d...", batch_size)
            from src.pipeline.batch_encode import encode_dataset
            embeddings = encode_dataset(records, self.encoder, batch_size=batch_size)
            
            # Save to cache
            try:
                from src.model.embeddings_cache import EmbeddingsCache
                logger.info("Saving embeddings and IDs to cache...")
                ids = [r.id for r in records]
                EmbeddingsCache.save(embeddings, ids, cache_base)
            except Exception as e:
                logger.warning("Failed to save embeddings to cache: %s", e)

        # Get embedding dimension dynamically
        dim = getattr(self.encoder.model.config, "projection_dim", 512)
        logger.info("Determined embedding dimension: %d", dim)

        logger.info("Building FAISS index (type: %s)...", self.index_type)
        index = FaissIndex(dim=dim, index_type=self.index_type)
        index.build(embeddings)
        
        logger.info("Saving FAISS index to %s...", index_file)
        index.save(index_file)

        logger.info("Saving metadata to %s...", metadata_file)
        save_metadata(records, str(metadata_file))

        logger.info("Indexing pipeline completed successfully.")
