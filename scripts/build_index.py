"""
scripts/build_index.py — CLI script to index images.
Owner: Tech Lead (TL-1, Sprint 2)
"""
import sys
import argparse
from pathlib import Path

# Add root folder to sys.path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config import get_config  # noqa: E402
from src.pipeline.indexer import Indexer  # noqa: E402
from src.utils.logger import get_logger  # noqa: E402

logger = get_logger(__name__)


def parse_args():
    """Parse CLI arguments."""
    config = get_config()
    parser = argparse.ArgumentParser(description="Build FAISS Index for Image Retrieval")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="data/index", help="Output dir for index assets")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    parser.add_argument("--model-name", type=str, default=config.MODEL_NAME, help="CLIP model name")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Computing device (cpu, cuda)")
    parser.add_argument("--index-type", type=str, default="flat", choices=["flat", "ivf"], help="FAISS index type")
    parser.add_argument("--force", action="store_true", help="Force rebuild index (bypass cache)")
    return parser.parse_args()


def main():
    """Orchestrate index building from CLI args."""
    args = parse_args()
    logger.info("Starting offline index build pipeline...")
    logger.info("Data directory: %s", args.data_dir)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Batch size: %d", args.batch_size)
    logger.info("Model name: %s", args.model_name)
    logger.info("Device: %s", args.device)
    logger.info("Index type: %s", args.index_type)
    logger.info("Force rebuild: %s", args.force)

    try:
        indexer = Indexer(
            data_dir=args.data_dir,
            model_name=args.model_name,
            device=args.device,
            index_type=args.index_type,
        )
        indexer.run(
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            force=args.force,
        )
        logger.info("Index building completed successfully.")
    except Exception as e:
        logger.exception("Index building failed with error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
