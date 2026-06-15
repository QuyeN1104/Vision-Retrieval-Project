"""
scripts/build_index.py — CLI script to index images.
Owner: Tech Lead (TL-1, Sprint 2)
"""
import argparse


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build FAISS Index for Image Retrieval")
    parser.add_argument("--data-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output-dir", type=str, default="data/index", help="Output dir for index assets")
    parser.add_argument("--batch-size", type=int, default=32, help="Encoding batch size")
    return parser.parse_args()


def main():
    """Orchestrate index building from CLI args."""
    args = parse_args()
    print(f"Starting index build pipeline for folder: {args.data_dir}")
    raise NotImplementedError("To be implemented by Tech Lead (TL-1) in Sprint 2")


if __name__ == "__main__":
    main()
