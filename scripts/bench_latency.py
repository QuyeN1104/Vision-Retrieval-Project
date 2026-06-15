"""
scripts/bench_latency.py — Benchmark latency of the retrieval system.
Owner: Tech Lead (TL-2, Sprint 3)
"""
import argparse


def parse_args():
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Benchmark Retrieval Latency")
    parser.add_argument("--index-path", type=str, default="data/index/faiss.index", help="Path to FAISS index")
    parser.add_argument("--num-runs", type=int, default=200, help="Number of test queries to run")
    return parser.parse_args()


def main():
    """Run latency benchmark under load."""
    args = parse_args()
    print(f"Running latency benchmark with {args.num_runs} runs.")
    raise NotImplementedError("To be implemented by Tech Lead (TL-2) in Sprint 3")


if __name__ == "__main__":
    main()
