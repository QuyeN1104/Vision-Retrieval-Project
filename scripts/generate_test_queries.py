import os
import sys
import json
import random

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.metadata_store import load_metadata

def main():
    metadata_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json"))
    output_queries_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "test_queries.json"))

    if not os.path.exists(metadata_path):
        print(f"Error: {metadata_path} not found. Please run prepare_dataset.py first.")
        return

    records = load_metadata(metadata_path)
    
    # Filter test records and group by label
    test_label_to_records = {}
    for r in records:
        if r.split == "test":
            if r.label not in test_label_to_records:
                test_label_to_records[r.label] = []
            test_label_to_records[r.label].append(r)

    if not test_label_to_records:
        print("Warning: No test records found in metadata. Did you split the dataset?")
        # Fallback to grouping everything if split hasn't been done
        for r in records:
            if r.label not in test_label_to_records:
                test_label_to_records[r.label] = []
            test_label_to_records[r.label].append(r)

    labels = list(test_label_to_records.keys())
    random.seed(42)
    random.shuffle(labels)
    
    # Select up to 50 videos
    selected_labels = labels[:50]
    
    queries = []
    for label in selected_labels:
        group = test_label_to_records[label]
        # Use caption from the first record as query
        caption = group[0].caption or f"Video {label}"
        
        # Ground truth: all frame IDs of this video
        relevant_ids = [r.id for r in group]
        
        queries.append({
            "query": caption,
            "relevant_ids": relevant_ids
        })

    os.makedirs(os.path.dirname(output_queries_path), exist_ok=True)
    with open(output_queries_path, "w", encoding="utf-8") as f:
        json.dump(queries, f, indent=4, ensure_ascii=False)

    print(f"Successfully generated {len(queries)} test queries in {output_queries_path}")

if __name__ == "__main__":
    main()
