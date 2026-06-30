import os
import sys
import json
import uuid

# Add project root to sys.path to import src.data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.datatypes import ImageRecord
from src.data.metadata_store import save_metadata

def main():
    keyframes_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "Keyframes_L30"))
    media_info_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "media_info_L30"))
    output_metadata_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "metadata.json"))

    if not os.path.exists(keyframes_dir):
        print(f"Error: {keyframes_dir} does not exist.")
        return

    records = []
    
    for video_id in os.listdir(keyframes_dir):
        video_dir = os.path.join(keyframes_dir, video_id)
        if not os.path.isdir(video_dir):
            continue
            
        # Try to read the metadata for the video to get caption
        caption = video_id
        media_info_path = os.path.join(media_info_dir, f"{video_id}.json")
        if os.path.exists(media_info_path):
            try:
                with open(media_info_path, 'r', encoding='utf-8') as f:
                    media_info = json.load(f)
                    caption = media_info.get("title", video_id)
            except Exception as e:
                print(f"Warning: Could not read {media_info_path}: {e}")

        # Iterate through all frames in the video directory
        for frame_file in sorted(os.listdir(video_dir)):
            if frame_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Store absolute path so it works everywhere, since data is outside the project root
                frame_path = os.path.join(video_dir, frame_file)
                record = ImageRecord(
                    id=str(uuid.uuid4()),
                    path=frame_path,
                    label=video_id,
                    caption=caption,
                    split=None
                )
                records.append(record)

    from src.data.dataset_loader import split_dataset
    print(f"Found {len(records)} images.")
    
    # Split the dataset into train/test
    train_recs, test_recs = split_dataset(records, test_ratio=0.1)
    records = train_recs + test_recs
    print(f"Split dataset: {len(train_recs)} train, {len(test_recs)} test.")
    
    os.makedirs(os.path.dirname(output_metadata_path), exist_ok=True)
    save_metadata(records, output_metadata_path)
    print(f"Successfully saved metadata to {output_metadata_path}")

if __name__ == "__main__":
    main()
