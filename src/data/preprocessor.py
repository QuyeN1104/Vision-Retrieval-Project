from PIL import Image

def preprocess_image(img_path: str) -> Image.Image:
    """
    Loads an image from path, validates its format, and converts it to RGB.
    """
    try:
        img = Image.open(img_path)
        img.verify()  # verify format
        # PIL's verify() closes the image, so we need to reopen it to use it
        img = Image.open(img_path)
        # Convert to RGB to ensure consistency (handles PNG with alpha, grayscale, etc.)
        img = img.convert("RGB")
        return img
    except Exception as e:
        raise ValueError(f"Failed to process image {img_path}: {e}")
