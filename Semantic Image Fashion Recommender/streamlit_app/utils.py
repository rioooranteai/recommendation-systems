from PIL import Image
import io
from typing import Optional


def resize_image(image: Image.Image, max_size: tuple = (800, 800)) -> Image.Image:
    """
    Resize image while maintaining aspect ratio

    Args:
        image: PIL Image
        max_size: Maximum (width, height)

    Returns:
        Resized image
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def validate_image(uploaded_file) -> Optional[Image.Image]:
    """
    Validate and load uploaded image

    Args:
        uploaded_file: Streamlit UploadedFile

    Returns:
        PIL Image or None if invalid
    """
    try:
        image = Image.open(uploaded_file)
        return image.convert('RGB')
    except Exception as e:
        return None


def format_score(score: float) -> str:
    """
    Format similarity score for display

    Args:
        score: Similarity score (0-1)

    Returns:
        Formatted string (e.g., "85%")
    """
    return f"{score * 100:.1f}%"


def get_result_color(score: float) -> str:
    """
    Get color based on similarity score

    Args:
        score: Similarity score (0-1)

    Returns:
        Color name
    """
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "orange"
    else:
        return "red"