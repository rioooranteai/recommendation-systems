import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image
from config import Config
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def validate_vector(values: List[float], expected_dim: int) -> None:
    """Assert that a vector is a correctly shaped list of numbers.

    Args:
        values: The vector to validate.
        expected_dim: Expected number of dimensions.

    Raises:
        TypeError: If ``values`` is not a list or contains non-numeric elements.
        ValueError: If the vector length does not match ``expected_dim``.
    """
    if not isinstance(values, list):
        raise TypeError(f"Vector must be list, got {type(values)}")
    if len(values) != expected_dim:
        raise ValueError(
            f"Vector dimension {len(values)} != expected {expected_dim}"
        )
    if not all(isinstance(x, (int, float)) for x in values):
        raise TypeError("Vector must contain only numbers")


def to_1d_list(arr) -> List[float]:
    """Convert any array-like object to a flat Python list of floats.

    Handles PyTorch tensors (detach + cpu), 2-D numpy arrays (flatten),
    and any other iterable with a ``.tolist()`` method.

    Args:
        arr: Array-like object — torch.Tensor, np.ndarray, or similar.

    Returns:
        Flat list of Python floats.
    """
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    if hasattr(arr, "ndim") and arr.ndim == 2:
        arr = arr.flatten()
    return [float(x) for x in arr.tolist()]


def build_text_doc(row: pd.Series) -> str:
    """Build a natural language document for embedding and reranking context.

    Combines product name, category, and description into a single
    human-readable sentence. Fields that are empty or ``"unknown"`` are
    omitted. Description is truncated to ``_DESCRIPTION_EMBED_LIMIT`` chars.

    Args:
        row: A single row from the product catalogue ``DataFrame``.

    Returns:
        A punctuated sentence string, or an empty string if all fields
        are blank.
    """
    parts = []

    product_name = str(row.get("display name", "")).strip()
    category = str(row.get("category", "")).strip()
    description = str(row.get("description", "")).strip()

    if product_name:
        parts.append(f"This is a {product_name}")

    if category and category.lower() != "unknown":
        parts.append(f"in the {category} category")

    if description:
        parts.append(f"It is {description[:Config._DESCRIPTION_EMBED_LIMIT].strip()}")

    result = ". ".join(parts)
    if result and not result.endswith("."):
        result += "."

    return result


def _flush_image_batch(
        pinecone_service: PineconeService,
        batch: List[Tuple],
        success_count: int,
        error_count: int,
) -> Tuple[int, int]:
    """Upsert a batch of image vectors and update counters.

    Args:
        pinecone_service: Pinecone service instance for upsert operations.
        batch: List of (id, values, metadata) tuples to upsert.
        success_count: Running total of successfully upserted image vectors.
        error_count: Running total of errors encountered.

    Returns:
        Updated ``(success_count, error_count)`` tuple.
    """
    try:
        pinecone_service.upsert_images(batch)
        success_count += len(batch)
    except Exception as e:
        logger.error("Image batch upload failed: %s", e)
        error_count += len(batch)
    return success_count, error_count


def _flush_text_batch(
        pinecone_service: PineconeService,
        batch: List[Tuple],
        success_count: int,
        error_count: int,
) -> Tuple[int, int]:
    """Upsert a batch of text vectors and update counters.

    Args:
        pinecone_service: Pinecone service instance for upsert operations.
        batch: List of (id, values, metadata) tuples to upsert.
        success_count: Running total of successfully upserted text vectors.
        error_count: Running total of errors encountered.

    Returns:
        Updated ``(success_count, error_count)`` tuple.
    """
    try:
        pinecone_service.upsert_text(batch)
        success_count += len(batch)
    except Exception as e:
        logger.error("Text batch upload failed: %s", e)
        error_count += len(batch)
    return success_count, error_count


def build_index(
        data_path: str = "data/fashion-mini/data.csv",
        image_dir: str = "data/fashion-mini/data",
        batch_size: int = 100,
        max_items: Optional[int] = None,
        include_text: bool = True,
) -> None:
    """Encode and index all products from a CSV catalogue into Pinecone.

    For each product row:
        1. Loads and encodes the product image (SigLIP, 768-dim).
        2. Builds and encodes a text document (BGE-M3, 1024-dim).
        3. Upserts both vectors into their respective Pinecone indexes
           in batches of ``batch_size``.

    Calls ``sys.exit(1)`` if no vectors were successfully indexed.

    Args:
        data_path: Path to the product catalogue CSV file.
        image_dir: Directory containing product images referenced in the CSV.
        batch_size: Number of vectors per upsert batch.
        max_items: If set, limits processing to the first N rows.
        include_text: If ``False``, skips text encoding and indexing.
    """
    df = pd.read_csv(data_path)

    if max_items:
        df = df.head(max_items)

    embedding_service = EmbeddingService()
    pinecone_service = PineconeService()

    logger.warning("Attempting to delete old vectors to refresh metadata...")
    try:
        pinecone_service.delete_all()
        logger.info("Old vectors deleted.")
    except Exception as e:
        logger.info(
            "Skipping deletion (database is likely empty). Reason: %s", e
        )

    text_dim = embedding_service.get_embedding_dim()
    image_dim = embedding_service.get_image_embedding_dim()

    logger.info("Text dim: %d, Image dim: %d", text_dim, image_dim)
    logger.info("Processing %d products", len(df))

    image_vectors_batch: List[Tuple] = []
    text_vectors_batch: List[Tuple] = []

    image_success_count = 0
    text_success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building Index"):
        try:
            image_path = Path(image_dir) / row["image"]

            if not image_path.exists():
                error_count += 1
                continue

            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                logger.warning("Failed to load image %s: %s", row["image"], e)
                error_count += 1
                continue

            product_id = str(row["image"].split(".")[0])
            display_name = str(row.get("display name", "")).strip()
            category_name = str(row.get("category", "unknown")).strip()
            description = str(row.get("description", "")).strip()
            filename = str(row["image"])

            # 1. Process image vector
            img_values = to_1d_list(embedding_service.encode_images(image))
            validate_vector(img_values, image_dim)

            image_vectors_batch.append((
                f"{product_id}#img",
                img_values,
                {
                    "product_id": product_id,
                    "kind": "img",
                    "category": category_name,
                    "filename": filename,
                    "name": display_name,
                },
            ))

            # 2. Process text vector
            if include_text:
                text_doc = build_text_doc(row)

                if not text_doc.strip():
                    logger.warning("Empty text for %s, skipping", product_id)
                else:
                    txt_values = to_1d_list(embedding_service.encode_text(text_doc))
                    validate_vector(txt_values, text_dim)

                    text_vectors_batch.append((
                        f"{product_id}#txt",
                        txt_values,
                        {
                            "product_id": product_id,
                            "kind": "txt",
                            "category": category_name,
                            "name": display_name,
                            "image": filename,
                            "description": description[:Config._DESCRIPTION_METADATA_LIMIT],
                            "text": text_doc,
                        },
                    ))

            # 3. Flush batches when threshold is reached
            if len(image_vectors_batch) >= batch_size:
                image_success_count, error_count = _flush_image_batch(
                    pinecone_service, image_vectors_batch,
                    image_success_count, error_count,
                )
                image_vectors_batch = []

            if len(text_vectors_batch) >= batch_size:
                text_success_count, error_count = _flush_text_batch(
                    pinecone_service, text_vectors_batch,
                    text_success_count, error_count,
                )
                text_vectors_batch = []

        except Exception as e:
            logger.error("Failed to process row %d: %s", idx, e)
            error_count += 1
            continue

    # Flush remaining vectors after the loop
    if image_vectors_batch:
        image_success_count, error_count = _flush_image_batch(
            pinecone_service, image_vectors_batch,
            image_success_count, error_count,
        )

    if text_vectors_batch:
        text_success_count, error_count = _flush_text_batch(
            pinecone_service, text_vectors_batch,
            text_success_count, error_count,
        )

    logger.info(
        "Summary: Images=%d, Text=%d, Errors=%d",
        image_success_count, text_success_count, error_count,
    )

    if image_success_count == 0 and text_success_count == 0:
        logger.error("No vectors were successfully indexed!")
        sys.exit(1)


if __name__ == "__main__":
    build_index(
        data_path="../data/fashion-mini/data.csv",
        image_dir="../data/fashion-mini/data",
        batch_size=100,
        max_items=None,
        include_text=True,
    )
