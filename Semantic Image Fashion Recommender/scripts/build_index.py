import logging
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_vector(values, expected_dim: int):
    if not isinstance(values, list):
        raise TypeError(f"Vector must be list, got {type(values)}")
    if len(values) != expected_dim:
        raise ValueError(f"Vector dimension {len(values)} != expected {expected_dim}")
    if not all(isinstance(x, (int, float)) for x in values):
        raise TypeError("Vector must contain only numbers")


def to_1d_list(arr):
    if hasattr(arr, 'detach'):
        arr = arr.detach().cpu().numpy()
    if hasattr(arr, 'ndim') and arr.ndim == 2:
        arr = arr.flatten()
    return [float(x) for x in arr.tolist()]


def build_text_doc(row) -> str:
    """Build natural language document for BGE-M3"""
    parts = []

    product_name = str(row.get('display name', '')).strip()
    category = str(row.get('category', '')).strip()
    description = str(row.get('description', '')).strip()

    if product_name:
        parts.append(f"This is a {product_name}")

    if category and category.lower() != 'unknown':
        parts.append(f"in the {category} category")

    if description:
        desc_clean = description[:500].strip()
        parts.append(f"It is {desc_clean}")

    result = ". ".join(parts)
    if result and not result.endswith('.'):
        result += '.'

    return result


def build_index(
        data_path: str = "data/fashion-mini/data.csv",
        image_dir: str = "data/fashion-mini/data",
        batch_size: int = 100,
        max_items: int = None,
        include_text: bool = True
):
    df = pd.read_csv(data_path)

    if max_items:
        df = df.head(max_items)

    embedding_service = EmbeddingService()
    pinecone_service = PineconeService()

    text_dim = embedding_service.get_embedding_dim()
    image_dim = embedding_service.get_image_embedding_dim()

    logger.info(f"Text dimension: {text_dim}, Image dimension: {image_dim}")
    logger.info(f"Processing {len(df)} products")

    image_vectors_batch = []
    text_vectors_batch = []

    image_success_count = 0
    text_success_count = 0
    error_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Building Index'):
        try:
            image_path = Path(image_dir) / row['image']

            if not image_path.exists():
                error_count += 1
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {row['image']}: {e}")
                error_count += 1
                continue

            # Encode IMAGE
            img_emb = embedding_service.encode_images(image)
            img_values = to_1d_list(img_emb)
            validate_vector(img_values, image_dim)

            product_id = str(row['image'].split(".")[0])

            img_metadata = {
                'product_id': product_id,
                'kind': 'img',
                'category': str(row.get('category', 'unknown')).strip(),
                'filename': str(row['image'])
            }

            image_vectors_batch.append((
                f"{product_id}#img",
                img_values,
                img_metadata
            ))

            # Encode TEXT
            if include_text:
                text_doc = build_text_doc(row)

                if not text_doc.strip():
                    logger.warning(f"Empty text for {product_id}, skipping text vector")
                else:
                    txt_emb = embedding_service.encode_text(text_doc)
                    txt_values = to_1d_list(txt_emb)
                    validate_vector(txt_values, text_dim)

                    txt_metadata = {
                        'product_id': product_id,
                        'kind': 'txt',
                        'category': str(row.get('category', 'unknown')).strip()
                    }

                    text_vectors_batch.append((
                        f"{product_id}#txt",
                        txt_values,
                        txt_metadata
                    ))

            # Batch Upload
            if len(image_vectors_batch) >= batch_size:
                try:
                    pinecone_service.upsert_images(image_vectors_batch)
                    image_success_count += len(image_vectors_batch)
                    image_vectors_batch = []
                except Exception as e:
                    logger.error(f"Image batch upload failed: {e}")
                    error_count += len(image_vectors_batch)
                    image_vectors_batch = []

            if len(text_vectors_batch) >= batch_size:
                try:
                    pinecone_service.upsert_text(text_vectors_batch)
                    text_success_count += len(text_vectors_batch)
                    text_vectors_batch = []
                except Exception as e:
                    logger.error(f"Text batch upload failed: {e}")
                    error_count += len(text_vectors_batch)
                    text_vectors_batch = []

        except Exception as e:
            logger.error(f"Failed to process row {idx}: {e}")
            error_count += 1
            continue

    # Upload remaining
    if image_vectors_batch:
        try:
            pinecone_service.upsert_images(image_vectors_batch)
            image_success_count += len(image_vectors_batch)
        except Exception as e:
            logger.error(f"Final image batch upload failed: {e}")
            error_count += len(image_vectors_batch)

    if text_vectors_batch:
        try:
            pinecone_service.upsert_text(text_vectors_batch)
            text_success_count += len(text_vectors_batch)
        except Exception as e:
            logger.error(f"Final text batch upload failed: {e}")
            error_count += len(text_vectors_batch)

    # Summary
    logger.info(f"✓ Image vectors: {image_success_count}")
    logger.info(f"✓ Text vectors: {text_success_count}")
    logger.info(f"✗ Errors: {error_count}")

    if image_success_count == 0 and text_success_count == 0:
        logger.error("No vectors were successfully indexed!")
        sys.exit(1)


if __name__ == "__main__":
    build_index(
        data_path="../data/fashion-mini/data.csv",
        image_dir="../data/fashion-mini/data",
        batch_size=100,
        max_items=None,
        include_text=True
    )
