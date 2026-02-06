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
    parts = []

    if pd.notna(row.get('display name')):
        parts.append(str(row['display name']).strip())

    if pd.notna(row.get('category')):
        parts.append(f'Category: {row["category"]}')

    if pd.notna(row.get('description')):
        desc = str(row['description']).strip()[:300]
        parts.append(desc)

    return ". ".join(parts)


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

    expected_dim = embedding_service.get_embedding_dim()

    vectors_batch = []
    success_count = 0
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

            # Encode image
            img_emb = embedding_service.encode_images(image)
            img_values = to_1d_list(img_emb)

            # validate
            validate_vector(img_values, expected_dim)

            # Product ID (dari filename tanpa extension)
            product_id = str(row['image'].split(".")[0])

            # Metadata
            metadata = {
                'product_id': product_id,
                'kind': 'img',  # Mark as image vector
                'category': str(row.get('category', 'unknown')).strip(),
                'filename': str(row['image']),
            }

            vectors_batch.append((
                f"{product_id}#img",
                img_values,
                metadata
            ))

            if include_text:

                text_doc = build_text_doc(row)

                if not text_doc.strip():
                    logger.warning(f"Empty text for {product_id}, skipping text vector")

                else:
                    txt_emb = embedding_service.encode_text(text_doc)
                    txt_values = to_1d_list(txt_emb)

                    validate_vector(txt_values, expected_dim)

                    txt_metadata = {
                        'product_id': product_id,
                        'kind': 'txt',  # Mark as text vector
                        'category': str(row.get('category', 'unknown')).strip(),
                    }

                    vectors_batch.append((
                        f"{product_id}#txt",  # ID with suffix
                        txt_values,
                        txt_metadata
                    ))

            # Batch Upload
            if len(vectors_batch) >= batch_size:
                try:
                    pinecone_service.upsert(vectors_batch)
                    success_count += len(vectors_batch)
                    vectors_batch = []
                except Exception as e:
                    logger.error(f"Batch upload failed: {e}")
                    error_count += len(vectors_batch)
                    vectors_batch = []

        except Exception as e:
            logger.error(f"Failed to process row {idx} (id={row.get('id', 'unknown')}) : {e}")
            error_count += 1
            continue

    if vectors_batch:
        try:
            pinecone_service.upsert(vectors_batch)
            success_count += len(vectors_batch)
        except Exception as e:
            logger.error(f"Final batch upload failed: {e}")
            error_count += len(vectors_batch)

    if success_count > 0:
        logger.info("Index build completed!")
    else:
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
