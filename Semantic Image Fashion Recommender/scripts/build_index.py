import logging
from pathlib import Path

import pandas as pd
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_index(
        data_path: str = "data/data.csv",
        image_dir: str = "data/images",
        batch_size: int = 100
):
    embedding_service = EmbeddingService()
    pinecone_service = PineconeService()

    df = pd.read_csv(data_path)

    vectors_batch = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building index"):
        try:
            image_path = Path(image_dir) / row['image_filename']

            if not image_path.exists():
                continue

            # Generate embedding
            embedding = embedding_service.encode_images(str(image_path))

            # Prepare vector
            vector_id = str(row['id'])
            metadata = {
                'title': row.get('title', ''),
                'category': row.get('category', ''),
                'price': float(row.get('price', 0)),
                'image_url': row.get('image_url', ''),
                'image_filename': row['image_filename']
            }

            vectors_batch.append((vector_id, embedding.tolist(), metadata))

            # Upload batch
            if len(vectors_batch) >= batch_size:
                pinecone_service.upsert(vectors_batch)
                vectors_batch = []

        except Exception as e:
            logger.error(f"Failed to process row {idx}: {e}")
            continue

    # Upload remaining
    if vectors_batch:
        pinecone_service.upsert(vectors_batch)


if __name__ == "__main__":
    build_index()
