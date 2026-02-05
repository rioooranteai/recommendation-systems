import logging
from pathlib import Path
import pandas as pd
from services.pinecone_service import PineconeService
from services.embedding_service import EmbeddingService
from tqdm import tqdm
from PIL import Image
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def build_index(
        data_path: str = "data/fashion-mini/data.csv",
        image_dir: str = "data/fashion-mini/data",
        batch_size: int = 100,
        max_items: int = None
):
    df = pd.read_csv(data_path)

    embedding_service = EmbeddingService()
    pinecone_service = PineconeService()

    vectors_batch = []
    success_count = []
    error_count = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc='Building Index'):
        try:
            image_path = Path(image_dir) / row['image']

            if not image_path.exists():
                error_count += 1
                continue

            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Failed to load image {row['filename']}: {e}")
                error_count += 1
                continue

            embedding = embedding_service.encode_images(image)

            vector_id = str(idx)

            metadata = {
                'title': str(row.get('display name', '')).strip()[:500],  # Limit length
                'description': str(row.get('description', '')).strip()[:1000],
                'category': str(row.get('category', 'unknown')).strip(),
                'filename': str(row['image']),
            }

            vectors_batch.append((
                vector_id,
                embedding.tolist(),
                metadata
            ))


            if len(vectors_batch) >= batch_size:
                try:
                    print(vectors_batch)
                    pinecone_service.upsert(vectors_batch)
                    success_count =+ len(vectors_batch)
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
    build_index("../data/fashion-mini/data.csv",
                "../data/fashion-mini/data",
                100,
                10)

        





