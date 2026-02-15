import logging
import sys
import time

from config import Config
from pinecone.grpc import PineconeGRPC as Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def list_index_names(pc):
    return [idx["name"] for idx in pc.list_indexes()]


def wait_until_deleted(pc, index_name: str, timeout_s: int = 60):
    start = time.time()
    while time.time() - start < timeout_s:
        if index_name not in list_index_names(pc):
            return True
        time.sleep(2)
    return False


def wait_until_ready(pc, index_name: str, timeout_s: int = 120):
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            idx = pc.Index(index_name)
            idx.describe_index_stats()
            return True
        except Exception:
            time.sleep(2)
    return False


def create_single_index(pc, index_name: str, dimension: int, force_recreate: bool = False):
    """Helper function to create a single index"""
    existing = list_index_names(pc)

    if index_name in existing:
        if not force_recreate:
            logger.info(f"Index '{index_name}' already exists. Skipping creation.")
            return

        logger.warning(f"Deleting index '{index_name}'...")
        pc.delete_index(index_name)

        if not wait_until_deleted(pc, index_name):
            logger.error(f"Timeout while waiting for index '{index_name}' deletion.")
            sys.exit(1)

    logger.info(
        f"Creating index '{index_name}' (dim={dimension}, metric=cosine)..."
    )

    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
    except Exception as e:
        logger.error(f"Failed to create index '{index_name}': {e}")
        sys.exit(1)

    if not wait_until_ready(pc, index_name):
        logger.error(f"Timeout while waiting for index '{index_name}' to be ready.")
        sys.exit(1)

    logger.info(f"Index '{index_name}' is ready.")


def create_indexes(force_recreate: bool = False):
    """Create both image and text indexes for two-stage retrieval"""
    if not Config.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is missing")
        sys.exit(1)

    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    # Index 1: Image embeddings (SigLIP - 768 dim)
    image_index_name = Config.PINECONE_IMAGE_INDEX_NAME
    image_dimension = Config.IMAGE_EMBEDDING_DIM  # 768

    # Index 2: Text embeddings (BGE-M3 - 1024 dim)
    text_index_name = Config.PINECONE_TEXT_INDEX_NAME
    text_dimension = Config.TEXT_EMBEDDING_DIM

    logger.info("=" * 60)
    logger.info("Creating Two-Stage Retrieval Indexes")
    logger.info("=" * 60)

    # Create Image Index
    logger.info("\n[1/2] Creating Image Index...")
    create_single_index(pc, image_index_name, image_dimension, force_recreate)

    # Create Text Index
    logger.info("\n[2/2] Creating Text Index...")
    create_single_index(pc, text_index_name, text_dimension, force_recreate)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Both indexes created successfully!")
    logger.info(f"  - Image Index: {image_index_name} ({image_dimension}-dim)")
    logger.info(f"  - Text Index: {text_index_name} ({text_dimension}-dim)")
    logger.info("=" * 60)


if __name__ == "__main__":
    create_indexes(force_recreate=True)