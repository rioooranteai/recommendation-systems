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


def create_index(force_recreate: bool = False):
    if not Config.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is missing")
        sys.exit(1)

    if not Config.EMBEDDING_DIM:
        logger.error("EMBEDDING_DIM is missing")
        sys.exit(1)

    pc = Pinecone(api_key=Config.PINECONE_API_KEY)
    index_name = Config.PINECONE_INDEX_NAME

    existing = list_index_names(pc)

    if index_name in existing:
        if not force_recreate:
            logger.info(f"Index '{index_name}' already exists. Skipping creation.")
            return

        logger.warning(f"Deleting index '{index_name}'...")
        pc.delete_index(index_name)

        if not wait_until_deleted(pc, index_name):
            logger.error("Timeout while waiting for index deletion.")
            sys.exit(1)

    logger.info(
        f"Creating index '{index_name}' (dim={int(Config.EMBEDDING_DIM)}, metric=cosine)..."
    )

    try:
        pc.create_index(
            name=index_name,
            dimension=int(Config.EMBEDDING_DIM),
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
    except Exception as e:
        logger.error(f"Failed to create index: {e}")
        sys.exit(1)

    if not wait_until_ready(pc, index_name):
        logger.error("Timeout while waiting for index to be ready.")
        sys.exit(1)

    logger.info("Index is ready.")


if __name__ == "__main__":
    create_index(force_recreate=True)
