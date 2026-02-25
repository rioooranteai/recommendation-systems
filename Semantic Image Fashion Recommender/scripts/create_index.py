import logging
import sys
import time

from config import Config
from pinecone.grpc import PineconeGRPC as Pinecone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 60


def list_index_names(pc: Pinecone) -> list[str]:
    """Return the names of all existing Pinecone indexes.

    Args:
        pc: An authenticated ``PineconeGRPC`` client instance.

    Returns:
        List of index name strings.
    """
    return [idx["name"] for idx in pc.list_indexes()]


def wait_until_deleted(
        pc: Pinecone,
        index_name: str,
        timeout_s: int = 60,
) -> bool:
    """Poll until an index is fully deleted or the timeout is reached.

    Args:
        pc: An authenticated ``PineconeGRPC`` client instance.
        index_name: Name of the index to wait on.
        timeout_s: Maximum seconds to wait before returning ``False``.

    Returns:
        ``True`` if the index was deleted within the timeout, ``False`` otherwise.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        if index_name not in list_index_names(pc):
            return True
        time.sleep(2)
    return False


def wait_until_ready(
        pc: Pinecone,
        index_name: str,
        timeout_s: int = 120,
) -> bool:
    """Poll until an index is ready to accept queries or the timeout is reached.

    Readiness is confirmed by a successful ``describe_index_stats()`` call.

    Args:
        pc: An authenticated ``PineconeGRPC`` client instance.
        index_name: Name of the index to wait on.
        timeout_s: Maximum seconds to wait before returning ``False``.

    Returns:
        ``True`` if the index became ready within the timeout, ``False`` otherwise.
    """
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            idx = pc.Index(index_name)
            idx.describe_index_stats()
            return True
        except Exception:
            time.sleep(2)
    return False


def create_single_index(
        pc: Pinecone,
        index_name: str,
        dimension: int,
        force_recreate: bool = False,
) -> None:
    """Create a single Pinecone serverless index (cosine metric, AWS us-east-1).

    If the index already exists and ``force_recreate`` is ``False``, the
    function skips creation. If ``force_recreate`` is ``True``, the existing
    index is deleted and recreated.

    Calls ``sys.exit(1)`` on unrecoverable errors (deletion timeout,
    creation failure, or readiness timeout).

    Args:
        pc: An authenticated ``PineconeGRPC`` client instance.
        index_name: Name of the index to create.
        dimension: Vector dimension for the index.
        force_recreate: If ``True``, delete the existing index before creating.
    """
    existing = list_index_names(pc)

    if index_name in existing:
        if not force_recreate:
            logger.info("Index '%s' already exists. Skipping creation.", index_name)
            return

        logger.warning("Deleting index '%s'...", index_name)
        pc.delete_index(index_name)

        if not wait_until_deleted(pc, index_name):
            logger.error(
                "Timeout while waiting for index '%s' deletion.", index_name
            )
            sys.exit(1)

    logger.info(
        "Creating index '%s' (dim=%d, metric=cosine)...", index_name, dimension
    )

    try:
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-east-1"}},
        )
    except Exception as e:
        logger.error("Failed to create index '%s': %s", index_name, e)
        sys.exit(1)

    if not wait_until_ready(pc, index_name):
        logger.error(
            "Timeout while waiting for index '%s' to be ready.", index_name
        )
        sys.exit(1)

    logger.info("Index '%s' is ready.", index_name)


def create_indexes(force_recreate: bool = False) -> None:
    """Provision both image and text Pinecone indexes for two-stage retrieval.

    Reads index names and dimensions from ``Config``. Calls ``sys.exit(1)``
    if the API key is missing or if any index operation fails.

    Args:
        force_recreate: If ``True``, delete and recreate existing indexes.
    """
    if not Config.PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is missing.")
        sys.exit(1)

    pc = Pinecone(api_key=Config.PINECONE_API_KEY)

    image_index_name = Config.PINECONE_IMAGE_INDEX_NAME
    image_dimension = Config.IMAGE_EMBEDDING_DIM  # SigLIP — 768-dim

    text_index_name = Config.PINECONE_TEXT_INDEX_NAME
    text_dimension = Config.TEXT_EMBEDDING_DIM  # BGE-M3 — 1024-dim

    logger.info(_SEPARATOR)
    logger.info("Creating Two-Stage Retrieval Indexes")
    logger.info(_SEPARATOR)

    logger.info("\n[1/2] Creating Image Index...")
    create_single_index(pc, image_index_name, image_dimension, force_recreate)

    logger.info("\n[2/2] Creating Text Index...")
    create_single_index(pc, text_index_name, text_dimension, force_recreate)

    logger.info("\n%s", _SEPARATOR)
    logger.info("Both indexes created successfully!")
    logger.info("  - Image Index : %s (%d-dim)", image_index_name, image_dimension)
    logger.info("  - Text Index  : %s (%d-dim)", text_index_name, text_dimension)
    logger.info(_SEPARATOR)


if __name__ == "__main__":
    create_indexes(force_recreate=True)
