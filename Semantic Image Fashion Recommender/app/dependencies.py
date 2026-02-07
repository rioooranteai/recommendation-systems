import logging
from typing import Optional
from fastapi import HTTPException

from app.core.search_engine import SearchEngine
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

_embedding_service: Optional[EmbeddingService] = None
_pinecone_service: Optional[PineconeService] = None
_search_engine: Optional[SearchEngine] = None


def set_services(
        embedding_svc: EmbeddingService,
        pinecone_svc: PineconeService,
        search_eng: SearchEngine
) -> None:
    global _embedding_service, _pinecone_service, _search_engine

    _embedding_service = embedding_svc
    _pinecone_service = pinecone_svc
    _search_engine = search_eng

    # Debug log untuk memastikan services ter-set
    logger.info("=" * 60)
    logger.info("Dependencies registered:")
    logger.info(f"  ✓ Embedding Service: {_embedding_service is not None}")
    logger.info(f"  ✓ Pinecone Service: {_pinecone_service is not None}")
    logger.info(f"  ✓ Search Engine: {_search_engine is not None}")
    logger.info("=" * 60)


def get_search_engine() -> SearchEngine:
    logger.debug(f"get_search_engine() called. _search_engine is None: {_search_engine is None}")

    if _search_engine is None:
        logger.error("Search engine is None! Services not initialized yet.")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service Unavailable",
                "message": "Search engine not initialized. Server may still be starting up.",
                "hint": "Please wait a few seconds and try again."
            }
        )
    return _search_engine


def get_embedding_service() -> EmbeddingService:
    if _embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized. Server may still be starting up."
        )
    return _embedding_service


def get_pinecone_service() -> PineconeService:
    if _pinecone_service is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone service not initialized. Server may still be starting up."
        )
    return _pinecone_service


def cleanup_services() -> None:
    global _embedding_service, _pinecone_service, _search_engine

    logger.info("Cleaning up services...")

    # Add cleanup logic if needed (close connections, free memory, etc.)
    _embedding_service = None
    _pinecone_service = None
    _search_engine = None

    logger.info("✓ Services cleaned up")


def is_initialized() -> bool:
    return all([
        _embedding_service is not None,
        _pinecone_service is not None,
        _search_engine is not None
    ])