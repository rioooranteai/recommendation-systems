import logging
from typing import Optional

from app.core.search_engine import SearchEngine
from fastapi import HTTPException
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 60

_embedding_service: Optional[EmbeddingService] = None
_pinecone_service: Optional[PineconeService] = None
_search_engine: Optional[SearchEngine] = None


def set_services(
        embedding_svc: EmbeddingService,
        pinecone_svc: PineconeService,
        search_eng: SearchEngine,
) -> None:
    """Register application-level service singletons.

    Must be called once during application startup (e.g. in the FastAPI
    ``lifespan`` handler) before any request handler invokes the
    ``get_*`` dependency functions.

    Args:
        embedding_svc: Initialized ``EmbeddingService`` instance.
        pinecone_svc: Initialized ``PineconeService`` instance.
        search_eng: Initialized ``SearchEngine`` instance.
    """
    global _embedding_service, _pinecone_service, _search_engine

    _embedding_service = embedding_svc
    _pinecone_service = pinecone_svc
    _search_engine = search_eng

    logger.info(_SEPARATOR)
    logger.info("Dependencies registered:")
    logger.info("  Embedding Service : %s", _embedding_service is not None)
    logger.info("  Pinecone Service  : %s", _pinecone_service is not None)
    logger.info("  Search Engine     : %s", _search_engine is not None)
    logger.info(_SEPARATOR)


def get_search_engine() -> SearchEngine:
    """Return the registered ``SearchEngine`` instance.

    Intended for use as a FastAPI dependency via ``Depends(get_search_engine)``.

    Returns:
        The application-level ``SearchEngine`` singleton.

    Raises:
        HTTPException: 503 if the search engine has not been initialized yet.
    """
    logger.debug(
        "get_search_engine() called. _search_engine is None: %s",
        _search_engine is None,
    )

    if _search_engine is None:
        logger.error("Search engine is None — services not initialized yet.")
        raise HTTPException(
            status_code=503,
            detail={
                "error": "Service Unavailable",
                "message": "Search engine not initialized. Server may still be starting up.",
                "hint": "Please wait a few seconds and try again.",
            },
        )
    return _search_engine


def get_embedding_service() -> EmbeddingService:
    """Return the registered ``EmbeddingService`` instance.

    Intended for use as a FastAPI dependency via ``Depends(get_embedding_service)``.

    Returns:
        The application-level ``EmbeddingService`` singleton.

    Raises:
        HTTPException: 503 if the embedding service has not been initialized yet.
    """
    if _embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized. Server may still be starting up.",
        )
    return _embedding_service


def get_pinecone_service() -> PineconeService:
    """Return the registered ``PineconeService`` instance.

    Intended for use as a FastAPI dependency via ``Depends(get_pinecone_service)``.

    Returns:
        The application-level ``PineconeService`` singleton.

    Raises:
        HTTPException: 503 if the Pinecone service has not been initialized yet.
    """
    if _pinecone_service is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone service not initialized. Server may still be starting up.",
        )
    return _pinecone_service


def cleanup_services() -> None:
    """Dereference all registered service singletons.

    Should be called during application shutdown (e.g. in the FastAPI
    ``lifespan`` handler teardown block) to release connections and
    free memory held by the service instances.
    """
    global _embedding_service, _pinecone_service, _search_engine

    logger.info("Cleaning up services...")

    _embedding_service = None
    _pinecone_service = None
    _search_engine = None

    logger.info("Services cleaned up.")


def is_initialized() -> bool:
    """Check whether all three service singletons have been registered.

    Returns:
        ``True`` if all services are set, ``False`` if any is ``None``.
    """
    return all([
        _embedding_service is not None,
        _pinecone_service is not None,
        _search_engine is not None,
    ])
