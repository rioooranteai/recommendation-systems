from typing import Optional
from fastapi import HTTPException

from app.core.search_engine import SearchEngine
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService

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

def get_search_engine() -> SearchEngine:
    if _search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine not initialized. Server may still be starting up"
        )

def get_embedding_service() -> EmbeddingService:
    if _embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding service not initialized. Server may still be starting up."
        )

def get_pinecone_service() -> PineconeService:
    if _pinecone_service is None:
        raise HTTPException(
            status_code=503,
            detail="Pinecone service not initialized. Server may still be starting up."
        )

def cleanup_services() -> None:
    global _embedding_service, _pinecone_service, _search_engine

    _embedding_service = None
    _pinecone_service = None
    _search_engine = None

