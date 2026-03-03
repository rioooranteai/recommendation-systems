import logging
import os
from io import BytesIO
from typing import Optional

from PIL import Image
from app.core.search_engine import SearchEngine
from app.dependencies import get_pinecone_service, get_search_engine, is_initialized
from app.schemas.models import HealthResponse, SearchResponse, StatsResponse
from config import Config
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

router = APIRouter()

_SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/jpg", "image/png"}

_CATEGORIES = [
    "Backpacks",
    "Bags",
    "Belts",
    "Caps",
    "Casual Shirts",
    "Casual Shoes",
    "Flip Flops",
    "Formal Shirts",
    "Formal Shoes",
    "Jeans",
    "Sandals",
    "Shirts",
    "Shorts",
    "Sports Shoes",
    "Sunglasses",
    "Track Pants",
    "Trousers",
    "Tshirts",
    "Wallets",
    "Watches",
]


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Return liveness status and active model/index configuration.

    Returns:
        ``HealthResponse`` containing device, model names, embedding
        dimensions, and Pinecone index identifiers.
    """
    return {
        "status": "healthy",
        "device": Config.DEVICE,
        "image_model": Config.SIGLIP_MODEL_NAME,
        "text_model": Config.TEXT_MODEL_NAME,
        "image_embedding_dim": Config.IMAGE_EMBEDDING_DIM,
        "text_embedding_dim": Config.TEXT_EMBEDDING_DIM,
        "pinecone_namespace": Config.PINECONE_NAMESPACE,
        "image_index": Config.PINECONE_IMAGE_INDEX_NAME,
        "text_index": Config.PINECONE_TEXT_INDEX_NAME,
    }


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
        file: UploadFile = File(..., description="Image file (JPG/PNG)"),
        text_query: Optional[str] = Form(
            None, description="Optional text query for hybrid search"
        ),
        top_k: int = Form(
            10, ge=1, le=50, description="Number of results to return"
        ),
        alpha: float = Form(
            0.7,
            ge=0.0,
            le=1.0,
            description="Image weight (0.0 = text only, 1.0 = image only)",
        ),
        category: Optional[str] = Form(None, description="Filter by category"),
        search_engine: SearchEngine = Depends(get_search_engine),
):
    """Search by image with an optional text query for hybrid retrieval.

    When ``text_query`` is provided, results are fused via RRF using
    ``alpha`` as the image weight. When omitted, performs pure image search.

    Args:
        file: Uploaded image file in JPEG or PNG format.
        text_query: Optional text string to enable hybrid search mode.
        top_k: Number of results to return (1–50).
        alpha: Image contribution weight for RRF fusion (0.0–1.0).
        category: Optional Pinecone metadata filter for product category.
        search_engine: Injected ``SearchEngine`` dependency.

    Returns:
        ``SearchResponse`` with ranked result list.

    Raises:
        HTTPException: 400 for unsupported file type or corrupt image.
        HTTPException: 500 for unexpected search failures.
    """
    try:
        if file.content_type not in _SUPPORTED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid file type: {file.content_type}. "
                    "Supported formats: JPEG, PNG."
                ),
            )

        contents = await file.read()

        try:
            image = Image.open(BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {e}",
            )

        filters = {"category": {"$eq": category}} if category else None

        results = search_engine.search(
            image=image,
            text_query=text_query,
            top_k=top_k,
            filters=filters,
            alpha=alpha,
        )

        return {
            "success": True,
            "query_type": "hybrid" if text_query else "image_only",
            "total_results": len(results),
            "results": results,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Image search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.post("/search/text", response_model=SearchResponse)
def search_by_text(
        text_query: str = Form(..., min_length=1, description="Text description"),
        top_k: int = Form(10, ge=1, le=50, description="Number of results"),
        category: Optional[str] = Form(None, description="Category filter"),
        search_engine: SearchEngine = Depends(get_search_engine),
):
    """Pure text search using BGE-M3 embeddings with optional reranking.

    Delegates to ``SearchEngine.search()`` with ``alpha=0.0`` to enforce
    text-only retrieval mode.

    Args:
        text_query: Non-empty text description to search against.
        top_k: Number of results to return (1–50).
        category: Optional Pinecone metadata filter for product category.
        search_engine: Injected ``SearchEngine`` dependency.

    Returns:
        ``SearchResponse`` with ranked result list.

    Raises:
        HTTPException: 500 for unexpected search failures.
    """
    try:
        filters = {"category": {"$eq": category}} if category else None

        logger.info(
            "Text search: query='%s', top_k=%d", text_query, top_k
        )

        results = search_engine.search(
            image=None,
            text_query=text_query,
            top_k=top_k,
            filters=filters,
            alpha=0.0,
        )

        logger.info("Text search completed: %d results", len(results))

        return {
            "success": True,
            "query_type": "text_only",
            "total_results": len(results),
            "results": results,
        }

    except Exception as e:
        logger.error("Text search failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@router.get("/images/{filename}")
def get_image(filename:str):
    if not filename.endswith(".jpg"):
        filename = f"{filename.strip()}.jpg"
    image_path = os.path.join(Config.IMAGE_DIR, filename)
    if not os.path.exists(image_path):
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(
        path=image_path,
        media_type="image/jpeg",
        headers={"Access-Control-Allow-Origin": "*"}
    )

@router.get("/categories")
def get_categories():
    """Return the list of supported product categories.

    Categories are pre-sorted alphabetically at module load time.

    Returns:
        Dict containing ``success``, ``total_categories``, and
        ``categories`` (sorted list of strings).
    """
    return {
        "success": True,
        "total_categories": len(_CATEGORIES),
        "categories": _CATEGORIES,
    }


@router.get("/stats", response_model=StatsResponse)
def get_stats(
        pinecone_service: PineconeService = Depends(get_pinecone_service),
):
    """Return vector count and dimension statistics for both Pinecone indexes.

    Args:
        pinecone_service: Injected ``PineconeService`` dependency.

    Returns:
        ``StatsResponse`` with per-index statistics.

    Raises:
        HTTPException: 500 if the Pinecone stats query fails.
    """
    try:
        stats = pinecone_service.get_index_stats()

        image_stats = stats["image_index"]
        text_stats = stats["text_index"]

        return {
            "success": True,
            "image_index": {
                "name": Config.PINECONE_IMAGE_INDEX_NAME,
                "total_vectors": image_stats.total_vector_count,
                "dimension": image_stats.dimension,
                "namespaces": {
                    ns: {"vector_count": info.vector_count}
                    for ns, info in image_stats.namespaces.items()
                } if image_stats.namespaces else {},
            },
            "text_index": {
                "name": Config.PINECONE_TEXT_INDEX_NAME,
                "total_vectors": text_stats.total_vector_count,
                "dimension": text_stats.dimension,
                "namespaces": {
                    ns: {"vector_count": info.vector_count}
                    for ns, info in text_stats.namespaces.items()
                } if text_stats.namespaces else {},
            },
        }

    except Exception as e:
        logger.error("Stats query failed: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get index statistics: {e}",
        )


@router.get("/test")
def test_endpoint(
        search_engine: SearchEngine = Depends(get_search_engine),
):
    """Smoke test to verify dependency injection is functioning correctly.

    Args:
        search_engine: Injected ``SearchEngine`` dependency.

    Returns:
        Dict with injection status and ``SearchEngine`` attribute summary.

    Raises:
        HTTPException: 503 if the injected ``search_engine`` is ``None``.
    """
    if search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine is None after dependency injection.",
        )

    return {
        "success": True,
        "message": "Dependency injection working.",
        "search_engine_type": type(search_engine).__name__,
        "has_embedding_service": search_engine.embedding_service is not None,
        "has_pinecone_service": search_engine.pinecone_service is not None,
        "namespace": search_engine.namespace,
    }


@router.get("/readiness")
def readiness_check():
    """Readiness probe for orchestrators (Kubernetes, Docker Compose, etc.).

    Returns 200 only when all service singletons have been registered.

    Returns:
        Dict with ``status`` and ``all_services_initialized`` flag.

    Raises:
        HTTPException: 503 if any service singleton is not yet initialized.
    """
    if not is_initialized():
        raise HTTPException(
            status_code=503,
            detail="Services not ready yet. Server is still initializing.",
        )

    return {
        "status": "ready",
        "all_services_initialized": True,
    }
