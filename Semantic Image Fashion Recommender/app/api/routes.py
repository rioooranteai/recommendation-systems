import logging
from io import BytesIO
from typing import Optional

from PIL import Image
from app.core.search_engine import SearchEngine
from app.dependencies import get_search_engine, get_pinecone_service, is_initialized
from app.schemas.models import SearchResponse, HealthResponse, StatsResponse
from config import Config
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Depends
from services.pinecone_service import PineconeService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "device": Config.DEVICE,
        "image_model": Config.IMAGE_MODEL_NAME,  # Changed
        "text_model": Config.TEXT_MODEL_NAME,  # Added
        "image_embedding_dim": Config.IMAGE_EMBEDDING_DIM,  # Changed
        "text_embedding_dim": Config.TEXT_EMBEDDING_DIM,  # Added
        "pinecone_namespace": Config.PINECONE_NAMESPACE,
        "image_index": Config.PINECONE_IMAGE_INDEX_NAME,  # Changed
        "text_index": Config.PINECONE_TEXT_INDEX_NAME  # Added
    }


@router.post("/search/image", response_model=SearchResponse)
async def search_by_image(
        file: UploadFile = File(..., description="Image file (JPG/PNG)"),
        text_query: Optional[str] = Form(None, description="Optional text query for hybrid search"),
        top_k: int = Form(10, ge=1, le=50, description="Number of results to return"),
        alpha: float = Form(0.7, ge=0.0, le=1.0, description="Image weight (0=text only, 1=image only)"),
        category: Optional[str] = Form(None, description="Filter by category"),
        search_engine: SearchEngine = Depends(get_search_engine)
):
    """
    Search by image with optional text query.
    - alpha=1.0: pure image search
    - alpha=0.5: balanced image+text
    - alpha=0.0: pure text search (but why use this endpoint?)
    """
    try:
        if file.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Supported formats: JPEG, PNG"
            )

        contents = await file.read()

        try:
            image = Image.open(BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid image file: {str(e)}"
            )

        filters = {}
        if category:
            filters['category'] = {"$eq": category}

        # Call search with new parameters
        results = search_engine.search(
            image=image,
            text_query=text_query,
            top_k=top_k,
            filters=filters if filters else None,
            alpha=alpha  # Changed from image_weight/text_weight
        )

        return {
            "success": True,
            "query_type": "hybrid" if text_query else "image_only",
            "total_results": len(results),
            "results": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/search/text", response_model=SearchResponse)
def search_by_text(
        text_query: str = Form(..., min_length=1, description="Text description"),
        top_k: int = Form(10, ge=1, le=50, description="Number of results"),
        category: Optional[str] = Form(None, description="Category filter"),
        search_engine: SearchEngine = Depends(get_search_engine)
):
    """Pure text search using BGE-M3 embeddings"""
    try:
        filters = {}
        if category:
            filters['category'] = {"$eq": category}

        logger.info(f"Text search: query='{text_query}', top_k={top_k}")

        # Text-only search (alpha=0 means pure text)
        results = search_engine.search(
            image=None,
            text_query=text_query,
            top_k=top_k,
            filters=filters if filters else None,
            alpha=0.0  # Pure text search
        )

        logger.info(f"Text search completed: {len(results)} results")

        return {
            "success": True,
            "query_type": "text_only",
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        logger.error(f"Text search failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/categories")
def get_categories():
    categories = [
        "Tshirts",
        "Shirts",
        "Casual Shirts",
        "Formal Shirts",
        "Jeans",
        "Trousers",
        "Track Pants",
        "Shorts",
        "Casual Shoes",
        "Formal Shoes",
        "Sports Shoes",
        "Sandals",
        "Flip Flops",
        "Watches",
        "Bags",
        "Backpacks",
        "Sunglasses",
        "Belts",
        "Wallets",
        "Caps"
    ]

    return {
        "success": True,
        "total_categories": len(categories),
        "categories": sorted(categories)
    }


@router.get("/stats", response_model=StatsResponse)
def get_stats(
        pinecone_service: PineconeService = Depends(get_pinecone_service)
):
    """Get statistics from both image and text indexes"""
    try:
        # Get stats from both indexes
        stats = pinecone_service.get_index_stats()

        image_stats = stats['image_index']
        text_stats = stats['text_index']

        return {
            "success": True,
            "image_index": {
                "name": Config.PINECONE_IMAGE_INDEX_NAME,
                "total_vectors": image_stats.total_vector_count,
                "dimension": image_stats.dimension,
                "namespaces": {
                    ns: {"vector_count": info.vector_count}
                    for ns, info in image_stats.namespaces.items()
                } if image_stats.namespaces else {}
            },
            "text_index": {
                "name": Config.PINECONE_TEXT_INDEX_NAME,
                "total_vectors": text_stats.total_vector_count,
                "dimension": text_stats.dimension,
                "namespaces": {
                    ns: {"vector_count": info.vector_count}
                    for ns, info in text_stats.namespaces.items()
                } if text_stats.namespaces else {}
            }
        }

    except Exception as e:
        logger.error(f"Stats query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get index statistics: {str(e)}"
        )


@router.get("/test")
def test_endpoint(
        search_engine: SearchEngine = Depends(get_search_engine)
):
    if search_engine is None:
        raise HTTPException(
            status_code=503,
            detail="Search engine is None after dependency injection!"
        )

    return {
        "success": True,
        "message": "Dependency injection working!",
        "search_engine_type": type(search_engine).__name__,
        "has_embedding_service": search_engine.embedding_service is not None,
        "has_pinecone_service": search_engine.pinecone_service is not None,
        "namespace": search_engine.namespace
    }


@router.get("/readiness")
def readiness_check():
    ready = is_initialized()

    if not ready:
        raise HTTPException(
            status_code=503,
            detail="Services not ready yet. Server is still initializing."
        )

    return {
        "status": "ready",
        "all_services_initialized": True
    }