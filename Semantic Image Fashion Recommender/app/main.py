import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

import uvicorn
from app import dependencies
from app.api import routes
from app.core.search_engine import SearchEngine
from config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

_SEPARATOR = "=" * 60

_IMAGE_DIR = Path(
    r"D:\recommendation-systems\Semantic Image Fashion Recommender\data\fashion-mini\data"
)

_API_BASE_URL = "http://127.0.0.1:8000"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Manage application startup and shutdown lifecycle.

    Startup:
        1. Initializes ``EmbeddingService`` (SigLIP + BGE-M3).
        2. Initializes ``PineconeService`` (dual gRPC indexes).
        3. Initializes ``SearchEngine`` (two-stage retrieval).
        4. Registers all services to the dependency container.

    Shutdown:
        Calls ``dependencies.cleanup_services()`` to release resources.

    Raises:
        Exception: Re-raises any initialization error after logging,
            which prevents the application from starting.
    """
    logger.info(_SEPARATOR)
    logger.info("Starting Semantic Fashion Search API (Two-Stage)")
    logger.info(_SEPARATOR)

    try:
        logger.info("Step 1/3: Initializing Embedding Service...")
        embedding_service = EmbeddingService()
        logger.info(
            "Image Embedding ready (SigLIP, dim=%d)",
            embedding_service.get_image_embedding_dim(),
        )
        logger.info(
            "Text Embedding ready (BGE-M3, dim=%d)",
            embedding_service.get_embedding_dim(),
        )

        logger.info("Step 2/3: Initializing Pinecone Service...")
        pinecone_service = PineconeService()
        logger.info("Pinecone Service ready (dual indexes)")

        logger.info("Step 3/3: Initializing Search Engine...")
        search_engine = SearchEngine(embedding_service, pinecone_service)
        logger.info("Search Engine ready (two-stage retrieval)")

        logger.info("Registering services to dependency container...")
        dependencies.set_services(
            embedding_svc=embedding_service,
            pinecone_svc=pinecone_service,
            search_eng=search_engine,
        )

        if not dependencies.is_initialized():
            raise RuntimeError("Services registration failed.")

        logger.info(_SEPARATOR)
        logger.info("Configuration:")
        logger.info("  Image Model      : %s", Config.SIGLIP_MODEL_NAME)
        logger.info("  Text Model       : %s", Config.TEXT_MODEL_NAME)
        logger.info("  Device           : %s", Config.DEVICE)
        logger.info("  Image Dimension  : %d", Config.IMAGE_EMBEDDING_DIM)
        logger.info("  Text Dimension   : %d", Config.TEXT_EMBEDDING_DIM)
        logger.info("  Image Index      : %s", Config.PINECONE_IMAGE_INDEX_NAME)
        logger.info("  Text Index       : %s", Config.PINECONE_TEXT_INDEX_NAME)
        logger.info("  Namespace        : %s", Config.PINECONE_NAMESPACE)
        logger.info(_SEPARATOR)
        logger.info("All services initialized successfully.")
        logger.info("API is ready to accept requests.")
        logger.info(_SEPARATOR)

    except Exception as e:
        logger.error(_SEPARATOR)
        logger.error("Failed to initialize services.")
        logger.error("Error: %s", e)
        logger.error(_SEPARATOR)
        raise

    yield

    logger.info(_SEPARATOR)
    logger.info("Shutting down services...")
    dependencies.cleanup_services()
    logger.info("Shutdown complete.")
    logger.info(_SEPARATOR)


app = FastAPI(
    title="Semantic Fashion Search API",
    description="Two-Stage Multimodal Fashion Search API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if _IMAGE_DIR.exists():
    app.mount(
        "/static/images",
        StaticFiles(directory=str(_IMAGE_DIR)),
        name="images",
    )
    logger.info(_SEPARATOR)
    logger.info("Static images mounted successfully.")
    logger.info("Image directory : %s", _IMAGE_DIR)
    logger.info(
        "URL pattern     : %s/static/images/{product_id}.jpg", _API_BASE_URL
    )

    sample_files = list(_IMAGE_DIR.glob("*.jpg"))[:5]
    if sample_files:
        logger.info("Sample images found:")
        for img in sample_files:
            logger.info("  - %s", img.name)

    logger.info(_SEPARATOR)
else:
    logger.error(_SEPARATOR)
    logger.error("Image directory not found — static images will be unavailable.")
    logger.error("Expected path: %s", _IMAGE_DIR)
    logger.error(_SEPARATOR)

app.include_router(routes.router, prefix="/api", tags=["Search"])


@app.get("/", tags=["Root"])
def root():
    """Return API metadata and available endpoint map.

    Returns:
        Dict containing version, architecture, model names, index names,
        static image status, and endpoint paths.
    """
    return {
        "message": "Semantic Fashion Search API (Two-Stage Architecture)",
        "version": "2.0.0",
        "status": "running",
        "architecture": "two-stage-retrieval",
        "models": {
            "image": Config.SIGLIP_MODEL_NAME,
            "text": Config.TEXT_MODEL_NAME,
        },
        "indexes": {
            "image": Config.PINECONE_IMAGE_INDEX_NAME,
            "text": Config.PINECONE_TEXT_INDEX_NAME,
        },
        "static_images": {
            "enabled": _IMAGE_DIR.exists(),
            "path": str(_IMAGE_DIR),
            "url_pattern": f"{_API_BASE_URL}/static/images/{{product_id}}.jpg",
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc",
        },
        "endpoints": {
            "health": "/api/health",
            "image_search": "/api/search/image",
            "text_search": "/api/search/text",
            "categories": "/api/categories",
            "stats": "/api/stats",
        },
    }


@app.get("/ping", tags=["Root"])
def ping():
    """Minimal liveness probe — returns immediately with no dependencies.

    Returns:
        Dict with ``status: "ok"``.
    """
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
