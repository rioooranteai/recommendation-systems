import logging
from contextlib import asynccontextmanager

import uvicorn
from app import dependencies
from app.api import routes
from app.core.search_engine import SearchEngine
from config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    """
    # ========== STARTUP ==========
    logger.info("=" * 60)
    logger.info("Starting Semantic Fashion Search API (Two-Stage)")
    logger.info("=" * 60)

    try:
        # Initialize services
        logger.info("Step 1/3: Initializing Embedding Service...")
        embedding_service = EmbeddingService()
        logger.info(f"✓ Image Embedding ready (SigLIP, dim={embedding_service.get_image_embedding_dim()})")
        logger.info(f"✓ Text Embedding ready (BGE-M3, dim={embedding_service.get_embedding_dim()})")

        logger.info("Step 2/3: Initializing Pinecone Service...")
        pinecone_service = PineconeService()
        logger.info("✓ Pinecone Service ready (dual indexes)")

        logger.info("Step 3/3: Initializing Search Engine...")
        search_engine = SearchEngine(embedding_service, pinecone_service)
        logger.info("✓ Search Engine ready (two-stage retrieval)")

        # Register services to dependency injection container
        logger.info("Registering services to dependency container...")
        dependencies.set_services(
            embedding_svc=embedding_service,
            pinecone_svc=pinecone_service,
            search_eng=search_engine
        )

        # Verify registration
        if not dependencies.is_initialized():
            raise RuntimeError("Services registration failed!")

        logger.info("=" * 60)
        logger.info("Configuration:")
        logger.info(f"  Image Model: {Config.SIGLIP_MODEL_NAME}")
        logger.info(f"  Text Model: {Config.TEXT_MODEL_NAME}")
        logger.info(f"  Device: {Config.DEVICE}")
        logger.info(f"  Image Dimension: {Config.IMAGE_EMBEDDING_DIM}")
        logger.info(f"  Text Dimension: {Config.TEXT_EMBEDDING_DIM}")
        logger.info(f"  Image Index: {Config.PINECONE_IMAGE_INDEX_NAME}")
        logger.info(f"  Text Index: {Config.PINECONE_TEXT_INDEX_NAME}")
        logger.info(f"  Namespace: {Config.PINECONE_NAMESPACE}")
        logger.info("=" * 60)
        logger.info("✓✓✓ ALL SERVICES INITIALIZED SUCCESSFULLY! ✓✓✓")
        logger.info("✓✓✓ API IS READY TO ACCEPT REQUESTS ✓✓✓")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"✗✗✗ FAILED TO INITIALIZE SERVICES ✗✗✗")
        logger.error(f"Error: {e}")
        logger.error("=" * 60)
        raise

    yield

    # ========== SHUTDOWN ==========
    logger.info("=" * 60)
    logger.info("Shutting down services...")
    dependencies.cleanup_services()
    logger.info("✓ Goodbye!")
    logger.info("=" * 60)


app = FastAPI(
    title="Semantic Fashion Search API",
    description="""
    Two-Stage Multimodal Fashion Search API

    Search for fashion items using:
    - **Image search**: Upload an image to find visually similar items (SigLIP 768-dim)
    - **Text search**: Use natural language descriptions (BGE-M3 1024-dim)
    - **Hybrid search**: Combine image + text with RRF fusion for best results

    Features:
    - Dual-index architecture (separate image & text indexes)
    - Visual similarity matching using SigLIP embeddings
    - Semantic text search using BGE-M3 embeddings
    - Reciprocal Rank Fusion (RRF) for hybrid results
    - Category filtering
    - Adjustable alpha weight (0=text only, 1=image only)

    Models:
    - Image: google/siglip-so400m-patch14-384 (768-dim)
    - Text: BAAI/bge-m3 (1024-dim)
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix='/api', tags=['Search'])


@app.get("/", tags=['Root'])
def root():
    return {
        "message": "Semantic Fashion Search API (Two-Stage Architecture)",
        "version": "2.0.0",
        "status": "running",
        "architecture": "two-stage-retrieval",
        "models": {
            "image": Config.SIGLIP_MODEL_NAME,
            "text": Config.TEXT_MODEL_NAME
        },
        "indexes": {
            "image": Config.PINECONE_IMAGE_INDEX_NAME,
            "text": Config.PINECONE_TEXT_INDEX_NAME
        },
        "documentation": {
            "swagger": "/docs",
            "redoc": "/redoc"
        },
        "endpoints": {
            "health": "/api/health",
            "image_search": "/api/search/image",
            "text_search": "/api/search/text",
            "categories": "/api/categories",
            "stats": "/api/stats"
        }
    }


@app.get("/ping", tags=['Root'])
def ping():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )