import logging
from contextlib import asynccontextmanager
from pathlib import Path

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
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
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

        # Register services
        logger.info("Registering services to dependency container...")
        dependencies.set_services(
            embedding_svc=embedding_service,
            pinecone_svc=pinecone_service,
            search_eng=search_engine
        )

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
    description="Two-Stage Multimodal Fashion Search API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ✅ CORS - Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ CRITICAL: Mount static images directory
IMAGE_DIR = Path(r"D:\recommendation-systems\Semantic Image Fashion Recommender\data\fashion-mini\data")

if IMAGE_DIR.exists():
    app.mount("/static/images", StaticFiles(directory=str(IMAGE_DIR)), name="images")
    logger.info("=" * 60)
    logger.info(f"✅ Static images mounted successfully!")
    logger.info(f"📁 Image directory: {IMAGE_DIR}")
    logger.info(f"🌐 URL pattern: http://127.0.0.1:8000/static/images/{{product_id}}.jpg")

    # List some sample files
    sample_files = list(IMAGE_DIR.glob("*.jpg"))[:5]
    if sample_files:
        logger.info(f"📸 Sample images found:")
        for img in sample_files:
            logger.info(f"   - {img.name}")
    logger.info("=" * 60)
else:
    logger.error("=" * 60)
    logger.error(f"❌ IMAGE DIRECTORY NOT FOUND!")
    logger.error(f"📁 Expected path: {IMAGE_DIR}")
    logger.error(f"⚠️  Images will not be available!")
    logger.error("=" * 60)

# Include API routes
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
        "static_images": {
            "enabled": IMAGE_DIR.exists(),
            "path": str(IMAGE_DIR),
            "url_pattern": "http://127.0.0.1:8000/static/images/{product_id}.jpg"
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