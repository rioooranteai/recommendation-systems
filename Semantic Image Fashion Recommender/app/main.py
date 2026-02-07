import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.embedding_service import EmbeddingService
from services.pinecone_service import PineconeService
from app.core.search_engine import SearchEngine
from app.api import routes
from app.schemas import models
from app import dependencies
from config import Config

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
    logger.info("Starting Semantic Fashion Search API...")
    logger.info("=" * 60)

    try:
        # Initialize services
        logger.info("Step 1/3: Initializing Embedding Service...")
        embedding_service = EmbeddingService()
        logger.info(f"✓ Embedding Service ready (dim={embedding_service.get_embedding_dim()})")

        logger.info("Step 2/3: Initializing Pinecone Service...")
        pinecone_service = PineconeService()
        logger.info("✓ Pinecone Service ready")

        logger.info("Step 3/3: Initializing Search Engine...")
        search_engine = SearchEngine(embedding_service, pinecone_service)
        logger.info("✓ Search Engine ready")

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
        logger.info(f"  Model: {Config.SIGLIP_MODEL_NAME}")
        logger.info(f"  Device: {Config.DEVICE}")
        logger.info(f"  Embedding Dimension: {Config.EMBEDDING_DIM}")
        logger.info(f"  Pinecone Index: {Config.PINECONE_INDEX_NAME}")
        logger.info(f"  Pinecone Namespace: {Config.PINECONE_NAMESPACE}")
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
    title="Semantif Fashion Search API",
    description="""
    Multimodal Fashion Search API
    
    Search for fashion items using:
    - **Image search**: Upload an image to find similar items
    - **Text search**: Use text descriptions to find items
    - **Hybrid search**: Combine image + text for refined results
    
    Features:
    - Visual similarity matching using SigLIP embeddings
    - Text-based reranking for preference refinement
    - Category filtering
    - Adjustable image/text weights
    """,
    version="1.0.0",
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
        "message": "Semantic Fashion Search API",
        "version": "1.0.0",
        "status": "running",
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


