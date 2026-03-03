import os
import torch
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Project Paths
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    DATA_DIR = BASE_DIR / "data"
    IMAGE_DIR = DATA_DIR / "fashion-mini/data"

    # Model directories
    SIGLIP_PYTORCH_DIR = MODELS_DIR / "siglip_base"
    SIGLIP_TENSORRT_DIR = MODELS_DIR / "siglip_tensorrt"

    # Image Model Settings
    SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-256"
    USE_TENSORRT = os.getenv('USE_TENSORRT', "false").lower() == 'true'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EMBEDDING_DIM = 768

    # Text Model Settings
    TEXT_MODEL_NAME = "BAAI/bge-m3"
    MAX_TOKEN_LENGTH = 8192

    # Pinecone Settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_NAMESPACE = "Fashion Product"
    PINECONE_IMAGE_INDEX_NAME = "fashion-image-768"
    IMAGE_EMBEDDING_DIM = 768
    PINECONE_TEXT_INDEX_NAME = "fashion-text-1024"
    TEXT_EMBEDDING_DIM = 1024
    PINECONE_RERANK_MODEL = "pinecone-rerank-v0"
    RERANK_WINDOW_SIZE = 50

    # Processing Settings
    TOP_K = 10
    BATCH_SIZE = 24
    IMAGE_SIZE = 224

    # Vector DB config
    USE_GRPC = True
    USE_ASYNC = False

    # TensorRT Settings
    TENSORRT_PRECISION = "fp16"
    TENSORRT_WORKSPACE = 2 << 30

    # Search Engine Config
    _RERANK_CANDIDATES: int = 30
    _RERANK_TOP_N: int = 50
    _RRF_K: int = 60

    # Dataset
    DATASET_PATH = "nirmalsankalana/mini-product-image-and-text-dataset"

    # Index
    _DESCRIPTION_METADATA_LIMIT: int = 200
    _DESCRIPTION_EMBED_LIMIT: int = 500
