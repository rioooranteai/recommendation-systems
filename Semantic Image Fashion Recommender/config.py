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

    # Model directories
    SIGLIP_PYTORCH_DIR = MODELS_DIR / "siglip_base"
    SIGLIP_TENSORRT_DIR = MODELS_DIR / "siglip_tensorrt"

    # Model Settings
    SIGLIP_MODEL_NAME = "google/siglip-base-patch16-224"
    USE_TENSORRT = os.getenv('USE_TENSORRT', "false").lower() == 'true'
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'  # typo: cude → cuda
    EMBEDDING_DIM = 512

    # Pinecone Settings
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")  # typo: PPINECONE
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fashion-search")

    # Processing Settings
    TOP_K = 10
    BATCH_SIZE = 32
    IMAGE_SIZE = 224

    # TensorRT Settings
    TENSORRT_PRECISION = "fp16"
    TENSORRT_WORKSPACE = 2 << 30  # 2GB

    @classmethod
    def validate(cls):
        """Validate configuration"""
        errors = []

        if cls.USE_TENSORRT and not torch.cuda.is_available():
            errors.append("TensorRT requires CUDA, but CUDA is not available")

        if cls.USE_TENSORRT and not cls.SIGLIP_TENSORRT_DIR.exists():
            errors.append(f"TensorRT model not found at {cls.SIGLIP_TENSORRT_DIR}")

        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY not set in .env")

        if errors:
            raise ValueError(f"Configuration errors:\n" + "\n".join(f"  ❌ {e}" for e in errors))

        return True