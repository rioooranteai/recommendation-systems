import os
import torch

from pathlib import Path
from dotenv import load_dotenv

class Config:
    
      BASE_DIR = Path.parent
      MODELS_DIR = BASE_DIR / "Models"
      DATA_DIR = BASE_DIR / "data"
      
      SIGLIP_MODEL_NAME = ""
      USE_TENSORRT = os.getenv('USE_TENSORRT', "false").lower() == 'true'
      DEVICE = 'cude' if torch.cuda.is_available() else 'cpu'
      
      PPINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
      PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fashion-search")
      PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fashion-search")
      
      TOP_K = 10
      BATCH_SIZE = 32
      IMAGE_SIZE = 224
        
    
      TENSORRT_PRECISION = "fp16"  
      TENSORRT_WORKSPACE = 2 << 30