from config import Config
from embedding.base_model import BaseEmbeddingModel
from embedding.siglip2_pytorch import SigLIPPytorch
from embedding.qwen3_pytorch import Qwen3Pytorch


class EmbeddingService:
    _instance = None
    _image_model: BaseEmbeddingModel = None
    _text_model: BaseEmbeddingModel = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._image_model = SigLIPPytorch()
        self._text_model = Qwen3Pytorch()
        self._initialized = True

    def encode_images(self, images, batch_size=None):
        """Encode images - delegates to SigLIP model"""
        return self._image_model.encode_images(images, batch_size)

    def encode_text(self, texts):
        """Encode text - delegates to Qwen3 model"""
        return self._text_model.encode_text(texts)

    def get_embedding_dim(self):
        """Get embedding dimension for text model"""
        return self._text_model.get_embedding_dim()

    def get_image_embedding_dim(self):
        """Get embedding dimension for image model"""
        return self._image_model.get_embedding_dim()

    def get_model_type(self) -> str:
        """Get current model types"""
        image_type = "TensorRT" if Config.USE_TENSORRT else "PyTorch"
        return f"Image: {image_type}, Text: Qwen3-PyTorch"

test_qwen = ["My Name Is Mario"]

service_test = EmbeddingService()

print(service_test.encode_text(test_qwen))