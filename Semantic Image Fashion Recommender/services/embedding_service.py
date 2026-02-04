from config import Config
from embedding.base_model import BaseEmbeddingModel
from embedding.siglip_pytorch import SigLIPPytorch


class EmbeddingService:
    _instance = None
    _model: BaseEmbeddingModel = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False

        return cls._instance

    def __init__(self):

        if self._initialized:
            return

        if Config.USE_TENSORRT:
            self._model = SigLIPPytorch()

        else:
            self._model = SigLIPPytorch()

        self._initialized = True

    def encode_images(self, images, batch_size=None):
        """Encode images - delegates to PyTorch or TensorRT"""
        return self._model.encode_images(images, batch_size)

    def encode_text(self, texts):
        """Encode text - delegates to PyTorch or TensorRT"""
        return self._model.encode_text(texts)

    def get_embedding_dim(self):
        """Get embedding dimension"""
        return self._model.get_embedding_dim()

    def get_model_type(self) -> str:
        """Get current model type"""
        return "TensorRT" if Config.USE_TENSORRT else "PyTorch"
