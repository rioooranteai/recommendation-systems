import logging
from typing import Any, List, Optional, Union

from config import Config
from embedding.base_model import BaseEmbeddingModel
from embedding.bge_m3_pytorch import BGEM3Pytorch
from embedding.siglip2_pytorch import SigLIPPytorch

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Singleton embedding service wrapping image and text encoder models.

    Ensures only one instance is created across the application lifecycle,
    avoiding redundant model loading into GPU/CPU memory.

    Models:
        - Image : SigLIP (768-dim), optionally accelerated via TensorRT.
        - Text  : BGE-M3 (1024-dim), PyTorch.
    """

    _instance: Optional["EmbeddingService"] = None
    _image_model: BaseEmbeddingModel = None
    _text_model: BaseEmbeddingModel = None

    def __new__(cls) -> "EmbeddingService":
        """Return the existing singleton instance or create a new one.

        Returns:
            The single shared ``EmbeddingService`` instance.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        """Load image and text models on first instantiation only.

        Subsequent calls are no-ops due to the singleton guard.
        """
        if self._initialized:
            return
        self._image_model = SigLIPPytorch()
        self._text_model = BGEM3Pytorch()
        self._initialized = True
        logger.info(
            "EmbeddingService initialized — image: SigLIPPytorch, text: BGEM3Pytorch"
        )

    def encode_images(
            self,
            images: Any,
            batch_size: Optional[int] = None,
    ) -> Any:
        """Encode one or more images using the SigLIP model.

        Args:
            images: A single PIL image or a list of PIL images to encode.
            batch_size: Optional batch size override for the underlying model.

        Returns:
            Image embeddings as returned by the SigLIP model.
        """
        return self._image_model.encode_images(images, batch_size)

    def encode_text(self, texts: Union[str, List[str]]) -> Any:
        """Encode one or more text strings using the BGE-M3 model.

        Args:
            texts: A single string or a list of strings to encode.

        Returns:
            Text embeddings as returned by the BGE-M3 model.
        """
        return self._text_model.encode_text(texts)

    def get_embedding_dim(self) -> int:
        """Return the embedding dimension of the text model (BGE-M3).

        Returns:
            Integer dimension of text embeddings (typically 1024).
        """
        return self._text_model.get_embedding_dim()

    def get_image_embedding_dim(self) -> int:
        """Return the embedding dimension of the image model (SigLIP).

        Returns:
            Integer dimension of image embeddings (typically 768).
        """
        return self._image_model.get_embedding_dim()

    def get_model_type(self) -> str:
        """Return a human-readable description of the active model backends.

        The image backend is either TensorRT or PyTorch depending on
        ``Config.USE_TENSORRT``. The text backend is always BGE-M3 PyTorch.

        Returns:
            A string in the format ``"Image: <backend>, Text: BGE-M3-PyTorch"``.
        """
        image_type = "TensorRT" if Config.USE_TENSORRT else "PyTorch"
        return f"Image: {image_type}, Text: BGE-M3-PyTorch"
