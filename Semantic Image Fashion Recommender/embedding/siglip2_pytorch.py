import logging
import warnings
from typing import List, Optional, Union

import numpy as np
import torch
import transformers
from PIL import Image
from config import Config
from transformers import AutoModel, AutoProcessor

from .base_model import BaseEmbeddingModel

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class SigLIPPytorch(BaseEmbeddingModel):
    """PyTorch wrapper for the SigLIP vision-language embedding model.

    Loads the model onto the device specified in ``Config.DEVICE`` and sets
    it to evaluation mode. Supports batched image encoding and single-image
    convenience methods.

    Image embeddings are produced via ``get_image_features()``, using
    ``pooler_output`` when available, otherwise the raw output tensor.
    Text embeddings follow the same pooling strategy via
    ``get_text_features()``.
    """

    def __init__(self) -> None:
        """Load the SigLIP model and processor onto the configured device."""
        self.device = Config.DEVICE

        logger.info("Loading SigLIP model onto device '%s'...", self.device)

        self.model = (
            AutoModel.from_pretrained(Config.SIGLIP_MODEL_NAME)
            .to(Config.DEVICE)
            .eval()
        )
        self.processor = AutoProcessor.from_pretrained(Config.SIGLIP_MODEL_NAME)

        logger.info(
            "SigLIP model loaded successfully on '%s'.", self.device
        )

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply L2 normalisation along the embedding dimension.

        Args:
            embeddings: Float tensor of shape ``(batch, dim)``.

        Returns:
            L2-normalised tensor of the same shape.
        """
        return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(
            self,
            images: Union[List[Image.Image], Image.Image],
            batch_size: Optional[int] = None,
            normalize: bool = True,
    ) -> np.ndarray:
        """Encode one or more PIL images into SigLIP embeddings.

        Images are processed in batches of ``batch_size``. When a single
        ``Image.Image`` is provided, the returned array is flattened to
        shape ``(768,)`` rather than ``(1, 768)``.

        Args:
            images: A single PIL image or a list of PIL images to encode.
            batch_size: Number of images per batch. Defaults to
                ``Config.BATCH_SIZE`` when ``None``.
            normalize: If ``True``, applies L2 normalisation to each embedding.

        Returns:
            NumPy array of shape ``(n_images, 768)`` for a list input,
            or ``(768,)`` for a single image input.
        """
        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        if isinstance(images, Image.Image):
            images = [images]
            single_image = True
        else:
            single_image = False

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i: i + batch_size]

            inputs = self.processor(
                images=batch,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.model.get_image_features(**inputs)

            embeddings = (
                outputs.pooler_output
                if hasattr(outputs, "pooler_output")
                else outputs
            )

            if normalize:
                embeddings = self._normalize_embeddings(embeddings)

            all_embeddings.append(embeddings.cpu())

        result = torch.cat(all_embeddings, dim=0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_np = result.numpy()
        if single_image and result_np.ndim == 2:
            result_np = result_np.flatten()

        return result_np

    @torch.no_grad()
    def encode_text(
            self,
            texts: Union[List[str], str],
            normalize: bool = True,
    ) -> np.ndarray:
        """Encode one or more text strings into SigLIP text embeddings.

        When a single string is provided, the returned array is flattened
        to shape ``(768,)`` rather than ``(1, 768)``.

        Args:
            texts: A single string or a list of strings to encode.
            normalize: If ``True``, applies L2 normalisation to each embedding.

        Returns:
            NumPy array of shape ``(n_texts, 768)`` for a list input,
            or ``(768,)`` for a single string input.
        """
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        inputs = self.processor(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)

        embeddings = (
            outputs.pooler_output
            if hasattr(outputs, "pooler_output")
            else outputs
        )

        if normalize:
            embeddings = self._normalize_embeddings(embeddings)

        result = embeddings.cpu()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_np = result.numpy()
        if single_text and result_np.ndim == 2:
            result_np = result_np.flatten()

        return result_np

    def encode_image(
            self,
            image: Image.Image,
            normalize: bool = True,
    ) -> np.ndarray:
        """Convenience wrapper to encode a single image.

        Delegates to ``encode_images()`` and returns the first (only) result.

        Args:
            image: A single PIL image to encode.
            normalize: If ``True``, applies L2 normalisation to the embedding.

        Returns:
            NumPy array of shape ``(768,)``.
        """
        return self.encode_images([image], normalize=normalize)[0]

    def get_embedding_dim(self) -> int:
        """Return the output embedding dimension of the SigLIP model.

        Returns:
            Integer dimension as defined by ``Config.EMBEDDING_DIM``
            (typically 768).
        """
        return Config.EMBEDDING_DIM
