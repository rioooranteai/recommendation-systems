import logging
import warnings
from typing import List, Union

import numpy as np
import torch
import transformers
from PIL import Image
from config import Config
from transformers import AutoModel, AutoProcessor

from .base_model import BaseEmbeddingModel

warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class SigLIPPytorch(BaseEmbeddingModel):
    def __init__(self):
        self.device = Config.DEVICE
        self.model = AutoModel.from_pretrained(
            Config.SIGLIP_MODEL_NAME,
        ).to(Config.DEVICE).eval()
        self.processor = AutoProcessor.from_pretrained(
            Config.SIGLIP_MODEL_NAME,
        )

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:

        return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(
            self,
            images: Union[List[Image.Image], Image.Image],
            batch_size: int = None,
            normalize: bool = True
    ) -> np.ndarray:

        if batch_size is None:
            batch_size = Config.BATCH_SIZE

        # Handle single image
        if isinstance(images, Image.Image):
            images = [images]
            single_image = True
        else:
            single_image = False

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Process batch
            inputs = self.processor(
                images=batch,
                return_tensors='pt'
            ).to(self.device)

            # Get image features
            outputs = self.model.get_image_features(**inputs)

            if hasattr(outputs, 'pooler_output'):
                embeddings = outputs.pooler_output
            else:
                embeddings = outputs

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
            normalize: bool = True
    ) -> np.ndarray:

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False

        inputs = self.processor(
            text=texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)

        outputs = self.model.get_text_features(**inputs)

        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs

        if normalize:
            embeddings = self._normalize_embeddings(embeddings)

        result = embeddings.cpu()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        result_np = result.numpy()
        if single_text and result_np.ndim == 2:
            result_np = result_np.flatten()

        return result_np

    def encode_image(self, image: Image.Image, normalize: bool = True) -> np.ndarray:
        embeddings = self.encode_images([image], normalize=normalize)
        return embeddings[0]

    def get_embedding_dim(self) -> int:
        return Config.EMBEDDING_DIM
