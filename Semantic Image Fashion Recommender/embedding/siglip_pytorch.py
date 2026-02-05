import warnings
import transformers
import torch
import numpy as np
import logging
from typing import List, Union
from PIL import Image
from config import Config
from transformers import AutoModel, AutoProcessor
from .base_model import BaseEmbeddingModel

# Suppress warnings
warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class SigLIPPytorch(BaseEmbeddingModel):
    def __init__(self):

        self.model = AutoModel.from_pretrained(
            Config.SIGLIP_MODEL_NAME,
            device_map='auto'
        ).eval()

        self.processor = AutoProcessor.from_pretrained(
            Config.SIGLIP_MODEL_NAME,
        )

        self.device = Config.DEVICE
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

        return result.numpy()

    @torch.no_grad()
    def encode_text(
            self,
            texts: Union[List[str], str],
            normalize: bool = True
    ) -> np.ndarray:

        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Process texts
        inputs = self.processor(
            text=texts,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).to(self.device)

        # Get text features
        outputs = self.model.get_text_features(**inputs)

        if hasattr(outputs, 'pooler_output'):
            embeddings = outputs.pooler_output
        else:
            embeddings = outputs

        # Normalize if requested
        if normalize:
            embeddings = self._normalize_embeddings(embeddings)

        # Move to CPU
        result = embeddings.cpu()

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Return as numpy array
        return result.numpy()

    def encode_image(self, image: Image.Image, normalize: bool = True) -> np.ndarray:
        embeddings = self.encode_images([image], normalize=normalize)
        return embeddings[0]

    def get_embedding_dim(self) -> int:
        return Config.EMBEDDING_DIM
