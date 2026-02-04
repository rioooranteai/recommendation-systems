import warnings
import transformers
import torch
from typing import List, Union
from PIL import Image
from config import Config
from transformers import AutoModel, AutoProcessor
from .base_model import BaseEmbeddingModel

warnings.filterwarnings('ignore')

transformers.logging.set_verbosity_error()

class SigLIPPytorch(BaseEmbeddingModel):

    def __init__(self):
        self.model = AutoModel.from_pretrained(
            Config.SIGLIP_MODEL_NAME,
            device_map='auto').eval()

        self.processor = AutoProcessor.from_pretrained(
            Config.SIGLIP_MODEL_NAME,
        )

        self.device = Config.DEVICE

    @torch.no_grad()
    def encode_images(self,
                      images: Union[List[Image.Image], Image.Image],
                      batch_size: int = 32) -> torch.Tensor:

        if batch_size is None:
           batch_size = Config.BATCH_SIZE

        if isinstance(images, Image.Image):
            images = [images]

        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]

            inputs = self.processor(
                images=batch,
                return_tensors='pt'
            ).to(self.device)

            embeddings = self.model.get_image_features(**inputs)
            all_embeddings.append(embeddings.cpu())

        return torch.cat(all_embeddings, dim=0)

    @torch.no_grad()
    def encode_text(self,
                    texts: Union[List[str], str]) -> torch.Tensor:

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.processor(
            text=texts,
            return_tensors='pt',
            padding=True
        ).to(self.device)

        embeddings = self.model.get_text_features(**inputs)

        return embeddings.cpu()

    def get_embedding_dim(self) -> int:

        return Config.EMBEDDING_DIM