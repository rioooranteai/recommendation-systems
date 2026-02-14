import logging
import warnings
from typing import List, Union

import numpy as np
import torch
import transformers
from config import Config
from transformers import AutoModel, AutoTokenizer

from embedding.base_model import BaseEmbeddingModel

warnings.filterwarnings('ignore')
transformers.logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)


class Qwen3Pytorch(BaseEmbeddingModel):
    def __init__(self):
        self.device = Config.DEVICE
        self.model = AutoModel.from_pretrained(
            Config.TEXT_MODEL_NAME,
            torch_dtype=torch.float16
        ).to(Config.DEVICE).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            Config.TEXT_MODEL_NAME
        )

    def _normalize_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / embeddings.norm(p=2, dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(
            self,
            texts: Union[List[str], str],
            normalize: bool = True
    ) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=Config.MAX_TOKEN_LENGTH
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

        if normalize:
            embeddings = self._normalize_embeddings(embeddings)

        embeddings = embeddings.cpu().numpy()

        return embeddings

    def encode_images(self, images, batch_size: int = 32) -> torch.Tensor:
        """
        Qwen3 adalah text model, tidak support image encoding.
        Method ini hanya untuk memenuhi interface BaseEmbeddingModel.
        """
        raise NotImplementedError(
            "Qwen3 is a text-only model and does not support image encoding. "
            "Use SigLIPPytorch for image encoding instead."
        )

    def get_embedding_dim(self) -> int:
        """Return embedding dimension of the Qwen3 model"""
        return self.model.config.hidden_size
