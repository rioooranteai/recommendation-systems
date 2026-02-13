import logging
import warnings
from typing import List, Union

import numpy as np
import torch
import transformers
from PIL import Image
from config import Config
from transformers import AutoModel, AutoTokenizer

from .base_model import BaseEmbeddingModel

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
