import logging
import warnings
from typing import List, Union

import numpy as np
import torch
import transformers
from config import Config
from embedding.base_model import BaseEmbeddingModel
from transformers import AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Fixed output dimension of the BGE-M3 model.
_BGE_M3_EMBEDDING_DIM: int = 1024


class BGEM3Pytorch(BaseEmbeddingModel):
    """PyTorch wrapper for the BGE-M3 text embedding model.

    Loads the model in ``float16`` with low CPU memory usage and moves it
    to the device specified in ``Config.DEVICE``. Inference runs under
    ``torch.no_grad()`` to avoid unnecessary gradient computation.

    Text is tokenized with padding, truncation, and a configurable max
    token length (``Config.MAX_TOKEN_LENGTH``). Embeddings are produced
    via mean pooling over the last hidden state.
    """

    def __init__(self) -> None:
        """Load the BGE-M3 model and tokenizer onto the configured device."""
        self.device = Config.DEVICE

        logger.info("Loading BGE-M3 model onto device '%s'...", self.device)

        self.model = (
            AutoModel.from_pretrained(
                Config.TEXT_MODEL_NAME,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )
            .to(Config.DEVICE)
            .eval()
        )
        self.tokenizer = AutoTokenizer.from_pretrained(Config.TEXT_MODEL_NAME)

        logger.info(
            "BGE-M3 model loaded successfully on '%s'.", self.device
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
    def encode_text(
            self,
            texts: Union[List[str], str],
            normalize: bool = True,
    ) -> np.ndarray:
        """Encode one or more text strings into BGE-M3 embeddings.

        Applies mean pooling over the last hidden state and optionally
        L2-normalises the result. BGE-M3 works best without any special
        instruction prefix — text is passed as-is.

        Args:
            texts: A single string or a list of strings to encode.
            normalize: If ``True``, applies L2 normalisation to the output.

        Returns:
            NumPy array of shape ``(n_texts, 1024)`` in ``float16``.
        """
        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=Config.MAX_TOKEN_LENGTH,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        # Mean pooling over the last hidden state.
        embeddings = outputs.last_hidden_state.mean(dim=1)

        if normalize:
            embeddings = self._normalize_embeddings(embeddings)

        return embeddings.cpu().numpy()

    def encode_images(self, images, batch_size: int = 32) -> torch.Tensor:
        """Not supported — BGE-M3 is a text-only model.

        This method exists solely to satisfy the ``BaseEmbeddingModel``
        interface contract. Use ``SigLIPPytorch`` for image encoding.

        Args:
            images: Unused.
            batch_size: Unused.

        Raises:
            NotImplementedError: Always, unconditionally.
        """
        raise NotImplementedError(
            "BGE-M3 is a text-only model and does not support image encoding. "
            "Use SigLIPPytorch for image encoding instead."
        )

    def get_embedding_dim(self) -> int:
        """Return the fixed output embedding dimension of BGE-M3.

        Returns:
            ``1024`` — the fixed output dimension of BGE-M3.
        """
        return _BGE_M3_EMBEDDING_DIM
