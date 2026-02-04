from abc import ABC, abstractmethod
from typing import List, Union

import torch
from PIL import Image


class BaseEmbeddingModel(ABC):
    @abstractmethod
    def encode_images(self,
                      images: Union[List[Image.Image], Image.Image],
                      batch_size: int = 32) -> torch.Tensor:
        pass

    @abstractmethod
    def encode_text(self,
                    texts: Union[List[str], str]) -> torch.Tensor:
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        pass
