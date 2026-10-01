from abc import ABC, abstractmethod
from typing import Sequence

import numpy as np

class Embedder(ABC):
    @abstractmethod
    def embed(self,text:str)->np.ndarray:
        pass
    
    def embed_many(self,texts:Sequence[str])->np.ndarray:
        return np.array([self.embed(text) for text in texts])