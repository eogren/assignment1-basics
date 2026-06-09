import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from .activations import Swiglu
from .linear import Linear
from .embedding import Embedding, RoPE
from .tokenizer import Tokenizer
from .norms import RMSNorm

__all__ = ["Embedding", "Linear", "RoPE", "RMSNorm", "Swiglu", "Tokenizer"]
