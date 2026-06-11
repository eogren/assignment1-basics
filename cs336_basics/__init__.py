import importlib.metadata

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

from .activations import Swiglu
from .attention import MultiheadSelfAttention
from .embedding import Embedding, RoPE
from .linear import Linear
from .norms import RMSNorm
from .tokenizer import Tokenizer

__all__ = ["Embedding", "Linear", "MultiheadSelfAttention", "RMSNorm", "RoPE", "Swiglu", "Tokenizer"]
