"""OTF-CBM: Optimal-Transport Flow Concept Bottleneck Models."""

from .config import load_config
from .concepts import ConceptBank

__all__ = ["ConceptBank", "load_config"]
__version__ = "0.1.0"
