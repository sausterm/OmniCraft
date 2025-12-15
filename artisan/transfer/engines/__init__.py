"""
Collection of style transfer engine implementations
"""

from .britto_engine import BrittoEngine
from .controlnet_engine import ControlNetEngine

# Optional engines (require API keys):
try:
    from .replicate_engine import ReplicateEngine
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    ReplicateEngine = None

# Future imports:
# from .stability_engine import StabilityEngine
# from .openai_engine import OpenAIEngine

__all__ = [
    'BrittoEngine',
    'ControlNetEngine',
    'ReplicateEngine',
]
