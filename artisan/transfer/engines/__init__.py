"""
Collection of style transfer engine implementations
"""

# Optional engines - import conditionally to avoid breaking if dependencies missing

# Replicate API (cloud-based, fast)
try:
    from .replicate_engine import ReplicateEngine
    REPLICATE_AVAILABLE = True
except ImportError:
    REPLICATE_AVAILABLE = False
    ReplicateEngine = None

# ControlNet (local, requires torch/diffusers)
try:
    from .controlnet_engine import ControlNetEngine
    CONTROLNET_AVAILABLE = True
except ImportError:
    CONTROLNET_AVAILABLE = False
    ControlNetEngine = None

# Britto engine (legacy, requires britto_style_transfer package)
try:
    from .britto_engine import BrittoEngine
    BRITTO_AVAILABLE = True
except ImportError:
    BRITTO_AVAILABLE = False
    BrittoEngine = None

__all__ = [
    'ReplicateEngine',
    'ControlNetEngine',
    'BrittoEngine',
    'REPLICATE_AVAILABLE',
    'CONTROLNET_AVAILABLE',
    'BRITTO_AVAILABLE',
]
