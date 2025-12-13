"""
Artisan Style Transfer System
A modular framework for applying various artistic styles to images
"""

from .base import StyleTransferEngine, StyleResult
from .registry import StyleRegistry, register_style, get_engine

__all__ = [
    'StyleTransferEngine',
    'StyleResult',
    'StyleRegistry',
    'register_style',
    'get_engine',
]
