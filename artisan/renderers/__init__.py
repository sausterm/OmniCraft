"""
Renderers - Medium-specific instruction rendering.

This module provides renderers that take abstract construction plans
and generate medium-specific instructions with detailed guidance.
"""

from .base import InstructionRenderer
from .acrylic_renderer import AcrylicInstructionRenderer
from .oil_renderer import OilInstructionRenderer

__all__ = [
    "InstructionRenderer",
    "AcrylicInstructionRenderer",
    "OilInstructionRenderer",
]
