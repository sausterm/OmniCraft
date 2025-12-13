"""
Artisan Mediums - Abstract interfaces for different art mediums.

Each medium (acrylic, watercolor, LEGO, cross-stitch, etc.) implements
the MediumBase interface to provide:
- Material lists (paints, threads, bricks, etc.)
- Layer planning (painting order, construction sequence)
- Granular substep generation (atomic creation actions)
"""

from .base import (
    MediumBase,
    Material,
    CanvasArea,
    Substep,
    Layer,
    MaterialType,
)

__all__ = [
    'MediumBase',
    'Material',
    'CanvasArea',
    'Substep',
    'Layer',
    'MaterialType',
]
