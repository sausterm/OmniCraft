"""
Acrylic Medium - Bob Ross style painting instructions.

Generates step-by-step acrylic painting guides with:
- Warm, encouraging tone
- Specific brush and technique instructions
- Layer-by-layer progression from background to highlights
- Color mixing guidance
- Granular substep breakdown by color and canvas region
"""

from .medium import AcrylicMedium
from .constants import BrushType, StrokeMotion, PAINT_NAMES, ENCOURAGEMENTS

__all__ = [
    'AcrylicMedium',
    'BrushType',
    'StrokeMotion',
    'PAINT_NAMES',
    'ENCOURAGEMENTS',
]
