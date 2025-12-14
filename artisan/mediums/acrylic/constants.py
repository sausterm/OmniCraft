"""
Constants for acrylic medium.

DEPRECATED: Import from artisan.core.types instead.
This module re-exports types for backward compatibility.
"""

import warnings

# Re-export from canonical location
from ...core.types import (
    BrushType,
    StrokeMotion,
    PAINT_NAMES,
    ENCOURAGEMENTS,
)

# Emit deprecation warning on import
warnings.warn(
    "Importing from mediums.acrylic.constants is deprecated. "
    "Use 'from artisan.core.types import BrushType, StrokeMotion, PAINT_NAMES, ENCOURAGEMENTS' instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ['BrushType', 'StrokeMotion', 'PAINT_NAMES', 'ENCOURAGEMENTS']
