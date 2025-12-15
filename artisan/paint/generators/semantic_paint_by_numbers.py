"""
DEPRECATED: SemanticPaintByNumbers has been superseded by YOLOBobRossPaint.

This module is kept for backward compatibility. New code should use:
    from artisan.generators.yolo_bob_ross_paint import YOLOBobRossPaint
"""

import warnings

warnings.warn(
    "SemanticPaintByNumbers is deprecated. "
    "Use 'from artisan.generators.yolo_bob_ross_paint import YOLOBobRossPaint' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
from .deprecated._semantic_paint_by_numbers import SemanticPaintByNumbers

__all__ = ['SemanticPaintByNumbers']
