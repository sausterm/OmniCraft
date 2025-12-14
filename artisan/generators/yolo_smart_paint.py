"""
DEPRECATED: YOLOSmartPaintByNumbers has been superseded by YOLOBobRossPaint.

This module is kept for backward compatibility. New code should use:
    from artisan.generators.yolo_bob_ross_paint import YOLOBobRossPaint
"""

import warnings

warnings.warn(
    "YOLOSmartPaintByNumbers is deprecated. "
    "Use 'from artisan.generators.yolo_bob_ross_paint import YOLOBobRossPaint' instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility
from .deprecated._yolo_smart_paint import YOLOSmartPaintByNumbers

__all__ = ['YOLOSmartPaintByNumbers']
