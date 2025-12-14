"""
Deprecated generators - kept for backward compatibility.

These generators have been superseded by YOLOBobRossPaint which combines
YOLO semantic segmentation with Bob Ross-style painting guidance.

RECOMMENDED: Use generators.yolo_bob_ross_paint.YOLOBobRossPaint instead.
"""

from ._smart_paint_by_numbers import SmartPaintByNumbers
from ._semantic_paint_by_numbers import SemanticPaintByNumbers
from ._yolo_smart_paint import YOLOSmartPaintByNumbers

__all__ = [
    'SmartPaintByNumbers',
    'SemanticPaintByNumbers',
    'YOLOSmartPaintByNumbers',
]
