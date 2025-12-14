"""
Artisan Generators - Paint-by-numbers and instruction generation modules.

Main Components:
- YOLOBobRossPaint: Context-aware paint-by-numbers with YOLO semantic segmentation
- PaintKitGenerator: Generate paint shopping lists matched to real brands
- UnifiedInstructionGenerator: Generate step-by-step painting instructions
"""

from .instruction_generator import UnifiedInstructionGenerator, InstructionLevel
from .paint_kit_generator import PaintKitGenerator, PaintKit, CANVAS_SIZES
from .smart_paint_by_numbers import SmartPaintByNumbers
from .bob_ross import BobRossGenerator

# New YOLO + Bob Ross context-aware system
from .yolo_bob_ross_paint import (
    YOLOBobRossPaint,
    SemanticPaintingLayer,
    PaintingSubstep,
    process_image,
)

__all__ = [
    # Primary - YOLO Bob Ross (recommended)
    "YOLOBobRossPaint",
    "SemanticPaintingLayer",
    "PaintingSubstep",
    "process_image",
    # Legacy generators
    "UnifiedInstructionGenerator",
    "InstructionLevel",
    "PaintKitGenerator",
    "PaintKit",
    "CANVAS_SIZES",
    "SmartPaintByNumbers",
    "BobRossGenerator",
]
