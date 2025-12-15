"""
Artisan - AI-Powered Paint-by-Numbers Generation System

A comprehensive system for converting images into step-by-step painting guides
with semantic understanding, context-aware layering, and Bob Ross methodology.

Quick Start (YOLO + Bob Ross - Recommended):
    >>> from artisan.paint import YOLOBobRossPaint
    >>>
    >>> # Simple usage
    >>> painter = YOLOBobRossPaint("dog.jpg")
    >>> painter.process()
    >>> painter.save_all("./output")

Features:
    - YOLO semantic segmentation (dogs, cats, people, etc.)
    - Scene context analysis (time of day, weather, lighting, mood)
    - Subject-specific painting strategies (fur, foliage, skin, etc.)
    - Spatial back-to-front layering (not just luminosity)
    - Three output views per step: cumulative, context, isolated
    - Bob Ross style instructions with brush/stroke suggestions

Package Structure:
    artisan/
    ├── paint/          - Paint-by-numbers generators (YOLOBobRossPaint)
    ├── vision/         - Scene analysis, YOLO segmentation, context detection
    ├── core/           - Color matching, paint database, shared types
    ├── mediums/        - Medium-specific implementations
    ├── transfer/       - Style transfer engines
    ├── api/            - FastAPI backend for web deployment
    └── cli/            - Command-line tools

Author: Artisan Team
Version: 6.0.0
"""

__version__ = "6.0.0"
__author__ = "Artisan Team"

# Core types and utilities
from .core.types import (
    BrushType,
    StrokeMotion,
    CanvasArea,
    MaterialType,
    TechniqueCategory,
    PaintingTechnique,
    PAINT_NAMES,
    ENCOURAGEMENTS,
)
from .core.constraints import (
    ArtConstraints,
    Medium,
    Style,
    SkillLevel,
    SubjectDomain,
    PRESETS,
)
from .core.paint_by_numbers import PaintByNumbers
from .core.color_matcher import ColorMatcher, ColorSolution, MixingRecipe, PaintMatch
from .core.paint_database import (
    PAINT_DATABASE,
    get_paint,
    get_paints_by_brand,
    get_primary_colors,
    Paint,
    PaintBrand,
    PaintType,
)

# Paint module - Primary generators
from .paint.generators.yolo_bob_ross_paint import (
    YOLOBobRossPaint,
    SemanticPaintingLayer,
    PaintingSubstep,
    process_image,
)
from .paint.generators.paint_kit_generator import PaintKitGenerator, CANVAS_SIZES, PaintKit
from .paint.generators.instruction_generator import UnifiedInstructionGenerator, InstructionLevel
from .paint.bob_ross import BobRossGenerator, PaintingStep
from .paint.optimization.budget_optimizer import (
    BudgetOptimizer,
    BudgetAnalysis,
    BudgetTier,
    MixingComplexity,
    BUDGET_TIERS,
    PAINT_SETS,
)

# Vision module - Perception and analysis
from .vision.segmentation.yolo_segmentation import YOLOSemanticSegmenter, SemanticRegion, YOLO_AVAILABLE
from .vision.analysis.scene_context import SceneContextAnalyzer, SceneContext, TimeOfDay, Weather
from .vision.analysis.scene_analyzer import SceneAnalyzer
from .vision.analysis.layering_strategies import LayeringStrategyEngine, SubjectType
from .vision.analysis.technique_analyzer import TechniqueAnalyzer, Technique, ImageAnalysis

# Mediums
from .mediums.base import MediumBase, Material, Substep, Layer

# Convenience aliases for backward compatibility
from .paint.generators.yolo_bob_ross_paint import YOLOBobRossPaint as Generator

__all__ = [
    # Version
    "__version__",
    "__author__",
    # Primary - YOLO + Bob Ross (Recommended)
    "YOLOBobRossPaint",
    "Generator",
    "SemanticPaintingLayer",
    "PaintingSubstep",
    "process_image",
    # Vision
    "YOLOSemanticSegmenter",
    "SemanticRegion",
    "YOLO_AVAILABLE",
    "SceneContextAnalyzer",
    "SceneContext",
    "SceneAnalyzer",
    "LayeringStrategyEngine",
    "TechniqueAnalyzer",
    # Paint utilities
    "PaintKitGenerator",
    "UnifiedInstructionGenerator",
    "BobRossGenerator",
    "BudgetOptimizer",
    # Core
    "PaintByNumbers",
    "ColorMatcher",
    "PAINT_DATABASE",
    "BrushType",
    "StrokeMotion",
    "CanvasArea",
    # Types
    "ArtConstraints",
    "Medium",
    "Style",
    "SkillLevel",
]
