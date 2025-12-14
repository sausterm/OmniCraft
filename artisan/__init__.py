"""
Artisan - AI-Powered Paint-by-Numbers Generation System

A comprehensive system for converting images into step-by-step painting guides
with semantic understanding, context-aware layering, and Bob Ross methodology.

Quick Start (YOLO + Bob Ross - Recommended):
    >>> from artisan.generators import YOLOBobRossPaint, process_image
    >>>
    >>> # Simple usage
    >>> process_image("dog.jpg", output_dir="./output")
    >>>
    >>> # Advanced usage
    >>> painter = YOLOBobRossPaint("dog.jpg", model_size="m", conf_threshold=0.2)
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
    ├── generators/     - Paint-by-numbers generators (YOLOBobRossPaint)
    ├── perception/     - Scene analysis, YOLO segmentation, context detection
    ├── core/           - Color matching, paint database, constraints
    ├── api/            - FastAPI backend for web deployment
    └── cli/            - Command-line tools

Author: Artisan Team
Version: 5.0.0
"""

__version__ = "5.0.0"
__author__ = "Artisan Team"

# New Architecture - Semantic-Aware Art Construction
from .artisan import Artisan, create_lesson
from .core.constraints import (
    ArtConstraints,
    Medium,
    Style,
    SkillLevel,
    SubjectDomain,
    PRESETS,
)
from .core.scene_graph import SceneGraph, Entity, EntityType
from .core.art_principles import ArtPrinciplesEngine, ConstructionPhilosophy
from .planning.lesson_plan import LessonPlan, LessonPlanGenerator
from .perception.semantic import SemanticSegmenter, SegmentationResult
from .perception.scene_builder import SceneGraphBuilder
from .perception.subject_detector import SubjectDetector, SubjectAnalysis

# Legacy Core components
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

# Generators - Primary (YOLO + Bob Ross)
from .generators.yolo_bob_ross_paint import (
    YOLOBobRossPaint,
    SemanticPaintingLayer,
    PaintingSubstep,
    process_image,
)

# Generators - Legacy
from .generators.paint_kit_generator import PaintKitGenerator, CANVAS_SIZES, PaintKit
from .generators.instruction_generator import UnifiedInstructionGenerator, InstructionLevel
from .generators.bob_ross import BobRossGenerator, PaintingStep, BrushType, StrokeMotion
from .generators.smart_paint_by_numbers import SmartPaintByNumbers

# Perception - YOLO + Context
from .perception.yolo_segmentation import YOLOSemanticSegmenter, SemanticRegion, YOLO_AVAILABLE
from .perception.scene_context import SceneContextAnalyzer, SceneContext, TimeOfDay, Weather, analyze_scene
from .perception.layering_strategies import LayeringStrategyEngine, SubjectType

# Analysis
from .analysis.technique_analyzer import TechniqueAnalyzer, Technique, ImageAnalysis
from .analysis.technique_visualizer import TechniqueVisualizer

# Optimization
from .optimization.budget_optimizer import (
    BudgetOptimizer,
    BudgetAnalysis,
    BudgetTier,
    MixingComplexity,
    BUDGET_TIERS,
    PAINT_SETS,
)

# Mediums
from .mediums import (
    MediumBase,
    Material,
    CanvasArea,
    Substep,
    Layer,
    MaterialType,
)
from .mediums.acrylic import AcrylicMedium, BrushType, StrokeMotion

__all__ = [
    # Primary - YOLO + Bob Ross (Recommended)
    "YOLOBobRossPaint",
    "SemanticPaintingLayer",
    "PaintingSubstep",
    "process_image",
    "YOLOSemanticSegmenter",
    "SemanticRegion",
    "YOLO_AVAILABLE",
    "SceneContextAnalyzer",
    "SceneContext",
    "TimeOfDay",
    "Weather",
    "analyze_scene",
    "LayeringStrategyEngine",
    "SubjectType",
    # Legacy Architecture
    "Artisan",
    "create_lesson",
    "ArtConstraints",
    "Medium",
    "Style",
    "SkillLevel",
    "SubjectDomain",
    "PRESETS",
    "SceneGraph",
    "Entity",
    "EntityType",
    "ArtPrinciplesEngine",
    "ConstructionPhilosophy",
    "LessonPlan",
    "LessonPlanGenerator",
    "SemanticSegmenter",
    "SegmentationResult",
    "SceneGraphBuilder",
    "SubjectDetector",
    "SubjectAnalysis",
    # Legacy Core
    "PaintByNumbers",
    "ColorMatcher",
    "ColorSolution",
    "MixingRecipe",
    "PaintMatch",
    "PAINT_DATABASE",
    "get_paint",
    "get_paints_by_brand",
    "get_primary_colors",
    "Paint",
    "PaintBrand",
    "PaintType",
    # Generators
    "PaintKitGenerator",
    "CANVAS_SIZES",
    "PaintKit",
    "UnifiedInstructionGenerator",
    "InstructionLevel",
    "BobRossGenerator",
    "PaintingStep",
    "BrushType",
    "StrokeMotion",
    "SmartPaintByNumbers",
    # Analysis
    "TechniqueAnalyzer",
    "Technique",
    "ImageAnalysis",
    "TechniqueVisualizer",
    # Optimization
    "BudgetOptimizer",
    "BudgetAnalysis",
    "BudgetTier",
    "MixingComplexity",
    "BUDGET_TIERS",
    "PAINT_SETS",
    # Mediums
    "MediumBase",
    "Material",
    "CanvasArea",
    "Substep",
    "Layer",
    "MaterialType",
    "AcrylicMedium",
    "BrushType",
    "StrokeMotion",
]
