"""
Artisan - Semantic-Aware Art Construction System

A comprehensive system for converting images into step-by-step art creation guides.
Uses semantic understanding (SAM-based segmentation) and art principles to generate
intelligent, pedagogically-sound lessons for any medium.

Package Structure:
    artisan/
    ├── core/           - Core data structures, constraints, art principles
    ├── perception/     - Semantic segmentation, scene understanding
    ├── planning/       - Lesson plan generation
    ├── renderers/      - Medium-specific instruction rendering
    ├── generators/     - Legacy kit and instruction generation
    ├── analysis/       - Technique analysis and visualization
    ├── optimization/   - Budget optimization
    ├── mediums/        - Medium-specific implementations
    └── cli/            - Command-line tools

New Architecture (v4.0):
    - Artisan: Main orchestrator for semantic-aware lesson generation
    - ArtConstraints: User-defined parameters (image, medium, style, skill)
    - SemanticSegmenter: SAM-based image understanding
    - LessonPlanGenerator: Creates pedagogically-sound lesson plans
    - InstructionRenderer: Medium-specific detailed instructions

Quick Start (New):
    >>> from artisan import Artisan, ArtConstraints
    >>>
    >>> # Create artisan and generate lesson
    >>> artisan = Artisan()
    >>> constraints = ArtConstraints.from_simple("dog.jpg", medium="acrylic")
    >>> lesson = artisan.create_lesson(constraints, output_dir="./output")
    >>>
    >>> # Or use convenience function
    >>> from artisan import create_lesson
    >>> lesson = create_lesson("portrait.jpg", medium="oil", style="realism")

Legacy Quick Start:
    >>> from artisan import PaintByNumbers, PaintKitGenerator
    >>>
    >>> # Simple paint-by-numbers
    >>> pbn = PaintByNumbers('image.jpg', n_colors=15)
    >>> pbn.process_all('output/')

CLI Usage:
    # New lesson generator
    python -m artisan.cli.lesson dog.jpg --medium acrylic --style painterly

    # Legacy generator
    python artisan/cli/generate.py aurora_maria kit 15 16x20 75

Author: Artisan Team
Version: 4.0.0
"""

__version__ = "4.0.0"
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

# Generators
from .generators.paint_kit_generator import PaintKitGenerator, CANVAS_SIZES, PaintKit
from .generators.instruction_generator import UnifiedInstructionGenerator, InstructionLevel
from .generators.bob_ross import BobRossGenerator, PaintingStep, BrushType, StrokeMotion
from .generators.smart_paint_by_numbers import SmartPaintByNumbers

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
    # New Architecture
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
