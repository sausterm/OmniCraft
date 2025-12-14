"""Core processing modules for Artisan."""

from .paint_by_numbers import PaintByNumbers
from .color_matcher import ColorMatcher, ColorSolution, MixingRecipe, PaintMatch
from .paint_database import PAINT_DATABASE, get_paint, get_paints_by_brand, get_primary_colors
from .regions import RegionDetector, RegionDetectionConfig
from .segmenter import ImageSegmenter, LayerDefinition, LayerStrategy, find_colors_in_layer

# Shared types (canonical location)
from .types import (
    BrushType,
    StrokeMotion,
    CanvasArea,
    TechniqueCategory,
    PaintingTechnique,
    MaterialType,
    PAINT_NAMES,
    ENCOURAGEMENTS,
)

__all__ = [
    # Core classes
    "PaintByNumbers",
    "ColorMatcher",
    "ColorSolution",
    "MixingRecipe",
    "PaintMatch",
    "PAINT_DATABASE",
    "get_paint",
    "get_paints_by_brand",
    "get_primary_colors",
    "RegionDetector",
    "RegionDetectionConfig",
    "ImageSegmenter",
    "LayerDefinition",
    "LayerStrategy",
    "find_colors_in_layer",
    # Shared types
    "BrushType",
    "StrokeMotion",
    "CanvasArea",
    "TechniqueCategory",
    "PaintingTechnique",
    "MaterialType",
    "PAINT_NAMES",
    "ENCOURAGEMENTS",
]
