"""Painting instruction generator subpackage with granular layer/substep support."""

from .generator import BobRossGenerator
from .steps import PaintingStep, PaintingLayer, PaintingSubstep
from artisan.core.types import BrushType, StrokeMotion, CanvasArea, PAINT_NAMES, ENCOURAGEMENTS

__all__ = [
    "BobRossGenerator",
    "PaintingStep",
    "PaintingLayer",
    "PaintingSubstep",
    "CanvasArea",
    "BrushType",
    "StrokeMotion",
    "PAINT_NAMES",
    "ENCOURAGEMENTS",
]
