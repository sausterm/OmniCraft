"""Painting instruction generator subpackage with granular layer/substep support."""

from .generator import BobRossGenerator
from .steps import PaintingStep, PaintingLayer, PaintingSubstep, CanvasArea
from .constants import BrushType, StrokeMotion, PAINT_NAMES, ENCOURAGEMENTS

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
