"""
Perception module - Understanding images through semantic segmentation.

This module provides image analysis capabilities:
- Semantic segmentation (SAM-based when available)
- Scene graph construction
- Depth-based layer ordering for back-to-front painting
- Material/texture analysis
- Subject detection
"""

from .semantic import (
    SemanticSegmenter,
    SegmentationResult,
    Segment,
)
from .scene_builder import SceneGraphBuilder
from .subject_detector import SubjectDetector
from .depth_ordering import (
    DepthAnalyzer,
    DepthZone,
    SpatialLayer,
    create_depth_ordered_layers,
)

__all__ = [
    "SemanticSegmenter",
    "SegmentationResult",
    "Segment",
    "SceneGraphBuilder",
    "SubjectDetector",
    "DepthAnalyzer",
    "DepthZone",
    "SpatialLayer",
    "create_depth_ordered_layers",
]
