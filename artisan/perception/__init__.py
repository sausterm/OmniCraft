"""
Perception module - Understanding images through semantic segmentation.

This module provides image analysis capabilities:
- YOLO semantic segmentation for object detection
- Scene context analysis (time of day, weather, lighting, mood)
- Subject-specific layering strategies
- Depth-based layer ordering for back-to-front painting
- Scene graph construction
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

# New YOLO + Context-aware components
from .yolo_segmentation import (
    YOLOSemanticSegmenter,
    SemanticRegion,
    segment_with_yolo,
    YOLO_AVAILABLE,
)
from .scene_context import (
    SceneContextAnalyzer,
    SceneContext,
    TimeOfDay,
    Weather,
    Setting,
    LightingType,
    Mood,
    analyze_scene,
)
from .layering_strategies import (
    LayeringStrategyEngine,
    LayerSubstep,
    SubjectType,
    classify_subject,
)

__all__ = [
    # Primary - YOLO + Context-aware (recommended)
    "YOLOSemanticSegmenter",
    "SemanticRegion",
    "segment_with_yolo",
    "YOLO_AVAILABLE",
    "SceneContextAnalyzer",
    "SceneContext",
    "TimeOfDay",
    "Weather",
    "Setting",
    "LightingType",
    "Mood",
    "analyze_scene",
    "LayeringStrategyEngine",
    "LayerSubstep",
    "SubjectType",
    "classify_subject",
    # Legacy components
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
