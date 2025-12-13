"""Painting step data structures for granular layer/substep instructions."""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

from .constants import BrushType, StrokeMotion


@dataclass
class PaintingStep:
    """A single step in the painting process (legacy, for backward compatibility)"""
    step_number: int
    title: str
    instruction: str
    brush: BrushType
    motion: StrokeMotion
    colors: List[str]  # Color names
    color_rgbs: List[Tuple[int, int, int]]
    canvas_region: str  # "entire canvas", "top third", "bottom", etc.
    region_mask: Optional[np.ndarray]
    technique_tip: str
    encouragement: str
    duration_hint: str  # "30 seconds", "2-3 minutes", etc.


@dataclass
class CanvasArea:
    """A specific region of the canvas"""
    name: str                    # Human-readable: "upper-left", "center", etc.
    bounds: Tuple[float, float, float, float]  # (y1, y2, x1, x2) as fractions 0-1
    mask: Optional[np.ndarray] = None   # Boolean mask of exact pixels
    coverage_percent: float = 0.0       # What % of total canvas this covers
    centroid: Tuple[float, float] = (0.5, 0.5)  # (y, x) center point as fractions


@dataclass
class PaintingSubstep:
    """A single atomic painting action - one color, one area, one technique"""
    substep_id: str              # e.g., "2.3" (layer 2, substep 3)
    color_name: str              # Single color: "Sap Green"
    color_rgb: Tuple[int, int, int]
    brush: BrushType
    motion: StrokeMotion
    technique: str               # "glazing", "dry brush", "base coat", "blend"
    area: CanvasArea             # Specific canvas region
    instruction: str             # Short, specific instruction
    duration_hint: str           # "30 seconds", "1-2 minutes"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            "id": self.substep_id,
            "color": self.color_name,
            "color_rgb": list(self.color_rgb),
            "brush": self.brush.value,
            "motion": self.motion.value,
            "technique": self.technique,
            "canvas_area": self.area.name,
            "canvas_bounds": list(self.area.bounds),
            "coverage_percent": self.area.coverage_percent,
            "instruction": self.instruction,
            "duration": self.duration_hint,
        }


@dataclass
class PaintingLayer:
    """A logical painting layer containing multiple substeps"""
    layer_number: int
    name: str                    # "Background", "Aurora Base", "Highlights"
    overview: str                # Layer-level description
    substeps: List[PaintingSubstep] = field(default_factory=list)
    dry_time: str = ""           # "Let dry 5-10 minutes"
    technique_tip: str = ""
    encouragement: str = ""

    @property
    def total_duration(self) -> str:
        """Estimate total duration for this layer"""
        # Simple heuristic based on substep count
        if len(self.substeps) <= 2:
            return "2-4 minutes"
        elif len(self.substeps) <= 5:
            return "5-10 minutes"
        else:
            return "10-15 minutes"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict"""
        return {
            "layer_number": self.layer_number,
            "name": self.name,
            "overview": self.overview,
            "substeps": [s.to_dict() for s in self.substeps],
            "dry_time": self.dry_time,
            "technique_tip": self.technique_tip,
            "encouragement": self.encouragement,
            "total_duration": self.total_duration,
        }
