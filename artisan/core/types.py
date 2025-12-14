"""
Shared type definitions for the Artisan paint-by-numbers system.

This module consolidates common enums and dataclasses that were previously
duplicated across multiple modules. All shared types should be imported from here.

Consolidated from:
- mediums/acrylic/constants.py
- generators/bob_ross/constants.py
- generators/bob_ross/steps.py
- mediums/base.py
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


# =============================================================================
# BRUSH AND STROKE TYPES
# =============================================================================

class BrushType(Enum):
    """Bob Ross / acrylic painting brush types."""
    TWO_INCH = "2-inch brush"
    ONE_INCH = "1-inch brush"
    FAN_BRUSH = "fan brush"
    LINER_BRUSH = "liner brush (script)"
    FILBERT = "filbert brush"
    PALETTE_KNIFE = "palette knife"
    ROUND = "small round brush"


class StrokeMotion(Enum):
    """Brush stroke motions for painting techniques."""
    CRISS_CROSS = "criss-cross strokes"
    VERTICAL_PULL = "pull down vertically"
    HORIZONTAL_PULL = "pull horizontally"
    TAP = "tap gently"
    STIPPLE = "stipple (pounce up and down)"
    CIRCULAR = "small circular motions"
    BLEND = "blend softly back and forth"
    LOAD_AND_PULL = "load brush and pull in one stroke"
    FLICK = "flick outward from center"


# =============================================================================
# CANVAS AND REGION TYPES
# =============================================================================

@dataclass
class CanvasArea:
    """
    A specific region of the canvas/workspace.

    Represents WHERE a material should be applied. Used across painting
    generators and medium implementations.
    """
    name: str                           # "upper-left", "center", "sky region"
    bounds: Tuple[float, float, float, float]  # (y1, y2, x1, x2) as fractions 0-1
    mask: Optional[np.ndarray] = None   # Boolean mask of exact pixels
    coverage_percent: float = 0.0       # What % of total canvas this covers
    centroid: Tuple[float, float] = (0.5, 0.5)  # (y, x) center as fractions
    semantic_label: str = ""            # "sky", "mountain", "water" (if available)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict (excludes mask)."""
        return {
            'name': self.name,
            'bounds': list(self.bounds),
            'coverage_percent': self.coverage_percent,
            'centroid': list(self.centroid),
            'semantic_label': self.semantic_label,
        }


# =============================================================================
# TECHNIQUE CATEGORIES
# =============================================================================

class TechniqueCategory(Enum):
    """High-level technique categories that apply across mediums."""
    FOUNDATION = "foundation"          # Base layers, underpainting
    BLOCKING = "blocking"              # Large shape establishment
    LAYERING = "layering"              # Building up layers
    BLENDING = "blending"              # Smooth transitions
    DETAILING = "detailing"            # Fine detail work
    TEXTURING = "texturing"            # Surface texture creation
    HIGHLIGHTING = "highlighting"      # Light/bright areas
    SHADOWING = "shadowing"            # Dark/shadow areas
    FINISHING = "finishing"            # Final touches


class PaintingTechnique(Enum):
    """Specific painting techniques that can be recommended."""
    STANDARD_FILL = "standard_fill"
    GLAZING = "glazing"
    WET_ON_WET = "wet_on_wet"
    DRY_BRUSH = "dry_brush"
    LAYERED_BUILDUP = "layered_buildup"
    GRADIENT_BLEND = "gradient_blend"


# =============================================================================
# MATERIAL TYPES
# =============================================================================

class MaterialType(Enum):
    """Types of materials used in different mediums."""
    PAINT = "paint"
    THREAD = "thread"
    BRICK = "brick"
    BEAD = "bead"
    YARN = "yarn"
    PENCIL = "pencil"
    MARKER = "marker"
    FABRIC = "fabric"
    PAPER = "paper"


# =============================================================================
# BOB ROSS CONSTANTS
# =============================================================================

# Bob Ross paint color names (mapped from common colors)
PAINT_NAMES = {
    'black': 'Midnight Black',
    'white': 'Titanium White',
    'blue': 'Prussian Blue',
    'light_blue': 'Pthalo Blue',
    'green': 'Sap Green',
    'light_green': 'Cadmium Yellow + Sap Green mix',
    'yellow': 'Cadmium Yellow',
    'orange': 'Cadmium Yellow + Alizarin Crimson',
    'red': 'Alizarin Crimson',
    'magenta': 'Alizarin Crimson + a touch of Pthalo Blue',
    'purple': 'Alizarin Crimson + Prussian Blue',
    'brown': 'Van Dyke Brown',
    'pink': 'Titanium White + Alizarin Crimson',
    'dark_sienna': 'Dark Sienna',
}

# Encouragements Bob Ross would say
ENCOURAGEMENTS = [
    "There are no mistakes, only happy accidents.",
    "You can do this. Anyone can paint.",
    "Let's get a little crazy here.",
    "This is your world. You can do anything you want.",
    "We don't make mistakes, we just have happy accidents.",
    "Take your time. There's no pressure here.",
    "Just let it happen. Let your brush do the work.",
    "Isn't that fantastic? Look at that.",
    "That's what makes painting so wonderful.",
    "Now then, let's get a little braver.",
    "See how easy that was?",
    "We're just having fun here.",
    "Let your imagination take over.",
    "Don't be afraid of the canvas.",
    "Every day is a good day when you paint.",
]
