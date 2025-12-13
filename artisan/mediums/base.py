"""
MediumBase - Abstract interface for all art mediums.

Defines the core abstractions that all mediums must implement:
- Materials (paints, threads, bricks, etc.)
- Canvas regions (areas where materials are applied)
- Substeps (atomic actions: apply material X to area Y with technique Z)
- Layers (logical groupings of substeps in execution order)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Optional, Dict, Any
import numpy as np


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


@dataclass
class Material:
    """
    A material used in the creation process.

    Medium-agnostic: could be paint, thread, LEGO brick, etc.
    """
    name: str                           # "Titanium White", "DMC 310", "LEGO Red 2x4"
    material_type: MaterialType         # paint, thread, brick, etc.
    color_rgb: Tuple[int, int, int]    # RGB color value
    color_hex: str = ""                # Hex color code
    identifier: str = ""               # Product code, SKU, DMC number, etc.
    brand: str = ""                    # "Bob Ross", "DMC", "LEGO"
    quantity: str = ""                 # "2.5 oz tube", "1 skein", "x12 bricks"
    price_usd: float = 0.0             # Unit price
    mixing_recipe: Optional[str] = None  # "2 parts white + 1 part blue"
    metadata: Dict[str, Any] = field(default_factory=dict)  # Medium-specific data

    def __post_init__(self):
        """Generate hex code if not provided."""
        if not self.color_hex:
            self.color_hex = '#{:02x}{:02x}{:02x}'.format(*self.color_rgb)


@dataclass
class CanvasArea:
    """
    A specific region of the canvas/workspace.

    Represents WHERE a material should be applied.
    """
    name: str                           # "upper-left", "center", "sky region"
    bounds: Tuple[float, float, float, float]  # (y1, y2, x1, x2) as fractions 0-1
    mask: Optional[np.ndarray] = None   # Boolean mask of exact pixels/cells
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


@dataclass
class Substep:
    """
    A single atomic creation action.

    Represents: Apply material X to area Y using technique Z.
    This is the fundamental unit of instruction.
    """
    substep_id: str                     # "1.3" (layer 1, substep 3)
    material: Material                  # What to use
    area: CanvasArea                    # Where to apply it
    technique: str                      # "dry brush", "glazing", "cross-stitch"
    instruction: str                    # Human-readable instruction
    tool: str = ""                      # "1-inch brush", "size 24 needle"
    motion: str = ""                    # "horizontal strokes", "diagonal stitches"
    duration_hint: str = ""             # "2-3 minutes", "30 seconds"
    order_priority: int = 0             # For sorting (lower = earlier)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'id': self.substep_id,
            'material': {
                'name': self.material.name,
                'type': self.material.material_type.value,
                'color_rgb': list(self.material.color_rgb),
                'color_hex': self.material.color_hex,
                'identifier': self.material.identifier,
            },
            'area': self.area.to_dict(),
            'technique': self.technique,
            'tool': self.tool,
            'motion': self.motion,
            'instruction': self.instruction,
            'duration': self.duration_hint,
            'order': self.order_priority,
        }


@dataclass
class Layer:
    """
    A logical grouping of substeps that should be executed together.

    Represents a complete phase of creation (e.g., "Background layer",
    "Detail layer", "Highlight layer").
    """
    layer_number: int
    name: str                           # "Dark Background", "Aurora Glow"
    description: str                    # Layer-level overview
    substeps: List[Substep] = field(default_factory=list)
    wait_time: str = ""                 # "Let dry 10 minutes", "No wait needed"
    technique_tip: str = ""             # Medium-specific advice
    encouragement: str = ""             # Motivational message

    @property
    def total_substeps(self) -> int:
        """Number of substeps in this layer."""
        return len(self.substeps)

    @property
    def estimated_duration(self) -> str:
        """Rough time estimate based on substep count."""
        if self.total_substeps <= 2:
            return "2-4 minutes"
        elif self.total_substeps <= 5:
            return "5-10 minutes"
        elif self.total_substeps <= 10:
            return "10-20 minutes"
        else:
            return "20-30 minutes"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            'layer_number': self.layer_number,
            'name': self.name,
            'description': self.description,
            'substeps': [s.to_dict() for s in self.substeps],
            'total_substeps': self.total_substeps,
            'estimated_duration': self.estimated_duration,
            'wait_time': self.wait_time,
            'technique_tip': self.technique_tip,
            'encouragement': self.encouragement,
        }


class MediumBase(ABC):
    """
    Abstract base class for all art mediums.

    Each medium (acrylic, watercolor, LEGO, cross-stitch) must implement:
    1. get_materials() - What materials are needed?
    2. plan_layers() - What order should creation happen?
    3. generate_substeps() - Break layers into atomic actions
    """

    def __init__(self, image_path: str, **kwargs):
        """
        Initialize medium with source image.

        Args:
            image_path: Path to source image
            **kwargs: Medium-specific configuration
        """
        self.image_path = image_path
        self.config = kwargs
        self.layers: List[Layer] = []
        self.materials: List[Material] = []

    @abstractmethod
    def get_materials(self) -> List[Material]:
        """
        Analyze image and determine required materials.

        Returns:
            List of Material objects needed for this creation
        """
        pass

    @abstractmethod
    def plan_layers(self) -> List[Layer]:
        """
        Plan the layer structure for creation.

        Determines the optimal order of operations (e.g., background to
        foreground, dark to light, etc.)

        Returns:
            List of Layer objects (substeps may be empty at this stage)
        """
        pass

    @abstractmethod
    def generate_substeps(self, layer: Layer) -> List[Substep]:
        """
        Break a layer into granular substeps.

        Each substep is an atomic action: apply one material to one area
        with one technique.

        Args:
            layer: The layer to break down

        Returns:
            List of Substep objects in execution order
        """
        pass

    def generate_full_guide(self) -> List[Layer]:
        """
        Generate complete guide with all layers and substeps.

        This is the main entry point for guide generation.

        Returns:
            List of Layer objects, each with populated substeps
        """
        # Get materials
        self.materials = self.get_materials()

        # Plan layers
        self.layers = self.plan_layers()

        # Generate substeps for each layer
        for layer in self.layers:
            layer.substeps = self.generate_substeps(layer)

        return self.layers

    def export_json(self) -> Dict[str, Any]:
        """
        Export guide to JSON-serializable format.

        Returns:
            Dictionary with complete guide data
        """
        if not self.layers:
            self.generate_full_guide()

        return {
            'image': self.image_path,
            'medium': self.__class__.__name__,
            'total_layers': len(self.layers),
            'total_substeps': sum(len(layer.substeps) for layer in self.layers),
            'materials': [
                {
                    'name': m.name,
                    'type': m.material_type.value,
                    'color_hex': m.color_hex,
                    'identifier': m.identifier,
                    'quantity': m.quantity,
                }
                for m in self.materials
            ],
            'layers': [layer.to_dict() for layer in self.layers],
        }
