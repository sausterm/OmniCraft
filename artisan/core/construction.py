"""
Construction - Data structures for building artwork step by step.

This module defines how artwork is constructed, independent of the
output medium. The ConstructionPlan is then rendered by medium-specific
renderers (painting, sculpture, textile, etc.).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np


class TechniqueCategory(Enum):
    """High-level technique categories that apply across mediums."""
    FOUNDATION = "foundation"          # Base layers, underpainting, armature
    BLOCKING = "blocking"              # Large shape establishment
    LAYERING = "layering"              # Building up layers
    BLENDING = "blending"              # Smooth transitions
    DETAILING = "detailing"            # Fine detail work
    TEXTURING = "texturing"            # Surface texture creation
    HIGHLIGHTING = "highlighting"      # Light/bright areas
    SHADOWING = "shadowing"            # Dark/shadow areas
    FINISHING = "finishing"            # Final touches, varnish, cleanup


@dataclass
class Technique:
    """
    A technique for creating part of the artwork.

    Techniques are abstract - they describe what to do, not how to do it
    in a specific medium. The renderer translates these to medium-specific
    instructions.
    """
    category: TechniqueCategory
    name: str
    description: str

    # Execution parameters
    precision_required: float = 0.5    # 0 = loose/expressive, 1 = precise
    coverage_mode: str = "fill"        # fill, stroke, stipple, blend
    direction: Optional[float] = None  # Angle in degrees if directional

    # Timing hints
    relative_duration: float = 1.0     # Multiplier for time estimate
    requires_drying: bool = False      # Needs to dry before next step

    # Dependencies
    requires_techniques: List[str] = field(default_factory=list)

    # Medium-specific hints (optional)
    medium_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Tool:
    """
    An abstract tool for creating artwork.

    The actual tool depends on the medium - a "broad applicator" might be
    a 2-inch brush for painting, a large paddle for clay, or a wide roller
    for printmaking.
    """
    name: str
    category: str              # brush, knife, roller, hand, etc.
    size: str                  # small, medium, large, extra-large
    shape: Optional[str] = None  # round, flat, fan, pointed, etc.

    # Usage hints
    precision: float = 0.5     # 0 = broad strokes, 1 = fine detail
    texture_capability: float = 0.5  # 0 = smooth only, 1 = heavy texture

    # Medium-specific mapping
    medium_equivalents: Dict[str, str] = field(default_factory=dict)


@dataclass
class Material:
    """
    A material used in construction.

    Abstract material that gets mapped to specific products by the renderer.
    """
    name: str
    category: str              # paint, clay, fabric, filament, etc.
    color: Optional[Tuple[int, int, int]] = None

    # Properties
    opacity: float = 1.0
    viscosity: float = 0.5     # 0 = thin/runny, 1 = thick/heavy
    drying_time: float = 0.5   # Relative drying time

    # Mixing
    is_mixture: bool = False
    components: List[Tuple[str, float]] = field(default_factory=list)  # (material, ratio)


@dataclass
class ConstructionStep:
    """
    A single step in constructing the artwork.

    Each step represents one action the creator takes. Steps are ordered
    and may have dependencies on previous steps.
    """
    # Required fields (no defaults)
    id: str
    step_number: int
    title: str
    entity_id: str                     # Which entity this step affects
    technique: Technique               # How to do it

    # Optional fields (with defaults)
    sub_region_mask: Optional[np.ndarray] = None  # Specific part of entity
    tools: List[Tool] = field(default_factory=list)
    materials: List[Material] = field(default_factory=list)

    # Instructions (abstract - rendered to specific text by output renderer)
    instruction_abstract: str = ""      # High-level instruction
    instruction_details: List[str] = field(default_factory=list)

    # Visual reference
    reference_before: Optional[np.ndarray] = None  # What it looks like before
    reference_after: Optional[np.ndarray] = None   # What it should look like after
    reference_focus: Optional[np.ndarray] = None   # Highlighted area to work on

    # Timing
    estimated_duration_minutes: float = 5.0
    requires_previous_dry: bool = False

    # Dependencies
    depends_on: List[str] = field(default_factory=list)  # Step IDs

    # Metadata
    tips: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConstructionPhase:
    """
    A group of related construction steps.

    Phases organize steps into logical groups (e.g., "Background",
    "Main Subject", "Details").
    """
    id: str
    name: str
    description: str
    steps: List[ConstructionStep] = field(default_factory=list)
    order: int = 0

    # Phase-level properties
    allow_parallel: bool = False       # Can steps in this phase be done in any order?
    checkpoint: bool = False           # Should creator pause and assess before continuing?

    def add_step(self, step: ConstructionStep):
        """Add a step to this phase."""
        self.steps.append(step)


class ConstructionPlan:
    """
    A complete plan for constructing the artwork.

    The plan is medium-agnostic - it describes what to build and in what
    order, but not the specific techniques for a given medium. Renderers
    translate this plan into medium-specific instructions.
    """

    def __init__(self, scene_graph, target_medium: str = "painting"):
        """
        Initialize construction plan.

        Args:
            scene_graph: The SceneGraph describing the source
            target_medium: Target medium (painting, sculpture, textile, etc.)
        """
        self.scene_graph = scene_graph
        self.target_medium = target_medium
        self.phases: List[ConstructionPhase] = []
        self._step_counter = 0

    def create_phase(
        self,
        name: str,
        description: str = "",
        order: Optional[int] = None
    ) -> ConstructionPhase:
        """Create and add a new phase."""
        if order is None:
            order = len(self.phases)

        phase = ConstructionPhase(
            id=f"phase_{len(self.phases):02d}",
            name=name,
            description=description,
            order=order
        )
        self.phases.append(phase)
        return phase

    def create_step(
        self,
        phase: ConstructionPhase,
        title: str,
        entity_id: str,
        technique: Technique,
        tools: List[Tool] = None,
        materials: List[Material] = None,
        instruction: str = "",
        sub_region_mask: np.ndarray = None,
        **kwargs
    ) -> ConstructionStep:
        """Create and add a step to a phase."""
        self._step_counter += 1

        step = ConstructionStep(
            id=f"step_{self._step_counter:04d}",
            step_number=self._step_counter,
            title=title,
            entity_id=entity_id,
            technique=technique,
            tools=tools or [],
            materials=materials or [],
            instruction_abstract=instruction,
            sub_region_mask=sub_region_mask,
            **kwargs
        )

        phase.add_step(step)
        return step

    def get_all_steps(self) -> List[ConstructionStep]:
        """Get all steps in order."""
        steps = []
        for phase in sorted(self.phases, key=lambda p: p.order):
            steps.extend(phase.steps)
        return steps

    def get_total_duration(self) -> float:
        """Estimate total construction time in minutes."""
        return sum(step.estimated_duration_minutes for step in self.get_all_steps())

    def get_materials_list(self) -> List[Material]:
        """Get deduplicated list of all materials needed."""
        materials = {}
        for step in self.get_all_steps():
            for material in step.materials:
                if material.name not in materials:
                    materials[material.name] = material
        return list(materials.values())

    def get_tools_list(self) -> List[Tool]:
        """Get deduplicated list of all tools needed."""
        tools = {}
        for step in self.get_all_steps():
            for tool in step.tools:
                if tool.name not in tools:
                    tools[tool.name] = tool
        return list(tools.values())

    def to_dict(self) -> Dict:
        """Serialize plan to dictionary."""
        return {
            "target_medium": self.target_medium,
            "total_steps": len(self.get_all_steps()),
            "estimated_duration_minutes": self.get_total_duration(),
            "phases": [
                {
                    "id": phase.id,
                    "name": phase.name,
                    "description": phase.description,
                    "order": phase.order,
                    "steps": [
                        {
                            "id": step.id,
                            "step_number": step.step_number,
                            "title": step.title,
                            "entity_id": step.entity_id,
                            "technique": {
                                "category": step.technique.category.value,
                                "name": step.technique.name,
                                "precision": step.technique.precision_required
                            },
                            "instruction": step.instruction_abstract,
                            "tips": step.tips,
                            "duration_minutes": step.estimated_duration_minutes
                        }
                        for step in phase.steps
                    ]
                }
                for phase in sorted(self.phases, key=lambda p: p.order)
            ],
            "materials": [
                {"name": m.name, "category": m.category, "color": m.color}
                for m in self.get_materials_list()
            ],
            "tools": [
                {"name": t.name, "category": t.category, "size": t.size}
                for t in self.get_tools_list()
            ]
        }

    def __repr__(self) -> str:
        return (f"ConstructionPlan(medium={self.target_medium}, "
                f"phases={len(self.phases)}, steps={len(self.get_all_steps())})")


# Pre-defined techniques (can be extended)
TECHNIQUES = {
    # Foundation
    "base_coat": Technique(
        category=TechniqueCategory.FOUNDATION,
        name="Base Coat",
        description="Apply initial layer to establish base color",
        precision_required=0.2,
        coverage_mode="fill"
    ),
    "underpainting": Technique(
        category=TechniqueCategory.FOUNDATION,
        name="Underpainting",
        description="Establish values and composition with initial layer",
        precision_required=0.3,
        coverage_mode="fill"
    ),

    # Blocking
    "block_in": Technique(
        category=TechniqueCategory.BLOCKING,
        name="Block In",
        description="Establish large shapes and masses",
        precision_required=0.3,
        coverage_mode="fill"
    ),
    "silhouette": Technique(
        category=TechniqueCategory.BLOCKING,
        name="Silhouette",
        description="Define the outer shape/outline",
        precision_required=0.5,
        coverage_mode="fill"
    ),

    # Layering
    "glaze": Technique(
        category=TechniqueCategory.LAYERING,
        name="Glaze",
        description="Apply thin transparent layer over dried paint",
        precision_required=0.4,
        coverage_mode="fill",
        requires_drying=True
    ),
    "scumble": Technique(
        category=TechniqueCategory.LAYERING,
        name="Scumble",
        description="Apply thin opaque layer allowing underlying to show",
        precision_required=0.3,
        coverage_mode="fill"
    ),

    # Blending
    "wet_blend": Technique(
        category=TechniqueCategory.BLENDING,
        name="Wet Blend",
        description="Blend colors while still wet",
        precision_required=0.4,
        coverage_mode="blend"
    ),
    "gradient": Technique(
        category=TechniqueCategory.BLENDING,
        name="Gradient",
        description="Create smooth transition between colors",
        precision_required=0.5,
        coverage_mode="blend"
    ),

    # Detailing
    "fine_detail": Technique(
        category=TechniqueCategory.DETAILING,
        name="Fine Detail",
        description="Add small precise details",
        precision_required=0.9,
        coverage_mode="stroke"
    ),
    "line_work": Technique(
        category=TechniqueCategory.DETAILING,
        name="Line Work",
        description="Add defining lines and edges",
        precision_required=0.8,
        coverage_mode="stroke"
    ),

    # Texturing
    "stipple": Technique(
        category=TechniqueCategory.TEXTURING,
        name="Stipple",
        description="Create texture with dots/dabbing",
        precision_required=0.4,
        coverage_mode="stipple"
    ),
    "dry_brush": Technique(
        category=TechniqueCategory.TEXTURING,
        name="Dry Brush",
        description="Create texture with minimal paint on brush",
        precision_required=0.3,
        coverage_mode="stroke"
    ),
    "fur_strokes": Technique(
        category=TechniqueCategory.TEXTURING,
        name="Fur Strokes",
        description="Directional strokes following fur/hair pattern",
        precision_required=0.6,
        coverage_mode="stroke"
    ),

    # Highlighting
    "highlight": Technique(
        category=TechniqueCategory.HIGHLIGHTING,
        name="Highlight",
        description="Add bright highlights where light hits",
        precision_required=0.7,
        coverage_mode="stroke"
    ),
    "rim_light": Technique(
        category=TechniqueCategory.HIGHLIGHTING,
        name="Rim Light",
        description="Add edge lighting effect",
        precision_required=0.6,
        coverage_mode="stroke"
    ),

    # Shadowing
    "shadow": Technique(
        category=TechniqueCategory.SHADOWING,
        name="Shadow",
        description="Deepen shadow areas",
        precision_required=0.5,
        coverage_mode="fill"
    ),
    "cast_shadow": Technique(
        category=TechniqueCategory.SHADOWING,
        name="Cast Shadow",
        description="Add shadows cast by objects",
        precision_required=0.6,
        coverage_mode="fill"
    ),

    # Finishing
    "refine_edges": Technique(
        category=TechniqueCategory.FINISHING,
        name="Refine Edges",
        description="Clean up and sharpen edges where needed",
        precision_required=0.8,
        coverage_mode="stroke"
    ),
    "final_highlights": Technique(
        category=TechniqueCategory.FINISHING,
        name="Final Highlights",
        description="Add final bright points (eyes, reflections)",
        precision_required=0.9,
        coverage_mode="stroke"
    ),
}


# Pre-defined abstract tools
TOOLS = {
    "broad_applicator": Tool(
        name="Broad Applicator",
        category="applicator",
        size="large",
        precision=0.2,
        texture_capability=0.3,
        medium_equivalents={
            "painting": "2-inch flat brush",
            "sculpture": "large paddle tool",
            "textile": "wide roller"
        }
    ),
    "medium_applicator": Tool(
        name="Medium Applicator",
        category="applicator",
        size="medium",
        precision=0.5,
        texture_capability=0.5,
        medium_equivalents={
            "painting": "1-inch flat brush",
            "sculpture": "medium modeling tool",
            "textile": "standard brush"
        }
    ),
    "detail_tool": Tool(
        name="Detail Tool",
        category="detail",
        size="small",
        precision=0.9,
        texture_capability=0.2,
        medium_equivalents={
            "painting": "liner brush / #2 round",
            "sculpture": "fine needle tool",
            "textile": "fine point marker"
        }
    ),
    "blending_tool": Tool(
        name="Blending Tool",
        category="blending",
        size="medium",
        precision=0.4,
        texture_capability=0.1,
        medium_equivalents={
            "painting": "soft mop brush / fan brush",
            "sculpture": "smoothing tool",
            "textile": "blending stump"
        }
    ),
    "texture_tool": Tool(
        name="Texture Tool",
        category="texture",
        size="medium",
        precision=0.4,
        texture_capability=0.9,
        medium_equivalents={
            "painting": "fan brush / palette knife",
            "sculpture": "texture stamp",
            "textile": "texture roller"
        }
    ),
}
