"""
Constraints - User-defined parameters for art generation.

This module defines the input constraints that shape how artwork is analyzed,
planned, and instructed. Constraints include the source image, target style,
medium, skill level, and other factors.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import numpy as np


class Medium(Enum):
    """Art mediums we support."""
    ACRYLIC = "acrylic"
    OIL = "oil"
    WATERCOLOR = "watercolor"
    GOUACHE = "gouache"
    COLORED_PENCIL = "colored_pencil"
    GRAPHITE = "graphite"
    CHARCOAL = "charcoal"
    PASTEL = "pastel"
    DIGITAL = "digital"
    MIXED_MEDIA = "mixed_media"


class Style(Enum):
    """Art styles that influence technique selection."""
    REALISM = "realism"                    # Accurate representation
    IMPRESSIONISM = "impressionism"        # Light, color, movement
    EXPRESSIONISM = "expressionism"        # Emotional, bold
    PHOTOREALISM = "photorealism"          # Extremely detailed
    PAINTERLY = "painterly"                # Visible brushwork
    FLAT = "flat"                          # Minimal shading, graphic
    LOOSE = "loose"                        # Expressive, gestural
    TIGHT = "tight"                        # Controlled, precise
    CLASSICAL = "classical"                # Traditional techniques
    CONTEMPORARY = "contemporary"          # Modern approaches


class SkillLevel(Enum):
    """Creator skill level - affects instruction complexity."""
    BEGINNER = "beginner"           # New to art, needs detailed guidance
    INTERMEDIATE = "intermediate"   # Some experience, knows basics
    ADVANCED = "advanced"           # Skilled, needs less hand-holding
    EXPERT = "expert"               # Professional level


class SubjectDomain(Enum):
    """Domain-specific handling for different subjects."""
    PORTRAIT = "portrait"           # Human faces/figures
    ANIMAL = "animal"               # Pets, wildlife
    LANDSCAPE = "landscape"         # Nature, scenery
    STILL_LIFE = "still_life"       # Objects, arrangements
    ARCHITECTURE = "architecture"   # Buildings, interiors
    ABSTRACT = "abstract"           # Non-representational
    BOTANICAL = "botanical"         # Plants, flowers
    MIXED = "mixed"                 # Multiple domains


@dataclass
class BudgetConstraint:
    """Budget limitations for materials."""
    max_budget: Optional[float] = None      # In dollars
    currency: str = "USD"
    prefer_budget_friendly: bool = False    # Optimize for cost
    existing_materials: List[str] = field(default_factory=list)


@dataclass
class TimeConstraint:
    """Time limitations for the project."""
    max_hours: Optional[float] = None       # Total project time
    session_length_hours: float = 2.0       # Typical session length
    deadline: Optional[str] = None          # ISO date string


@dataclass
class SurfaceConstraint:
    """Canvas/surface specifications."""
    width_inches: float = 16.0
    height_inches: float = 20.0
    surface_type: str = "canvas"            # canvas, paper, board, etc.
    texture: str = "medium"                 # smooth, medium, rough
    pre_primed: bool = True


@dataclass
class OutputConstraint:
    """Output format preferences."""
    generate_supply_list: bool = True
    generate_color_mixing_guide: bool = True
    generate_step_images: bool = True
    generate_video_guide: bool = False
    instruction_verbosity: str = "detailed"  # minimal, moderate, detailed
    include_tips: bool = True
    include_common_mistakes: bool = True


@dataclass
class ArtConstraints:
    """
    Complete set of constraints for generating art instructions.

    This is the primary input object that captures everything the user
    wants: what to create, how to create it, and what limitations exist.
    """
    # Source image (required)
    source_image: Optional[np.ndarray] = None
    source_path: Optional[Path] = None

    # Target specifications
    medium: Medium = Medium.ACRYLIC
    style: Style = Style.PAINTERLY
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE

    # Subject domain (auto-detected if not specified)
    subject_domain: Optional[SubjectDomain] = None

    # Detailed constraints
    budget: BudgetConstraint = field(default_factory=BudgetConstraint)
    time: TimeConstraint = field(default_factory=TimeConstraint)
    surface: SurfaceConstraint = field(default_factory=SurfaceConstraint)
    output: OutputConstraint = field(default_factory=OutputConstraint)

    # Color preferences
    color_palette_name: Optional[str] = None    # e.g., "limited palette", "earth tones"
    max_colors: Optional[int] = None            # Limit number of paints
    required_colors: List[str] = field(default_factory=list)
    forbidden_colors: List[str] = field(default_factory=list)

    # Technique preferences
    preferred_techniques: List[str] = field(default_factory=list)
    forbidden_techniques: List[str] = field(default_factory=list)

    # Focus areas - what to emphasize in instructions
    focus_areas: List[str] = field(default_factory=list)  # e.g., ["fur texture", "eyes"]

    # Learning goals - what the user wants to learn
    learning_goals: List[str] = field(default_factory=list)

    # Reference style images (for style transfer)
    style_references: List[Path] = field(default_factory=list)

    # Additional metadata
    title: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and process constraints."""
        if self.source_path and self.source_image is None:
            # Load image from path
            from PIL import Image
            img = Image.open(self.source_path)
            self.source_image = np.array(img)

    @property
    def image_dimensions(self) -> Optional[Tuple[int, int]]:
        """Get source image dimensions (height, width)."""
        if self.source_image is not None:
            return self.source_image.shape[:2]
        return None

    @property
    def is_portrait_orientation(self) -> bool:
        """Check if source is portrait orientation."""
        if dims := self.image_dimensions:
            return dims[0] > dims[1]
        return False

    def get_complexity_multiplier(self) -> float:
        """Get complexity based on style and skill level."""
        style_complexity = {
            Style.FLAT: 0.5,
            Style.LOOSE: 0.6,
            Style.IMPRESSIONISM: 0.7,
            Style.PAINTERLY: 0.8,
            Style.EXPRESSIONISM: 0.8,
            Style.REALISM: 1.0,
            Style.CLASSICAL: 1.1,
            Style.TIGHT: 1.2,
            Style.CONTEMPORARY: 1.0,
            Style.PHOTOREALISM: 1.5,
        }

        skill_detail = {
            SkillLevel.BEGINNER: 0.7,      # Fewer steps, less detail
            SkillLevel.INTERMEDIATE: 1.0,
            SkillLevel.ADVANCED: 1.2,
            SkillLevel.EXPERT: 1.4,
        }

        return style_complexity.get(self.style, 1.0) * skill_detail.get(self.skill_level, 1.0)

    def get_instruction_detail_level(self) -> int:
        """Get instruction detail level (1-5) based on skill."""
        skill_detail = {
            SkillLevel.BEGINNER: 5,        # Most detailed
            SkillLevel.INTERMEDIATE: 3,
            SkillLevel.ADVANCED: 2,
            SkillLevel.EXPERT: 1,          # Minimal guidance
        }
        return skill_detail.get(self.skill_level, 3)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize constraints to dictionary."""
        return {
            "medium": self.medium.value,
            "style": self.style.value,
            "skill_level": self.skill_level.value,
            "subject_domain": self.subject_domain.value if self.subject_domain else None,
            "surface": {
                "width": self.surface.width_inches,
                "height": self.surface.height_inches,
                "type": self.surface.surface_type,
            },
            "budget": {
                "max": self.budget.max_budget,
                "currency": self.budget.currency,
            },
            "time": {
                "max_hours": self.time.max_hours,
                "session_length": self.time.session_length_hours,
            },
            "color_palette": self.color_palette_name,
            "max_colors": self.max_colors,
            "focus_areas": self.focus_areas,
            "learning_goals": self.learning_goals,
        }

    @classmethod
    def from_simple(
        cls,
        image_path: str,
        medium: str = "acrylic",
        style: str = "painterly",
        skill: str = "intermediate"
    ) -> "ArtConstraints":
        """Create constraints from simple string inputs."""
        return cls(
            source_path=Path(image_path),
            medium=Medium(medium),
            style=Style(style),
            skill_level=SkillLevel(skill),
        )


# Preset constraint configurations
PRESETS = {
    "beginner_acrylic": ArtConstraints(
        medium=Medium.ACRYLIC,
        style=Style.PAINTERLY,
        skill_level=SkillLevel.BEGINNER,
        budget=BudgetConstraint(max_budget=50, prefer_budget_friendly=True),
        time=TimeConstraint(max_hours=8, session_length_hours=2),
        output=OutputConstraint(
            instruction_verbosity="detailed",
            include_tips=True,
            include_common_mistakes=True,
        ),
        max_colors=8,
    ),

    "intermediate_oil": ArtConstraints(
        medium=Medium.OIL,
        style=Style.CLASSICAL,
        skill_level=SkillLevel.INTERMEDIATE,
        time=TimeConstraint(max_hours=20, session_length_hours=3),
        output=OutputConstraint(instruction_verbosity="moderate"),
    ),

    "quick_sketch": ArtConstraints(
        medium=Medium.GRAPHITE,
        style=Style.LOOSE,
        skill_level=SkillLevel.INTERMEDIATE,
        time=TimeConstraint(max_hours=1),
        output=OutputConstraint(
            instruction_verbosity="minimal",
            generate_step_images=False,
        ),
    ),

    "detailed_portrait": ArtConstraints(
        medium=Medium.OIL,
        style=Style.REALISM,
        skill_level=SkillLevel.ADVANCED,
        subject_domain=SubjectDomain.PORTRAIT,
        focus_areas=["skin tones", "eyes", "hair", "lighting"],
        learning_goals=["flesh mixing", "eye rendering", "hair texture"],
    ),

    "pet_portrait": ArtConstraints(
        medium=Medium.ACRYLIC,
        style=Style.PAINTERLY,
        skill_level=SkillLevel.INTERMEDIATE,
        subject_domain=SubjectDomain.ANIMAL,
        focus_areas=["fur texture", "eyes", "expression"],
        learning_goals=["fur strokes", "animal anatomy", "expressive eyes"],
    ),
}
