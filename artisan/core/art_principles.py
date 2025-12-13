"""
Art Principles - Foundation of art education and construction.

This module encodes fundamental principles of art and design that inform
how artwork should be constructed. These principles are universal across
mediums and styles - the specifics of application vary, but the principles
remain constant.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Callable
import numpy as np


class Principle(Enum):
    """Fundamental principles of art and design."""
    # Elements of Art
    LINE = "line"                      # Marks, edges, contours
    SHAPE = "shape"                    # 2D areas (geometric, organic)
    FORM = "form"                      # 3D illusion
    VALUE = "value"                    # Light and dark
    COLOR = "color"                    # Hue, saturation, intensity
    TEXTURE = "texture"                # Surface quality
    SPACE = "space"                    # Positive/negative, depth

    # Principles of Design
    BALANCE = "balance"                # Visual equilibrium
    CONTRAST = "contrast"              # Differences create interest
    EMPHASIS = "emphasis"              # Focal point
    MOVEMENT = "movement"              # Eye flow through composition
    PATTERN = "pattern"                # Repetition
    RHYTHM = "rhythm"                  # Visual tempo
    UNITY = "unity"                    # Cohesion


class ConstructionOrder(Enum):
    """Standard construction orderings used in classical art education."""
    BACK_TO_FRONT = "back_to_front"    # Traditional: background first
    DARK_TO_LIGHT = "dark_to_light"    # Value-based: darks establish form
    LIGHT_TO_DARK = "light_to_dark"    # Watercolor approach
    LARGE_TO_SMALL = "large_to_small"  # Shape-based: big shapes first
    GENERAL_TO_SPECIFIC = "general_to_specific"  # Start loose, refine
    WARM_TO_COOL = "warm_to_cool"      # Color temperature approach
    LEAN_TO_FAT = "lean_to_fat"        # Oil painting rule


@dataclass
class ConstructionPhilosophy:
    """
    A philosophy of construction that guides how artwork is built.

    Different approaches (Alla Prima, Flemish, Bob Ross wet-on-wet)
    use different construction philosophies.
    """
    name: str
    description: str

    # Primary ordering principle
    primary_order: ConstructionOrder

    # Whether layers must dry between steps
    requires_drying: bool = False
    drying_between_phases: bool = True

    # Layer approach
    uses_underpainting: bool = True
    uses_glazing: bool = False
    uses_impasto: bool = False

    # Working style
    wet_on_wet: bool = False
    wet_on_dry: bool = True

    # Typical number of passes/layers
    typical_layers: int = 3

    # Time characteristics
    single_session: bool = False

    # Best for
    suited_for_styles: List[str] = field(default_factory=list)
    suited_for_subjects: List[str] = field(default_factory=list)


# Standard construction philosophies
PHILOSOPHIES = {
    "classical_realism": ConstructionPhilosophy(
        name="Classical Realism",
        description="Traditional layered approach: drawing, underpainting, dead layer, color layers, glazes",
        primary_order=ConstructionOrder.DARK_TO_LIGHT,
        requires_drying=True,
        drying_between_phases=True,
        uses_underpainting=True,
        uses_glazing=True,
        typical_layers=5,
        suited_for_styles=["realism", "classical", "tight"],
        suited_for_subjects=["portrait", "still_life"],
    ),

    "alla_prima": ConstructionPhilosophy(
        name="Alla Prima (Direct Painting)",
        description="Complete painting in one session while paint is wet",
        primary_order=ConstructionOrder.BACK_TO_FRONT,
        requires_drying=False,
        drying_between_phases=False,
        uses_underpainting=False,
        wet_on_wet=True,
        wet_on_dry=False,
        typical_layers=1,
        single_session=True,
        suited_for_styles=["impressionism", "loose", "expressionism"],
        suited_for_subjects=["landscape", "portrait"],
    ),

    "bob_ross": ConstructionPhilosophy(
        name="Bob Ross Wet-on-Wet",
        description="Liquid white base, wet-on-wet technique, complete in one session",
        primary_order=ConstructionOrder.BACK_TO_FRONT,
        requires_drying=False,
        wet_on_wet=True,
        wet_on_dry=False,
        uses_underpainting=True,  # Liquid white as base
        typical_layers=1,
        single_session=True,
        suited_for_styles=["painterly", "loose"],
        suited_for_subjects=["landscape"],
    ),

    "watercolor_traditional": ConstructionPhilosophy(
        name="Traditional Watercolor",
        description="Light to dark, transparent layers, preserve white of paper",
        primary_order=ConstructionOrder.LIGHT_TO_DARK,
        requires_drying=True,
        wet_on_dry=True,
        wet_on_wet=True,
        uses_glazing=True,
        typical_layers=4,
        suited_for_styles=["loose", "impressionism"],
        suited_for_subjects=["landscape", "botanical"],
    ),

    "acrylic_layered": ConstructionPhilosophy(
        name="Layered Acrylic",
        description="Build up layers with fast-drying acrylic, can work wet or dry",
        primary_order=ConstructionOrder.GENERAL_TO_SPECIFIC,
        requires_drying=False,  # Fast drying allows flexibility
        uses_underpainting=True,
        uses_glazing=True,
        typical_layers=3,
        suited_for_styles=["painterly", "realism", "contemporary"],
        suited_for_subjects=["portrait", "animal", "landscape"],
    ),
}


@dataclass
class LayerPurpose:
    """Defines the purpose and approach for a construction layer."""
    name: str
    description: str
    principles_applied: List[Principle]
    typical_coverage: float          # 0-1, how much of canvas
    precision_level: float           # 0-1, loose to tight
    time_allocation: float           # Fraction of total time


# Standard construction layers used across mediums
CONSTRUCTION_LAYERS = {
    # Foundation layers
    "toning": LayerPurpose(
        name="Toning/Ground",
        description="Establish mid-tone ground to work on",
        principles_applied=[Principle.VALUE, Principle.COLOR],
        typical_coverage=1.0,
        precision_level=0.1,
        time_allocation=0.05,
    ),

    "drawing": LayerPurpose(
        name="Drawing/Sketch",
        description="Establish proportions, placement, and major shapes",
        principles_applied=[Principle.LINE, Principle.SHAPE, Principle.SPACE],
        typical_coverage=0.8,
        precision_level=0.5,
        time_allocation=0.1,
    ),

    "underpainting": LayerPurpose(
        name="Underpainting",
        description="Establish values, often in monochrome or limited color",
        principles_applied=[Principle.VALUE, Principle.FORM],
        typical_coverage=1.0,
        precision_level=0.4,
        time_allocation=0.15,
    ),

    "block_in": LayerPurpose(
        name="Block In",
        description="Establish major color masses and relationships",
        principles_applied=[Principle.COLOR, Principle.SHAPE, Principle.VALUE],
        typical_coverage=1.0,
        precision_level=0.3,
        time_allocation=0.2,
    ),

    "development": LayerPurpose(
        name="Development",
        description="Refine shapes, values, and color relationships",
        principles_applied=[Principle.FORM, Principle.COLOR, Principle.VALUE],
        typical_coverage=0.9,
        precision_level=0.6,
        time_allocation=0.25,
    ),

    "detail": LayerPurpose(
        name="Detail",
        description="Add specific details, textures, and refinements",
        principles_applied=[Principle.TEXTURE, Principle.LINE, Principle.EMPHASIS],
        typical_coverage=0.3,
        precision_level=0.8,
        time_allocation=0.15,
    ),

    "finishing": LayerPurpose(
        name="Finishing",
        description="Final adjustments, highlights, accents",
        principles_applied=[Principle.EMPHASIS, Principle.CONTRAST, Principle.UNITY],
        typical_coverage=0.1,
        precision_level=0.9,
        time_allocation=0.1,
    ),
}


@dataclass
class TechniqueKnowledge:
    """
    Knowledge about a specific technique - how to do it, when to use it.
    """
    name: str
    description: str

    # When to use
    layer_applicability: List[str]      # Which layers this works for
    subject_suitability: List[str]      # What subjects benefit
    texture_creation: List[str]         # What textures it creates

    # How to do it
    brush_motion: str                   # Direction, pressure, speed
    paint_consistency: str              # Thick, thin, medium
    working_time: str                   # Fast, slow, variable

    # Learning curve
    difficulty: float                   # 0-1
    common_mistakes: List[str] = field(default_factory=list)
    tips: List[str] = field(default_factory=list)

    # What it achieves
    principles_expressed: List[Principle] = field(default_factory=list)


# Technique knowledge base
TECHNIQUES = {
    "wet_blend": TechniqueKnowledge(
        name="Wet Blending",
        description="Blend two colors together while both are wet on the surface",
        layer_applicability=["block_in", "development"],
        subject_suitability=["sky", "skin", "fur", "fabric"],
        texture_creation=["smooth", "gradient"],
        brush_motion="Gentle, overlapping strokes where colors meet",
        paint_consistency="Medium to thin, workable",
        working_time="Must work quickly before paint dries",
        difficulty=0.4,
        common_mistakes=[
            "Working too slowly - paint dries",
            "Too much pressure - lifting paint",
            "Overworking - creating mud",
        ],
        tips=[
            "Work in small sections",
            "Keep brush clean between strokes",
            "Use a misting spray to extend working time (acrylic)",
        ],
        principles_expressed=[Principle.VALUE, Principle.COLOR],
    ),

    "dry_brush": TechniqueKnowledge(
        name="Dry Brush",
        description="Use minimal paint on brush to create broken, textured strokes",
        layer_applicability=["development", "detail", "finishing"],
        subject_suitability=["fur", "hair", "grass", "bark", "fabric"],
        texture_creation=["rough", "organic", "directional"],
        brush_motion="Light, quick strokes following form direction",
        paint_consistency="Thick, minimal on brush",
        working_time="Can work slowly",
        difficulty=0.3,
        common_mistakes=[
            "Too much paint - loses texture effect",
            "Too much pressure - solid coverage",
            "Wrong brush angle",
        ],
        tips=[
            "Remove excess paint on paper towel first",
            "Hold brush at low angle",
            "Build up gradually with multiple passes",
        ],
        principles_expressed=[Principle.TEXTURE, Principle.MOVEMENT],
    ),

    "glazing": TechniqueKnowledge(
        name="Glazing",
        description="Apply thin, transparent layer over dried paint to modify color",
        layer_applicability=["development", "finishing"],
        subject_suitability=["skin", "fabric", "glass", "shadow"],
        texture_creation=["smooth", "luminous"],
        brush_motion="Smooth, even application",
        paint_consistency="Very thin, transparent",
        working_time="Can work slowly",
        difficulty=0.5,
        common_mistakes=[
            "Paint too thick - becomes opaque",
            "Working over wet paint - lifts underlayer",
            "Uneven application - streaks",
        ],
        tips=[
            "Ensure underlayer is completely dry",
            "Use glazing medium",
            "Apply in thin, even coats",
        ],
        principles_expressed=[Principle.COLOR, Principle.VALUE],
    ),

    "scumbling": TechniqueKnowledge(
        name="Scumbling",
        description="Apply thin, opaque/semi-opaque layer allowing underlayer to show through",
        layer_applicability=["development", "detail"],
        subject_suitability=["atmosphere", "fog", "aged surfaces", "skin"],
        texture_creation=["hazy", "textured", "atmospheric"],
        brush_motion="Light, broken strokes",
        paint_consistency="Thin but opaque",
        working_time="Medium pace",
        difficulty=0.4,
        common_mistakes=[
            "Too heavy application",
            "Complete coverage - loses effect",
        ],
        tips=[
            "Use a dry brush technique",
            "Build up gradually",
        ],
        principles_expressed=[Principle.TEXTURE, Principle.SPACE],
    ),

    "impasto": TechniqueKnowledge(
        name="Impasto",
        description="Apply paint thickly to create texture and dimension",
        layer_applicability=["development", "detail", "finishing"],
        subject_suitability=["highlights", "flowers", "waves", "clouds"],
        texture_creation=["thick", "dimensional", "sculptural"],
        brush_motion="Bold, confident strokes",
        paint_consistency="Very thick",
        working_time="Variable",
        difficulty=0.4,
        common_mistakes=[
            "Overworking - flattens texture",
            "Inconsistent thickness",
        ],
        tips=[
            "Use palette knife for best texture",
            "Plan thick areas carefully",
            "Let dry completely before glazing over",
        ],
        principles_expressed=[Principle.TEXTURE, Principle.EMPHASIS],
    ),

    "stippling": TechniqueKnowledge(
        name="Stippling",
        description="Create texture and tone using dots/dabbing",
        layer_applicability=["development", "detail"],
        subject_suitability=["foliage", "fur", "skin texture", "fabric"],
        texture_creation=["dotted", "organic", "varied"],
        brush_motion="Perpendicular dabbing motion",
        paint_consistency="Medium",
        working_time="Slow, methodical",
        difficulty=0.3,
        common_mistakes=[
            "Too uniform - looks mechanical",
            "Dots too large - loses effect",
        ],
        tips=[
            "Vary dot size and spacing",
            "Use different brushes for variety",
            "Build up density gradually",
        ],
        principles_expressed=[Principle.TEXTURE, Principle.PATTERN],
    ),

    "feathering": TechniqueKnowledge(
        name="Feathering",
        description="Create soft, gradient transitions using light, feathery strokes",
        layer_applicability=["block_in", "development"],
        subject_suitability=["sky", "skin", "soft edges", "backgrounds"],
        texture_creation=["smooth", "soft", "gradient"],
        brush_motion="Light, decreasing pressure strokes",
        paint_consistency="Medium to thin",
        working_time="Medium pace",
        difficulty=0.4,
        common_mistakes=[
            "Too much pressure",
            "Strokes too visible",
        ],
        tips=[
            "Use very light touch",
            "Soft brush works best",
            "Work while paint is workable",
        ],
        principles_expressed=[Principle.VALUE, Principle.COLOR],
    ),
}


class ArtPrinciplesEngine:
    """
    Engine that applies art principles to generate construction strategies.

    This is the "brain" that knows how art should be made based on
    centuries of artistic knowledge and pedagogy.
    """

    def __init__(self):
        self.philosophies = PHILOSOPHIES
        self.layers = CONSTRUCTION_LAYERS
        self.techniques = TECHNIQUES

    def select_philosophy(
        self,
        style: str,
        medium: str,
        subject: str,
        time_constraint: Optional[float] = None
    ) -> ConstructionPhilosophy:
        """
        Select appropriate construction philosophy based on constraints.
        """
        candidates = []

        for name, philosophy in self.philosophies.items():
            score = 0

            # Score based on style match
            if style in philosophy.suited_for_styles:
                score += 3

            # Score based on subject match
            if subject in philosophy.suited_for_subjects:
                score += 2

            # Time constraint consideration
            if time_constraint:
                if time_constraint < 4 and philosophy.single_session:
                    score += 2
                elif time_constraint >= 10 and not philosophy.single_session:
                    score += 1

            # Medium considerations
            if medium == "oil" and philosophy.uses_glazing:
                score += 1
            if medium == "acrylic" and not philosophy.requires_drying:
                score += 1
            if medium == "watercolor" and philosophy.primary_order == ConstructionOrder.LIGHT_TO_DARK:
                score += 3

            candidates.append((score, name, philosophy))

        # Return highest scoring, or default
        candidates.sort(reverse=True, key=lambda x: x[0])
        return candidates[0][2] if candidates else self.philosophies["acrylic_layered"]

    def get_layer_sequence(
        self,
        philosophy: ConstructionPhilosophy,
        complexity: float = 1.0
    ) -> List[LayerPurpose]:
        """
        Get the sequence of layers for a given philosophy.
        """
        base_sequence = ["toning", "drawing", "block_in", "development", "detail", "finishing"]

        if philosophy.uses_underpainting:
            base_sequence.insert(2, "underpainting")

        # Adjust for complexity
        if complexity < 0.5:
            # Simplified approach
            return [self.layers[l] for l in ["drawing", "block_in", "development", "finishing"]
                    if l in self.layers]
        elif complexity > 1.2:
            # Full classical approach
            return [self.layers[l] for l in base_sequence if l in self.layers]
        else:
            # Standard approach
            return [self.layers[l] for l in ["toning", "drawing", "block_in", "development", "detail", "finishing"]
                    if l in self.layers]

    def suggest_techniques(
        self,
        layer: str,
        subject_type: str,
        textures_needed: List[str]
    ) -> List[TechniqueKnowledge]:
        """
        Suggest techniques appropriate for a layer and subject.
        """
        suggestions = []

        for name, technique in self.techniques.items():
            score = 0

            if layer in technique.layer_applicability:
                score += 2

            if subject_type in technique.subject_suitability:
                score += 2

            for texture in textures_needed:
                if texture in technique.texture_creation:
                    score += 1

            if score > 0:
                suggestions.append((score, technique))

        suggestions.sort(reverse=True, key=lambda x: x[0])
        return [t for _, t in suggestions[:3]]

    def get_construction_rules(self, medium: str) -> Dict[str, Any]:
        """
        Get medium-specific construction rules.
        """
        rules = {
            "acrylic": {
                "fat_over_lean": False,  # Not applicable
                "drying_time_minutes": 15,
                "can_reactivate": False,
                "glazing_medium_required": True,
                "working_time_minutes": 20,
                "layer_sequence": "flexible",
            },
            "oil": {
                "fat_over_lean": True,  # Critical rule
                "drying_time_minutes": 1440,  # 24 hours minimum
                "can_reactivate": True,  # Can blend into dried layer
                "glazing_medium_required": True,
                "working_time_minutes": 480,  # All day
                "layer_sequence": "strict",
                "rules": [
                    "Fat over lean: Each layer should have more oil than the previous",
                    "Thick over thin: Apply thick paint over thin",
                    "Slow drying over fast drying: Don't put fast-drying under slow",
                ],
            },
            "watercolor": {
                "preserve_whites": True,  # No white paint
                "work_light_to_dark": True,
                "drying_time_minutes": 5,
                "can_reactivate": True,  # Can lift dried paint
                "working_time_minutes": 5,
                "layer_sequence": "strict",
                "rules": [
                    "Preserve white of paper for highlights",
                    "Work light to dark - cannot lighten",
                    "Let layers dry to preserve transparency",
                ],
            },
            "gouache": {
                "can_lighten": True,  # Unlike watercolor
                "drying_time_minutes": 10,
                "can_reactivate": True,
                "working_time_minutes": 15,
                "layer_sequence": "flexible",
            },
        }

        return rules.get(medium, rules["acrylic"])

    def generate_learning_objectives(
        self,
        subject: str,
        techniques_used: List[str],
        skill_level: str
    ) -> List[str]:
        """
        Generate learning objectives for the lesson.
        """
        objectives = []

        # Subject-based objectives
        subject_objectives = {
            "portrait": [
                "Understand facial proportions and structure",
                "Mix accurate skin tones",
                "Render eyes, nose, and mouth effectively",
            ],
            "animal": [
                "Understand animal anatomy basics",
                "Create convincing fur/hair texture",
                "Capture expression and character",
            ],
            "landscape": [
                "Create atmospheric perspective",
                "Establish believable light and shadow",
                "Develop foreground, middle ground, background",
            ],
            "still_life": [
                "Observe and render accurate values",
                "Understand light on different surfaces",
                "Create convincing textures",
            ],
        }

        if subject in subject_objectives:
            objectives.extend(subject_objectives[subject])

        # Technique-based objectives
        for tech_name in techniques_used:
            if tech := self.techniques.get(tech_name):
                objectives.append(f"Practice {tech.name}: {tech.description}")

        # Skill-level adjustments
        if skill_level == "beginner":
            objectives = objectives[:3]  # Focus on fewer objectives
            objectives.insert(0, "Build confidence through structured guidance")
        elif skill_level == "advanced":
            objectives.append("Develop personal style and interpretation")

        return objectives
