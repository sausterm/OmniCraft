"""
Lesson Plan Generator - Creates structured art lessons from constraints.

This module generates comprehensive lesson plans that guide creators through
the process of recreating an image in their chosen medium. It combines:
- Scene understanding (what's in the image)
- Art principles (how to construct it)
- Pedagogical structure (how to teach it)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from artisan.core.constraints import ArtConstraints, Medium, Style, SkillLevel, SubjectDomain
from artisan.core.art_principles import (
    ArtPrinciplesEngine, ConstructionPhilosophy, LayerPurpose,
    TechniqueKnowledge, PHILOSOPHIES, CONSTRUCTION_LAYERS
)
from artisan.core.scene_graph import SceneGraph, Entity, EntityType
from artisan.core.construction import (
    ConstructionPlan, ConstructionPhase, ConstructionStep,
    Technique, Tool, Material, TECHNIQUES, TOOLS
)


@dataclass
class MaterialRequirement:
    """A required material with specifics."""
    name: str
    category: str                      # paint, brush, surface, medium, etc.
    quantity: str                      # "1 tube", "2oz", etc.
    purpose: str                       # Why it's needed
    alternatives: List[str] = field(default_factory=list)
    optional: bool = False
    estimated_cost: Optional[float] = None


@dataclass
class ColorMixingRecipe:
    """Recipe for mixing a specific color."""
    target_color: Tuple[int, int, int]  # RGB
    target_name: str                     # "Warm mid-skin tone"
    base_colors: List[str]               # Paint names
    ratios: List[float]                  # Mixing ratios
    notes: str = ""


@dataclass
class LessonStep:
    """A single step in the lesson."""
    step_number: int
    title: str
    objective: str                      # What this step achieves
    instruction: str                    # Detailed instruction
    technique: str                      # Primary technique used
    duration_minutes: int

    # Visual aids
    reference_image_key: Optional[str] = None  # Key to lookup reference image
    focus_region_key: Optional[str] = None     # Region being worked on

    # Teaching elements
    tips: List[str] = field(default_factory=list)
    common_mistakes: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)  # What to check before continuing

    # Materials for this step
    colors_used: List[str] = field(default_factory=list)
    tools_used: List[str] = field(default_factory=list)


@dataclass
class LessonPhase:
    """A phase of the lesson containing multiple steps."""
    phase_number: int
    name: str
    description: str
    purpose: str                        # Art principle/purpose
    estimated_duration_minutes: int
    steps: List[LessonStep] = field(default_factory=list)

    # Phase-level guidance
    overview: str = ""
    key_concepts: List[str] = field(default_factory=list)
    rest_after: bool = False            # Recommend break after this phase


@dataclass
class LessonPlan:
    """
    A complete lesson plan for creating artwork.

    This is the main output of the planning system - a structured,
    pedagogically sound guide to creating the artwork.
    """
    # Metadata
    title: str
    description: str
    created_at: str
    version: str = "1.0"

    # Source and target
    source_image_path: Optional[str] = None
    target_medium: str = "acrylic"
    target_style: str = "painterly"
    skill_level: str = "intermediate"
    subject_domain: str = "mixed"

    # Construction philosophy
    philosophy: str = ""
    philosophy_description: str = ""

    # Learning objectives
    learning_objectives: List[str] = field(default_factory=list)

    # Materials
    materials: List[MaterialRequirement] = field(default_factory=list)
    color_palette: List[ColorMixingRecipe] = field(default_factory=list)
    total_estimated_cost: Optional[float] = None

    # Time estimates
    total_duration_minutes: int = 0
    recommended_sessions: int = 1
    session_duration_minutes: int = 120

    # The actual lesson content
    phases: List[LessonPhase] = field(default_factory=list)

    # Reference materials (generated separately)
    reference_images: Dict[str, str] = field(default_factory=dict)  # key -> path

    # Additional guidance
    preparation_checklist: List[str] = field(default_factory=list)
    workspace_setup: List[str] = field(default_factory=list)
    warmup_exercises: List[str] = field(default_factory=list)

    def get_total_steps(self) -> int:
        """Get total number of steps across all phases."""
        return sum(len(phase.steps) for phase in self.phases)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON export."""
        return {
            "metadata": {
                "title": self.title,
                "description": self.description,
                "created_at": self.created_at,
                "version": self.version,
            },
            "target": {
                "medium": self.target_medium,
                "style": self.target_style,
                "skill_level": self.skill_level,
                "subject_domain": self.subject_domain,
            },
            "philosophy": {
                "name": self.philosophy,
                "description": self.philosophy_description,
            },
            "learning_objectives": self.learning_objectives,
            "time_estimate": {
                "total_minutes": self.total_duration_minutes,
                "recommended_sessions": self.recommended_sessions,
                "session_length_minutes": self.session_duration_minutes,
            },
            "materials": [
                {
                    "name": m.name,
                    "category": m.category,
                    "quantity": m.quantity,
                    "purpose": m.purpose,
                    "alternatives": m.alternatives,
                    "optional": m.optional,
                    "estimated_cost": m.estimated_cost,
                }
                for m in self.materials
            ],
            "color_palette": [
                {
                    "name": c.target_name,
                    "rgb": c.target_color,
                    "base_colors": c.base_colors,
                    "ratios": c.ratios,
                    "notes": c.notes,
                }
                for c in self.color_palette
            ],
            "preparation": {
                "checklist": self.preparation_checklist,
                "workspace": self.workspace_setup,
                "warmup": self.warmup_exercises,
            },
            "phases": [
                {
                    "phase_number": p.phase_number,
                    "name": p.name,
                    "description": p.description,
                    "purpose": p.purpose,
                    "duration_minutes": p.estimated_duration_minutes,
                    "overview": p.overview,
                    "key_concepts": p.key_concepts,
                    "rest_after": p.rest_after,
                    "steps": [
                        {
                            "step_number": s.step_number,
                            "title": s.title,
                            "objective": s.objective,
                            "instruction": s.instruction,
                            "technique": s.technique,
                            "duration_minutes": s.duration_minutes,
                            "tips": s.tips,
                            "common_mistakes": s.common_mistakes,
                            "checkpoints": s.checkpoints,
                            "colors_used": s.colors_used,
                            "tools_used": s.tools_used,
                        }
                        for s in p.steps
                    ],
                }
                for p in self.phases
            ],
            "reference_images": self.reference_images,
        }

    def save(self, path: Path):
        """Save lesson plan to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class LessonPlanGenerator:
    """
    Generates lesson plans from constraints and scene analysis.

    This is the main orchestrator that brings together:
    - User constraints (what they want to create)
    - Scene understanding (what's in the image)
    - Art principles (how to construct it)
    - Pedagogical knowledge (how to teach it)
    """

    def __init__(self):
        self.art_engine = ArtPrinciplesEngine()

    def generate(
        self,
        constraints: ArtConstraints,
        scene_graph: Optional[SceneGraph] = None
    ) -> LessonPlan:
        """
        Generate a complete lesson plan.

        Args:
            constraints: User-defined constraints for the artwork
            scene_graph: Analyzed scene graph (if available)

        Returns:
            Complete LessonPlan ready for execution
        """
        # Select construction philosophy
        philosophy = self.art_engine.select_philosophy(
            style=constraints.style.value,
            medium=constraints.medium.value,
            subject=constraints.subject_domain.value if constraints.subject_domain else "mixed",
            time_constraint=constraints.time.max_hours,
        )

        # Get layer sequence
        complexity = constraints.get_complexity_multiplier()
        layers = self.art_engine.get_layer_sequence(philosophy, complexity)

        # Generate lesson plan structure
        plan = LessonPlan(
            title=constraints.title or self._generate_title(constraints),
            description=constraints.description or self._generate_description(constraints),
            created_at=datetime.now().isoformat(),
            source_image_path=str(constraints.source_path) if constraints.source_path else None,
            target_medium=constraints.medium.value,
            target_style=constraints.style.value,
            skill_level=constraints.skill_level.value,
            subject_domain=constraints.subject_domain.value if constraints.subject_domain else "mixed",
            philosophy=philosophy.name,
            philosophy_description=philosophy.description,
        )

        # Generate learning objectives
        plan.learning_objectives = self.art_engine.generate_learning_objectives(
            subject=constraints.subject_domain.value if constraints.subject_domain else "mixed",
            techniques_used=[],  # Will be populated as we generate
            skill_level=constraints.skill_level.value,
        )

        # Generate materials list
        plan.materials = self._generate_materials_list(constraints)

        # Generate color palette
        if scene_graph and constraints.source_image is not None:
            plan.color_palette = self._generate_color_palette(
                constraints.source_image, scene_graph, constraints
            )

        # Generate phases
        plan.phases = self._generate_phases(constraints, scene_graph, layers, philosophy)

        # Calculate totals
        plan.total_duration_minutes = sum(p.estimated_duration_minutes for p in plan.phases)
        plan.recommended_sessions = max(1, plan.total_duration_minutes // constraints.time.session_length_hours // 60)

        # Generate preparation checklist
        plan.preparation_checklist = self._generate_preparation_checklist(constraints)
        plan.workspace_setup = self._generate_workspace_setup(constraints)
        plan.warmup_exercises = self._generate_warmup_exercises(constraints)

        return plan

    def _generate_title(self, constraints: ArtConstraints) -> str:
        """Generate a title for the lesson."""
        medium = constraints.medium.value.replace("_", " ").title()
        style = constraints.style.value.replace("_", " ").title()
        domain = constraints.subject_domain.value.title() if constraints.subject_domain else "Study"
        return f"{style} {domain} in {medium}"

    def _generate_description(self, constraints: ArtConstraints) -> str:
        """Generate a description for the lesson."""
        skill = constraints.skill_level.value
        return f"A {skill}-level lesson in creating {constraints.style.value} artwork using {constraints.medium.value}."

    def _generate_materials_list(self, constraints: ArtConstraints) -> List[MaterialRequirement]:
        """Generate the materials needed for this lesson."""
        materials = []

        # Surface
        materials.append(MaterialRequirement(
            name=f"{constraints.surface.width_inches}x{constraints.surface.height_inches}\" {constraints.surface.surface_type.title()}",
            category="surface",
            quantity="1",
            purpose="Painting surface",
            estimated_cost=15.0,
        ))

        # Medium-specific materials
        if constraints.medium == Medium.ACRYLIC:
            materials.extend([
                MaterialRequirement(
                    name="Titanium White", category="paint", quantity="1 tube (large)",
                    purpose="Mixing, highlights", estimated_cost=8.0,
                ),
                MaterialRequirement(
                    name="Mars Black", category="paint", quantity="1 tube",
                    purpose="Mixing, shadows", estimated_cost=5.0,
                    alternatives=["Ivory Black"],
                ),
                MaterialRequirement(
                    name="Cadmium Yellow Medium", category="paint", quantity="1 tube",
                    purpose="Primary yellow", estimated_cost=8.0,
                    alternatives=["Hansa Yellow"],
                ),
                MaterialRequirement(
                    name="Cadmium Red Medium", category="paint", quantity="1 tube",
                    purpose="Primary red", estimated_cost=8.0,
                    alternatives=["Naphthol Red"],
                ),
                MaterialRequirement(
                    name="Ultramarine Blue", category="paint", quantity="1 tube",
                    purpose="Primary blue", estimated_cost=5.0,
                ),
                MaterialRequirement(
                    name="Burnt Sienna", category="paint", quantity="1 tube",
                    purpose="Warm earth tone, mixing", estimated_cost=5.0,
                ),
                MaterialRequirement(
                    name="Flat Brush Set", category="brush", quantity="1 set",
                    purpose="Blocking in, large areas", estimated_cost=15.0,
                ),
                MaterialRequirement(
                    name="Round Brush Set", category="brush", quantity="1 set",
                    purpose="Details, blending", estimated_cost=12.0,
                ),
                MaterialRequirement(
                    name="Palette", category="tool", quantity="1",
                    purpose="Color mixing", estimated_cost=8.0,
                ),
                MaterialRequirement(
                    name="Water Container", category="tool", quantity="2",
                    purpose="Brush cleaning", estimated_cost=0.0,
                ),
            ])

        elif constraints.medium == Medium.OIL:
            materials.extend([
                MaterialRequirement(
                    name="Odorless Mineral Spirits", category="solvent", quantity="16oz",
                    purpose="Brush cleaning, thinning", estimated_cost=10.0,
                ),
                MaterialRequirement(
                    name="Linseed Oil", category="medium", quantity="4oz",
                    purpose="Fat over lean, glazing", estimated_cost=8.0,
                ),
            ])

        # Add optional items based on skill level
        if constraints.skill_level == SkillLevel.BEGINNER:
            materials.append(MaterialRequirement(
                name="Practice Canvas", category="surface", quantity="2",
                purpose="Practice before main piece", optional=True, estimated_cost=10.0,
            ))

        return materials

    def _generate_color_palette(
        self,
        image: np.ndarray,
        scene_graph: SceneGraph,
        constraints: ArtConstraints
    ) -> List[ColorMixingRecipe]:
        """Generate color mixing recipes from image analysis."""
        recipes = []

        # Extract dominant colors from image
        from artisan.core.color_matcher import ColorMatcher
        # Placeholder - would analyze image for key colors
        # and generate mixing recipes

        return recipes

    def _generate_phases(
        self,
        constraints: ArtConstraints,
        scene_graph: Optional[SceneGraph],
        layers: List[LayerPurpose],
        philosophy: ConstructionPhilosophy
    ) -> List[LessonPhase]:
        """Generate the phases of the lesson."""
        phases = []
        step_counter = 0
        detail_level = constraints.get_instruction_detail_level()

        # Phase 0: Preparation (always included for beginners)
        if constraints.skill_level in [SkillLevel.BEGINNER, SkillLevel.INTERMEDIATE]:
            prep_phase = LessonPhase(
                phase_number=0,
                name="Preparation",
                description="Set up your workspace and materials",
                purpose="Proper preparation leads to better results",
                estimated_duration_minutes=15,
                overview="Before we begin painting, let's make sure everything is ready.",
                key_concepts=["Organization", "Material preparation", "Workspace setup"],
            )
            prep_phase.steps = self._generate_preparation_steps(constraints, step_counter)
            step_counter += len(prep_phase.steps)
            phases.append(prep_phase)

        # Generate phases for each construction layer
        for i, layer in enumerate(layers):
            phase = self._generate_layer_phase(
                layer=layer,
                phase_number=i + 1,
                constraints=constraints,
                scene_graph=scene_graph,
                step_counter=step_counter,
                detail_level=detail_level,
            )
            step_counter += len(phase.steps)
            phases.append(phase)

            # Add rest recommendations
            if layer.name in ["Block In", "Development"]:
                phase.rest_after = True

        return phases

    def _generate_preparation_steps(
        self,
        constraints: ArtConstraints,
        start_step: int
    ) -> List[LessonStep]:
        """Generate preparation steps."""
        steps = []

        steps.append(LessonStep(
            step_number=start_step + 1,
            title="Organize Materials",
            objective="Have all materials within reach",
            instruction="Lay out all your paints, brushes, and tools. Fill water containers. "
                       "Place your reference image where you can easily see it.",
            technique="preparation",
            duration_minutes=5,
            tips=["Keep frequently used colors closer to your dominant hand"],
            checkpoints=["All paints opened and ready", "Brushes clean and dry", "Reference visible"],
        ))

        if constraints.medium == Medium.ACRYLIC:
            steps.append(LessonStep(
                step_number=start_step + 2,
                title="Prepare Palette",
                objective="Set up colors for efficient mixing",
                instruction="Squeeze out your colors around the edge of the palette, leaving the center "
                           "clear for mixing. Start with small amounts - acrylic dries quickly.",
                technique="preparation",
                duration_minutes=5,
                tips=[
                    "Arrange colors in spectrum order (warm to cool)",
                    "Keep white separate and larger than other colors",
                    "Use a stay-wet palette if you have one",
                ],
                common_mistakes=["Squeezing too much paint (wastes paint as it dries)"],
            ))

        return steps

    def _generate_layer_phase(
        self,
        layer: LayerPurpose,
        phase_number: int,
        constraints: ArtConstraints,
        scene_graph: Optional[SceneGraph],
        step_counter: int,
        detail_level: int
    ) -> LessonPhase:
        """Generate a phase for a construction layer."""
        phase = LessonPhase(
            phase_number=phase_number,
            name=layer.name,
            description=layer.description,
            purpose=f"Applying {', '.join(p.value for p in layer.principles_applied)}",
            estimated_duration_minutes=int(120 * layer.time_allocation),
            key_concepts=[p.value.title() for p in layer.principles_applied],
        )

        # Generate overview based on layer
        phase.overview = self._generate_layer_overview(layer, constraints)

        # Generate steps for this layer
        phase.steps = self._generate_layer_steps(
            layer=layer,
            constraints=constraints,
            scene_graph=scene_graph,
            start_step=step_counter,
            detail_level=detail_level,
        )

        return phase

    def _generate_layer_overview(
        self,
        layer: LayerPurpose,
        constraints: ArtConstraints
    ) -> str:
        """Generate overview text for a layer phase."""
        overviews = {
            "Toning/Ground": (
                "We'll start by applying a thin, even tone to the entire canvas. "
                "This eliminates the intimidating white surface and provides a unified "
                "starting point for our values."
            ),
            "Drawing/Sketch": (
                "Now we'll lightly sketch the main shapes and proportions. "
                "Don't worry about details yet - focus on accurate placement and "
                "relative sizes of the major elements."
            ),
            "Underpainting": (
                "The underpainting establishes our value structure. Working in a "
                "limited palette, we'll map out the lights and darks that will "
                "guide our color application."
            ),
            "Block In": (
                "Time to add color! We'll work quickly to cover the canvas with "
                "approximate colors. Don't blend too much - we're establishing "
                "the overall color relationships."
            ),
            "Development": (
                "Now we refine. We'll adjust colors, improve shapes, and begin "
                "developing form. This is where the painting starts to come together."
            ),
            "Detail": (
                "With the foundation solid, we can add the details that bring "
                "the painting to life. Work from general to specific."
            ),
            "Finishing": (
                "Final adjustments: punch up highlights, deepen shadows where needed, "
                "and add those final touches that make the painting sing."
            ),
        }

        return overviews.get(layer.name, layer.description)

    def _generate_layer_steps(
        self,
        layer: LayerPurpose,
        constraints: ArtConstraints,
        scene_graph: Optional[SceneGraph],
        start_step: int,
        detail_level: int
    ) -> List[LessonStep]:
        """Generate steps for a construction layer."""
        steps = []
        current_step = start_step

        # Get techniques appropriate for this layer
        techniques = self.art_engine.suggest_techniques(
            layer=layer.name.lower().replace("/", "_").replace(" ", "_"),
            subject_type=constraints.subject_domain.value if constraints.subject_domain else "mixed",
            textures_needed=[]
        )

        if scene_graph:
            # Generate steps based on scene entities
            entities = self._get_entities_for_layer(scene_graph, layer)

            for entity in entities:
                current_step += 1
                step = self._generate_entity_step(
                    entity=entity,
                    layer=layer,
                    step_number=current_step,
                    constraints=constraints,
                    techniques=techniques,
                    detail_level=detail_level,
                )
                steps.append(step)
        else:
            # Generate generic steps for this layer
            current_step += 1
            step = LessonStep(
                step_number=current_step,
                title=f"Apply {layer.name}",
                objective=layer.description,
                instruction=self._generate_generic_instruction(layer, constraints),
                technique=techniques[0].name if techniques else "direct application",
                duration_minutes=int(120 * layer.time_allocation / max(1, detail_level)),
            )
            steps.append(step)

        return steps

    def _get_entities_for_layer(
        self,
        scene_graph: SceneGraph,
        layer: LayerPurpose
    ) -> List[Entity]:
        """Get entities relevant to a construction layer."""
        # Map layers to entity types
        layer_entity_map = {
            "Toning/Ground": [],  # No specific entities
            "Drawing/Sketch": [EntityType.SUBJECT, EntityType.ENVIRONMENT],
            "Underpainting": [EntityType.BACKGROUND, EntityType.ENVIRONMENT, EntityType.SUBJECT],
            "Block In": [EntityType.BACKGROUND, EntityType.ENVIRONMENT, EntityType.SUBJECT],
            "Development": [EntityType.SUBJECT, EntityType.ENVIRONMENT],
            "Detail": [EntityType.SUBJECT, EntityType.DETAIL],
            "Finishing": [EntityType.ACCENT, EntityType.DETAIL],
        }

        target_types = layer_entity_map.get(layer.name, [])
        entities = []

        for entity_type in target_types:
            entities.extend(scene_graph.get_entities_by_type(entity_type))

        # Sort by depth (background first for most layers)
        return sorted(entities, key=lambda e: e.properties.depth_hint if e.properties else 0.5)

    def _generate_entity_step(
        self,
        entity: Entity,
        layer: LayerPurpose,
        step_number: int,
        constraints: ArtConstraints,
        techniques: List[TechniqueKnowledge],
        detail_level: int
    ) -> LessonStep:
        """Generate a step for working on a specific entity."""
        # Select appropriate technique
        technique = techniques[0] if techniques else None

        # Generate instruction based on entity and layer
        instruction = self._generate_entity_instruction(entity, layer, constraints, technique)

        step = LessonStep(
            step_number=step_number,
            title=f"{layer.name}: {entity.name}",
            objective=f"Apply {layer.name.lower()} to {entity.name}",
            instruction=instruction,
            technique=technique.name if technique else "direct application",
            duration_minutes=int(10 * layer.time_allocation * (1 + entity.coverage)),
            focus_region_key=entity.id,
        )

        # Add tips and common mistakes from technique
        if technique:
            step.tips = technique.tips[:detail_level]
            step.common_mistakes = technique.common_mistakes[:detail_level]

        return step

    def _generate_entity_instruction(
        self,
        entity: Entity,
        layer: LayerPurpose,
        constraints: ArtConstraints,
        technique: Optional[TechniqueKnowledge]
    ) -> str:
        """Generate instruction text for working on an entity."""
        # Base instruction
        instruction = f"Working on the {entity.name.lower()}"

        # Add technique-specific guidance
        if technique:
            instruction += f", use {technique.name.lower()}. "
            instruction += technique.description + " "
            instruction += f"Brush motion: {technique.brush_motion}. "

        # Add entity-specific guidance
        if entity.entity_type == EntityType.BACKGROUND:
            instruction += "Work loosely - this area supports the subject, not competes with it. "
        elif entity.entity_type == EntityType.SUBJECT:
            instruction += "Take your time here - this is a focal area. "

        # Add skill-level adjustments
        if constraints.skill_level == SkillLevel.BEGINNER:
            instruction += "Don't worry if it's not perfect - we can refine it later. "

        return instruction.strip()

    def _generate_generic_instruction(
        self,
        layer: LayerPurpose,
        constraints: ArtConstraints
    ) -> str:
        """Generate generic instruction for a layer when no scene graph."""
        instructions = {
            "Toning/Ground": (
                "Mix a neutral mid-tone (try burnt sienna thinned with water/medium). "
                "Apply evenly across the entire canvas using large, sweeping strokes. "
                "The tone should be transparent enough to see through, but eliminate "
                "the white of the canvas."
            ),
            "Drawing/Sketch": (
                "Using a thin brush and diluted paint (or pencil), lightly sketch "
                "the main shapes. Focus on proportions and placement. Keep lines "
                "light - they'll be covered by paint."
            ),
            "Block In": (
                "Working from the background forward, block in the major color areas. "
                "Don't worry about blending yet - focus on getting colors approximately "
                "correct and covering the canvas."
            ),
            "Development": (
                "Now refine the colors and shapes. Adjust values, improve edges, "
                "and begin developing form through value transitions."
            ),
            "Detail": (
                "Add specific details: textures, patterns, small forms. "
                "Work from general areas to specific points of interest."
            ),
            "Finishing": (
                "Add final highlights with thick, confident strokes. "
                "Deepen shadows where needed for contrast. "
                "Step back frequently to assess the overall effect."
            ),
        }

        return instructions.get(layer.name, layer.description)

    def _generate_preparation_checklist(self, constraints: ArtConstraints) -> List[str]:
        """Generate preparation checklist."""
        checklist = [
            "Reference image printed or displayed",
            "Canvas/surface secured and at comfortable height",
            "All paints available and in good condition",
            "Brushes clean and ready",
            "Palette cleaned and prepared",
            "Water/solvent containers filled",
            "Paper towels or rags available",
            "Good lighting (ideally north light or daylight bulbs)",
        ]

        if constraints.skill_level == SkillLevel.BEGINNER:
            checklist.extend([
                "Timer set for breaks every 30-45 minutes",
                "Reference guide or color chart available",
            ])

        return checklist

    def _generate_workspace_setup(self, constraints: ArtConstraints) -> List[str]:
        """Generate workspace setup instructions."""
        return [
            "Position your canvas at a slight angle (15-30 degrees) to reduce glare",
            "Place palette on your dominant side, within easy reach",
            "Keep reference image at eye level, adjacent to canvas",
            "Ensure ventilation if using solvents",
            "Protect floor/table with drop cloth",
            "Have a comfortable seat or standing position you can maintain",
        ]

    def _generate_warmup_exercises(self, constraints: ArtConstraints) -> List[str]:
        """Generate warmup exercises based on skill level."""
        exercises = []

        if constraints.skill_level == SkillLevel.BEGINNER:
            exercises = [
                "Practice brush control: paint parallel lines, then curves",
                "Mix 3 values of one color (light, mid, dark)",
                "Paint a simple gradient from one color to another",
            ]
        elif constraints.skill_level == SkillLevel.INTERMEDIATE:
            exercises = [
                "Quick value study sketch (5 minutes)",
                "Mix your key colors and test on scrap",
            ]
        else:
            exercises = [
                "Quick gesture sketch to loosen up",
            ]

        return exercises
