"""
Oil Instruction Renderer - Detailed oil painting instructions.

Oil-specific considerations:
- Slow drying (days to weeks)
- Fat-over-lean rule
- Requires solvents for cleanup
- Blends for hours
- Rich, luminous colors
- Centuries of traditional technique
"""

from typing import List, Dict, Any
from .base import (
    InstructionRenderer, RenderedLesson, RenderedPhase,
    RenderedInstruction
)
from ..planning.lesson_plan import LessonPlan, LessonStep, LessonPhase


class OilInstructionRenderer(InstructionRenderer):
    """Renders instructions specifically for oil painting."""

    @property
    def medium_name(self) -> str:
        return "Oil"

    @property
    def medium_characteristics(self) -> Dict[str, Any]:
        return {
            "drying_time": "Days to weeks (varies by color)",
            "cleanup": "Odorless mineral spirits or soap/oil",
            "layering": "Fat over lean - each layer needs more oil",
            "transparency": "Naturally transparent when thinned",
            "color_shift": "Minimal when dry",
            "working_time": "Hours to days",
            "flexibility": "Brittle if rules not followed",
            "toxicity": "Some pigments toxic, requires ventilation",
        }

    TECHNIQUE_INSTRUCTIONS = {
        "wet_blend": (
            "Oil's long working time makes blending a joy. Apply both colors to canvas, "
            "then blend where they meet using a clean, soft brush. Work gently with "
            "light, feathery strokes. You can continue refining for hours if needed. "
            "For ultra-smooth blends, use a badger blender or soft mop."
        ),
        "dry_brush": (
            "Load your brush with paint, wipe on a rag until almost dry. Lightly skim "
            "across the surface. For oil, this works best over dried or tacky underlayers. "
            "The slow drying of oil means you can build up texture gradually over sessions."
        ),
        "glazing": (
            "Wait until underlayer is completely dry (touch dry minimum, fully cured "
            "preferred - several days to weeks). Mix paint with glazing medium "
            "(1 part paint to 3-5 parts medium). Apply in thin, even strokes. "
            "Oil glazes have unmatched luminosity due to their transparency."
        ),
        "scumbling": (
            "Apply a thin layer of lighter, opaque paint over a darker dried layer. "
            "Use a stiff brush with minimal paint, dragging lightly. The underlayer "
            "shows through the broken coverage, creating atmospheric effects."
        ),
        "impasto": (
            "Use paint straight from the tube or add cold wax or impasto medium for body. "
            "Apply boldly with brush or palette knife. Thick oil paint creates beautiful "
            "texture but takes longer to dry - thick areas may take months to cure fully."
        ),
        "underpainting": (
            "Mix paint with solvent (no oil) for a lean first layer. Use earth tones "
            "or monochrome. Establish values and composition. This layer should dry "
            "completely (24-48 hours) before adding fat layers."
        ),
        "fat_over_lean": (
            "CRITICAL RULE: Each successive layer should contain more oil than the "
            "previous. Start with paint thinned with solvent (lean), progress to paint "
            "with oil added (fat). Violating this rule causes cracking as layers dry."
        ),
    }

    CONSISTENCY_GUIDE = {
        "wet_blend": "Creamy, as it comes from tube",
        "dry_brush": "Thick, wiped nearly dry on rag",
        "glazing": "Very thin - paint + lots of glazing medium",
        "scumbling": "Thick, dry on brush",
        "impasto": "Very thick - may add impasto medium",
        "underpainting": "Thin with solvent - should dry quickly",
        "block_in": "Slightly thinned with solvent/medium",
        "fine_detail": "Creamy - may thin slightly for flow",
    }

    DRYING_TIMES = {
        "Titanium White": "3-5 days",
        "Cadmium Yellow": "5-7 days",
        "Cadmium Red": "5-7 days",
        "Ultramarine Blue": "2-3 days",
        "Burnt Sienna": "2-3 days",
        "Burnt Umber": "1-2 days (fastest drying)",
        "Ivory Black": "5-7 days",
        "Alizarin Crimson": "5-7 days (very slow)",
    }

    def render_lesson(self, lesson: LessonPlan) -> RenderedLesson:
        """Render complete oil painting lesson."""
        rendered = RenderedLesson(
            title=lesson.title,
            medium=self.medium_name,
            style=lesson.target_style,
            skill_level=lesson.skill_level,
            introduction=self._generate_oil_introduction(lesson),
            learning_objectives=lesson.learning_objectives + [
                "Understand and apply the fat-over-lean principle",
                "Manage oil paint drying times across sessions",
            ],
            supply_list=self._generate_supply_list(lesson),
            color_mixing_guide=self._generate_color_guide(lesson),
            workspace_setup=self._generate_workspace_setup(),
            preparation_steps=self._generate_prep_steps(lesson),
        )

        # Render each phase
        for phase in lesson.phases:
            rendered_phase = self._render_phase(phase)
            rendered.phases.append(rendered_phase)

        rendered.finishing_checklist = self._generate_finishing_checklist()
        rendered.care_instructions = self._generate_care_instructions()
        rendered.next_steps = self._generate_next_steps(lesson)

        return rendered

    def _generate_oil_introduction(self, lesson: LessonPlan) -> str:
        """Generate oil-specific introduction."""
        intro = self._generate_introduction(lesson)
        intro += (
            "\n\nOil paint has been the medium of choice for master painters for over "
            "500 years. Its slow drying time allows for endless blending and refinement, "
            "and its rich colors are unmatched. Key things to know:\n"
            "- FAT OVER LEAN: Each layer must have more oil than the previous (critical!)\n"
            "- Oil paint stays workable for hours - no rushing\n"
            "- Clean up requires solvents or soap/oil - not water\n"
            "- Work in well-ventilated space if using solvents\n"
            "- Allow proper drying time between sessions (typically 24-48 hours minimum)"
        )
        return intro

    def _generate_supply_list(self, lesson: LessonPlan) -> List[Dict[str, Any]]:
        """Generate oil-specific supply list."""
        supplies = []

        # Essential oil painting supplies
        oil_essentials = [
            {
                "name": "Odorless Mineral Spirits (OMS)",
                "category": "solvent",
                "quantity": "16 oz",
                "purpose": "Brush cleaning and thinning paint (lean layers)",
                "required": True,
            },
            {
                "name": "Linseed Oil or Painting Medium",
                "category": "medium",
                "quantity": "4 oz",
                "purpose": "Fat layers, glazing, extending paint",
                "required": True,
            },
            {
                "name": "Brush Cleaner Container",
                "category": "tool",
                "quantity": "1",
                "purpose": "Holds solvent for brush cleaning",
                "required": True,
            },
            {
                "name": "Palette Knife",
                "category": "tool",
                "quantity": "1",
                "purpose": "Mixing paint, cleaning palette, impasto",
                "required": True,
            },
            {
                "name": "Lint-free Rags",
                "category": "tool",
                "quantity": "Several",
                "purpose": "Wiping brushes, cleaning up",
                "required": True,
            },
        ]

        supplies.extend(oil_essentials)

        # Add from lesson materials
        for material in lesson.materials:
            supply = {
                "name": material.name,
                "category": material.category,
                "quantity": material.quantity,
                "purpose": material.purpose,
                "required": not material.optional,
            }
            supplies.append(supply)

        return supplies

    def _generate_color_guide(self, lesson: LessonPlan) -> List[Dict[str, Any]]:
        """Generate color guide with oil-specific notes."""
        guides = []

        for recipe in lesson.color_palette:
            guide = {
                "color_name": recipe.target_name,
                "target_rgb": recipe.target_color,
                "base_paints": recipe.base_colors,
                "mixing_ratios": recipe.ratios,
                "oil_notes": self._get_oil_mixing_notes(recipe.base_colors),
            }
            guides.append(guide)

        # Add drying time reference
        guides.append({
            "color_name": "Drying Times Reference",
            "oil_notes": "Know your drying times: Burnt Umber is fastest (1-2 days), "
                        "Alizarin Crimson and Blacks are slowest (5-7 days). "
                        "Plan layers accordingly.",
            "drying_reference": self.DRYING_TIMES,
        })

        return guides

    def _get_oil_mixing_notes(self, colors: List[str]) -> str:
        """Get oil-specific mixing notes."""
        # Check for slow-drying colors
        slow_colors = ["alizarin", "black", "white"]
        has_slow = any(any(s in c.lower() for s in slow_colors) for c in colors)

        if has_slow:
            return (
                "This mix contains slow-drying pigments. Allow extra drying time "
                "before adding additional layers. Consider adding a small amount of "
                "cobalt drier if needed (use sparingly)."
            )
        return "Mix thoroughly on palette. Test on scrap before applying."

    def _generate_workspace_setup(self) -> List[str]:
        """Generate oil workspace setup."""
        return [
            "Ensure good ventilation - open windows or use exhaust fan if using solvents",
            "Cover work surface with disposable palette paper or glass palette",
            "Set up brush cleaning container with OMS - keep covered when not in use",
            "Have lint-free rags ready for wiping brushes",
            "Position canvas at comfortable angle - easel recommended",
            "Keep mediums and solvents organized and clearly labeled",
            "Have palette knife clean and ready for mixing",
            "Wear old clothes - oil paint is permanent on fabric",
            "Keep a trash container nearby for used rags (fire hazard - dispose properly)",
        ]

    def _generate_prep_steps(self, lesson: LessonPlan) -> List[str]:
        """Generate oil-specific preparation steps."""
        return [
            "Squeeze out colors in organized arrangement around palette edge",
            "Prepare medium mixture (e.g., 1 part OMS to 1 part linseed oil for fat layers)",
            "Pour small amount of OMS into brush cleaning container",
            "Ensure all brushes are clean and dry - oil and water don't mix",
            "If working over previous session, check if paint is dry to touch",
            "Plan your session - which layers need to be applied today?",
        ]

    def _render_phase(self, phase: LessonPhase) -> RenderedPhase:
        """Render phase with oil-specific guidance."""
        rendered = RenderedPhase(
            phase_number=phase.phase_number,
            name=phase.name,
            overview=phase.overview,
            key_concepts=phase.key_concepts + ["Fat over lean"],
            instructions=[],
            estimated_duration=self._format_duration(phase.estimated_duration_minutes),
        )

        # Add fat-over-lean notes
        if phase.phase_number == 1:
            rendered.setup_notes = (
                "First layer should be LEAN - thin paint with solvent only, no oil. "
                "This creates a stable foundation for subsequent fat layers."
            )
        else:
            rendered.setup_notes = (
                "This layer can be FATTER than the previous. Add medium containing oil "
                "to your paint. Each layer builds on the previous."
            )

        for step in phase.steps:
            rendered_instruction = self.render_step(step)
            rendered.instructions.append(rendered_instruction)

        rendered.cleanup_notes = (
            "Clean brushes thoroughly with OMS, then with brush soap. "
            "Never leave brushes in solvent. Dispose of solvent-soaked rags properly - "
            "they can spontaneously combust."
        )

        return rendered

    def render_step(self, step: LessonStep) -> RenderedInstruction:
        """Render step with oil-specific details."""
        technique = step.technique.lower().replace(" ", "_")

        base_instruction = self.TECHNIQUE_INSTRUCTIONS.get(technique, step.instruction)

        rendered = RenderedInstruction(
            step_number=step.step_number,
            title=step.title,
            instruction_text=f"{step.instruction}\n\n**Oil technique:** {base_instruction}",
            paint_consistency=self.CONSISTENCY_GUIDE.get(technique, "Creamy from tube"),
            brush_load="Fully loaded with paint/medium mixture",
            stroke_type=self._get_stroke_type(technique),
            paints=step.colors_used,
            brushes=step.tools_used,
            working_time="No rush - oil stays wet for hours",
            drying_time=self._estimate_drying_time(step.colors_used),
            tips=step.tips + self._get_oil_tips(technique),
            warnings=self._get_oil_warnings(technique),
            checkpoints=step.checkpoints + ["Check layer leanness/fatness"],
        )

        return rendered

    def _get_stroke_type(self, technique: str) -> str:
        """Get stroke type for technique."""
        strokes = {
            "wet_blend": "Soft, feathery strokes with clean brush",
            "glazing": "Thin, even strokes in one direction",
            "impasto": "Bold, confident strokes - place and leave",
            "underpainting": "Thin, sketchy strokes to establish values",
        }
        return strokes.get(technique, "Natural brush strokes")

    def _estimate_drying_time(self, colors: List[str]) -> str:
        """Estimate drying time based on colors used."""
        if not colors:
            return "24-48 hours typical"

        # Check for slow-drying colors
        slow = any("white" in c.lower() or "black" in c.lower() for c in colors)
        if slow:
            return "3-5 days (contains slow-drying pigments)"
        return "1-3 days (depending on thickness)"

    def _get_oil_tips(self, technique: str) -> List[str]:
        """Get oil-specific tips."""
        tips = {
            "wet_blend": [
                "You have hours to perfect your blend - no rushing",
                "Use a badger blender for ultra-smooth transitions",
            ],
            "glazing": [
                "Oil glazes are luminous because light passes through to underlayer",
                "Wait until underlayer is fully cured for best results",
            ],
            "underpainting": [
                "Use fast-drying colors like Burnt Umber for underpainting",
                "Keep this layer thin and matte - no shine",
            ],
        }
        return tips.get(technique, [])

    def _get_oil_warnings(self, technique: str) -> List[str]:
        """Get oil-specific warnings."""
        return [
            "ALWAYS follow fat-over-lean rule to prevent cracking",
            "Dispose of oil/solvent rags properly - fire hazard",
            "Work in ventilated space",
        ]

    def _generate_finishing_checklist(self) -> List[str]:
        """Generate oil finishing checklist."""
        return [
            "Allow painting to dry completely (touch dry: 1-2 weeks, cured: 6-12 months)",
            "Do not varnish until fully cured (6-12 months minimum)",
            "Apply retouch varnish after 2 weeks if needed for even sheen",
            "Sign your work with oil paint or wait until dry and use permanent marker",
        ]

    def _generate_care_instructions(self) -> List[str]:
        """Generate care instructions for oil painting."""
        return [
            "Store paintings in dust-free environment while drying",
            "Keep paintings away from direct heat sources while curing",
            "Do not stack wet or tacky paintings face-to-face",
            "Apply final varnish after 6-12 months of curing",
            "Clean with soft, dry cloth - no solvents on dried paint",
        ]

    def _generate_next_steps(self, lesson: LessonPlan) -> List[str]:
        """Generate next steps for oil painting learning."""
        return [
            "Practice alla prima (wet-into-wet) technique",
            "Explore the glazing masters: Vermeer, Rembrandt",
            "Experiment with different mediums (stand oil, alkyd)",
            "Try a limited palette study for color mixing mastery",
        ]
