"""
Acrylic Instruction Renderer - Detailed acrylic painting instructions.

Acrylic-specific considerations:
- Fast drying (15-30 minutes)
- Can layer immediately
- Water cleanup
- Darkens slightly when dry
- Can mimic oil or watercolor techniques
- Flexible film, less prone to cracking
"""

from typing import List, Dict, Any
from .base import (
    InstructionRenderer, RenderedLesson, RenderedPhase,
    RenderedInstruction
)
from artisan.paint.planning.lesson_plan import LessonPlan, LessonStep, LessonPhase


class AcrylicInstructionRenderer(InstructionRenderer):
    """Renders instructions specifically for acrylic painting."""

    @property
    def medium_name(self) -> str:
        return "Acrylic"

    @property
    def medium_characteristics(self) -> Dict[str, Any]:
        return {
            "drying_time": "15-30 minutes",
            "cleanup": "Water while wet",
            "layering": "Immediate once dry",
            "transparency": "Can be opaque or transparent",
            "color_shift": "Darkens 5-10% when dry",
            "working_time": "20-30 minutes before drying",
            "flexibility": "Very flexible when dry",
            "toxicity": "Non-toxic (most brands)",
        }

    # Technique-specific instructions for acrylic
    TECHNIQUE_INSTRUCTIONS = {
        "wet_blend": (
            "Work quickly - acrylic dries fast. Apply both colors to your surface, "
            "then immediately blend where they meet using a clean, damp brush. "
            "Use light, overlapping strokes. If paint starts to drag, mist lightly "
            "with water or add a touch of retarder medium."
        ),
        "dry_brush": (
            "Load your brush with paint, then wipe most of it off on a paper towel "
            "until the brush is almost dry. Lightly drag across the surface with "
            "the side of the brush, allowing texture to show through. Build up "
            "gradually with multiple passes."
        ),
        "glazing": (
            "Ensure the underlayer is completely dry (wait at least 30 minutes). "
            "Mix paint with glazing medium until very transparent. Apply in thin, "
            "even strokes. Don't over-brush - acrylic dries quickly. Allow to dry "
            "fully before adding additional glazes."
        ),
        "scumbling": (
            "Use a small amount of thick paint on a dry brush. Lightly drag across "
            "the dried surface, allowing the underlayer to show through the broken "
            "coverage. Keep the brush relatively dry and use light pressure."
        ),
        "impasto": (
            "Use paint straight from the tube or mix with heavy gel medium for extra body. "
            "Apply thickly with brush or palette knife. Don't overwork - place strokes "
            "confidently and leave them. Thick areas may take 24+ hours to dry fully."
        ),
        "stippling": (
            "Hold brush perpendicular to canvas. Dab with the tip in a bouncing motion, "
            "varying the dot size by adjusting pressure. Work methodically across the area, "
            "building up density and color variation."
        ),
        "gradient": (
            "Lay both colors side by side on canvas. Working quickly before they dry, "
            "blend the transition zone with a clean, damp brush using horizontal strokes "
            "that cross the boundary. Work across the entire width for even blending. "
            "Mist with water if needed to extend working time."
        ),
        "base_coat": (
            "Thin paint with water to a creamy consistency. Apply with a large flat brush "
            "using long, even strokes. Cover the entire area, working the paint into the "
            "canvas texture. This layer should be thin and even."
        ),
        "block_in": (
            "Mix approximate colors for each major area. Apply with a medium-sized brush, "
            "filling in the shapes you've sketched. Don't worry about precise edges yet - "
            "focus on establishing the overall color relationships. Work from large areas "
            "to small."
        ),
        "fine_detail": (
            "Use a small round brush (size 0-2) with paint at a creamy consistency. "
            "Brace your hand for stability. Work slowly and deliberately. Clean your brush "
            "frequently to maintain sharp edges. For the finest details, thin paint slightly "
            "with water."
        ),
        "highlighting": (
            "Mix highlight colors with Titanium White - they should be thick and opaque. "
            "Apply with a confident single stroke where light hits strongest. Don't "
            "over-blend. The highlight should sit on top of the underlying paint, "
            "creating slight texture and emphasis."
        ),
        "fur_strokes": (
            "Use a small round or filbert brush. Load with paint, then wipe partially dry. "
            "Apply short, directional strokes following the fur growth pattern. Vary stroke "
            "length and overlap strokes for natural appearance. Work from the base of the fur "
            "outward, lighter strokes on top."
        ),
    }

    # Brush recommendations for acrylic
    BRUSH_MAP = {
        "broad_applicator": "2-inch flat synthetic brush",
        "medium_applicator": "1-inch flat synthetic brush or #12 flat",
        "detail_tool": "#2 round synthetic brush or liner brush",
        "blending_tool": "Soft mop brush or large filbert",
        "texture_tool": "Fan brush or old stiff bristle brush",
    }

    # Consistency guide
    CONSISTENCY_GUIDE = {
        "wet_blend": "Creamy, like soft butter - add water or slow-dry medium",
        "dry_brush": "Thick, straight from tube, then wipe most off",
        "glazing": "Very thin, like colored water - mostly medium",
        "scumbling": "Thick, dry - minimal paint on brush",
        "impasto": "Very thick, may add heavy gel medium",
        "stippling": "Medium consistency, creamy",
        "gradient": "Creamy, slightly thinned with water",
        "base_coat": "Thin, like heavy cream - add water",
        "block_in": "Medium consistency, creamy",
        "fine_detail": "Creamy to slightly thin - test on palette",
        "highlighting": "Thick, opaque - straight from tube",
        "fur_strokes": "Medium to thick, slightly dry on brush",
    }

    def render_lesson(self, lesson: LessonPlan) -> RenderedLesson:
        """Render complete acrylic lesson."""
        rendered = RenderedLesson(
            title=lesson.title,
            medium=self.medium_name,
            style=lesson.target_style,
            skill_level=lesson.skill_level,
            introduction=self._generate_acrylic_introduction(lesson),
            learning_objectives=lesson.learning_objectives,
            supply_list=self._generate_supply_list(lesson),
            color_mixing_guide=self._generate_color_guide(lesson),
            workspace_setup=self._generate_workspace_setup(),
            preparation_steps=self._generate_prep_steps(lesson),
        )

        # Render each phase
        for phase in lesson.phases:
            rendered_phase = self._render_phase(phase)
            rendered.phases.append(rendered_phase)

        # Add finishing sections
        rendered.finishing_checklist = self._generate_finishing_checklist()
        rendered.care_instructions = self._generate_care_instructions()
        rendered.next_steps = self._generate_next_steps(lesson)

        return rendered

    def _generate_acrylic_introduction(self, lesson: LessonPlan) -> str:
        """Generate acrylic-specific introduction."""
        intro = self._generate_introduction(lesson)
        intro += (
            "\n\nAcrylic paint is wonderfully versatile - it dries quickly (15-30 minutes), "
            "cleans up with water, and can achieve both transparent and opaque effects. "
            "Key tips for working with acrylics:\n"
            "- Work in small sections to avoid paint drying before you can blend\n"
            "- Keep a spray bottle of water handy to mist your palette and canvas\n"
            "- Colors darken slightly as they dry - mix a shade lighter than you want\n"
            "- Once dry, acrylic is permanent and can be painted over without lifting"
        )
        return intro

    def _generate_supply_list(self, lesson: LessonPlan) -> List[Dict[str, Any]]:
        """Generate acrylic-specific supply list."""
        supplies = []

        # From lesson materials
        for material in lesson.materials:
            supply = {
                "name": material.name,
                "category": material.category,
                "quantity": material.quantity,
                "purpose": material.purpose,
                "required": not material.optional,
            }
            if material.alternatives:
                supply["alternatives"] = material.alternatives
            if material.estimated_cost:
                supply["estimated_cost"] = material.estimated_cost
            supplies.append(supply)

        # Add acrylic-specific essentials if not present
        essential_names = [s["name"] for s in supplies]

        if not any("spray bottle" in n.lower() for n in essential_names):
            supplies.append({
                "name": "Spray Bottle with Water",
                "category": "tool",
                "quantity": "1",
                "purpose": "Keep paint workable and extend blending time",
                "required": True,
            })

        if not any("stay-wet" in n.lower() or "palette" in n.lower() for n in essential_names):
            supplies.append({
                "name": "Stay-Wet Palette or Covered Palette",
                "category": "tool",
                "quantity": "1",
                "purpose": "Keeps acrylic paint from drying on palette",
                "required": False,
            })

        return supplies

    def _generate_color_guide(self, lesson: LessonPlan) -> List[Dict[str, Any]]:
        """Generate color mixing guide for acrylic."""
        guides = []

        for recipe in lesson.color_palette:
            guide = {
                "color_name": recipe.target_name,
                "target_rgb": recipe.target_color,
                "base_paints": recipe.base_colors,
                "mixing_ratios": recipe.ratios,
                "acrylic_notes": self._get_mixing_notes(recipe.target_name),
            }
            guides.append(guide)

        # Add common acrylic mixing tips
        if not guides:
            guides.append({
                "color_name": "General Mixing Tips",
                "acrylic_notes": (
                    "Start with the lighter color and add darker colors gradually. "
                    "Mix more than you think you need - matching dried acrylic is difficult. "
                    "Test your mix on scrap and let it dry to see true color."
                ),
            })

        return guides

    def _get_mixing_notes(self, color_name: str) -> str:
        """Get mixing notes for specific colors."""
        color_lower = color_name.lower()

        if "skin" in color_lower:
            return (
                "Start with white + yellow ochre, add tiny amounts of red. "
                "Cool skin tones: add touch of blue. Warm skin: add more red/orange. "
                "Acrylic skin tones should be mixed slightly lighter - they darken when dry."
            )
        elif "sky" in color_lower:
            return (
                "Start with white, add ultramarine gradually. For warm sky, "
                "add touch of red. Mix plenty - sky requires a lot of paint. "
                "Work quickly when applying to prevent lap marks."
            )
        elif "shadow" in color_lower:
            return (
                "Avoid pure black for shadows. Mix complements or use "
                "ultramarine + burnt sienna. Add to base color to darken. "
                "Shadows often contain reflected colors from surroundings."
            )
        else:
            return "Mix on palette until uniform. Test on scrap and allow to dry."

    def _generate_workspace_setup(self) -> List[str]:
        """Generate acrylic workspace setup instructions."""
        return [
            "Cover your work surface with a plastic sheet or newspaper",
            "Set up two water containers: one for cleaning brushes, one for clean water",
            "Prepare your palette - if using a stay-wet palette, dampen the sponge and paper",
            "Lay out paper towels or rags for wiping brushes",
            "Position your canvas at a comfortable angle (flat or slightly tilted)",
            "Have your spray bottle filled and within reach",
            "Ensure good lighting - natural north light or daylight bulbs are best",
            "Wear old clothes or an apron - acrylic doesn't wash out once dry!",
        ]

    def _generate_prep_steps(self, lesson: LessonPlan) -> List[str]:
        """Generate acrylic-specific preparation steps."""
        steps = [
            "Squeeze out small amounts of each paint color around the edge of your palette",
            "Keep colors organized - warm colors together, cool colors together",
            "Mist your palette with water to slow drying",
            "Clean all brushes and have them ready - don't let acrylic dry in bristles",
        ]

        if lesson.skill_level == "beginner":
            steps.extend([
                "Print your reference image at a similar size to your canvas",
                "Do a quick sketch on scrap paper to warm up",
                "Have this lesson guide open and visible for reference",
            ])

        return steps

    def _render_phase(self, phase: LessonPhase) -> RenderedPhase:
        """Render a phase with acrylic-specific guidance."""
        rendered = RenderedPhase(
            phase_number=phase.phase_number,
            name=phase.name,
            overview=phase.overview,
            key_concepts=phase.key_concepts,
            instructions=[],
            estimated_duration=self._format_duration(phase.estimated_duration_minutes),
        )

        # Add phase-specific setup notes for acrylic
        if "block" in phase.name.lower():
            rendered.setup_notes = (
                "Block-in phase: Work quickly to establish colors while paint is wet. "
                "Mist your canvas occasionally if needed. Don't overthink - just get "
                "approximate colors down."
            )
        elif "detail" in phase.name.lower():
            rendered.setup_notes = (
                "Detail phase: Work slowly and deliberately. Keep detail brushes clean. "
                "Let areas dry between detail passes to avoid lifting paint."
            )

        # Render each step
        for step in phase.steps:
            rendered_instruction = self.render_step(step)
            rendered.instructions.append(rendered_instruction)

        # Add break recommendations
        if phase.estimated_duration_minutes > 45:
            rendered.recommended_breaks = [
                f"Take a 5-10 minute break halfway through this phase",
                "Step back and assess your work - fresh eyes catch problems",
            ]

        return rendered

    def render_step(self, step: LessonStep) -> RenderedInstruction:
        """Render a single step with acrylic-specific details."""
        technique = step.technique.lower().replace(" ", "_")

        # Get technique-specific instruction
        base_instruction = self.TECHNIQUE_INSTRUCTIONS.get(
            technique,
            step.instruction
        )

        # Build rendered instruction
        rendered = RenderedInstruction(
            step_number=step.step_number,
            title=step.title,
            instruction_text=f"{step.instruction}\n\n**Technique detail:** {base_instruction}",
            paint_consistency=self.CONSISTENCY_GUIDE.get(technique, "Medium consistency"),
            brush_load="Load brush, then wipe excess on paper towel" if "dry" in technique else "Fully loaded",
            stroke_type=self._get_stroke_type(technique),
            paints=step.colors_used,
            brushes=[self.BRUSH_MAP.get(t, t) for t in step.tools_used],
            working_time="Work quickly - 10-15 minutes before drying" if "blend" in technique else "No rush",
            tips=step.tips + self._get_acrylic_tips(technique),
            warnings=self._get_acrylic_warnings(technique),
            checkpoints=step.checkpoints,
        )

        return rendered

    def _get_stroke_type(self, technique: str) -> str:
        """Get stroke type description for a technique."""
        stroke_types = {
            "wet_blend": "Soft, overlapping horizontal strokes",
            "dry_brush": "Light, skimming strokes with brush held at low angle",
            "glazing": "Even, parallel strokes in one direction",
            "stippling": "Perpendicular dabbing motion",
            "impasto": "Bold, confident strokes - place and leave",
            "fur_strokes": "Short, directional strokes following fur growth",
            "gradient": "Horizontal strokes crossing the color boundary",
            "fine_detail": "Precise, controlled strokes - brace your hand",
        }
        return stroke_types.get(technique, "Natural brush strokes")

    def _get_acrylic_tips(self, technique: str) -> List[str]:
        """Get acrylic-specific tips for a technique."""
        tips = {
            "wet_blend": [
                "Mist canvas lightly before blending to extend working time",
                "Add a drop of slow-dry medium to paint for more blending time",
                "Work in smaller sections if paint is drying too fast",
            ],
            "dry_brush": [
                "The drier the brush, the more texture you'll get",
                "Build up layers - it's easier to add more than remove",
            ],
            "glazing": [
                "Test glaze on scrap first - it's easy to go too dark",
                "Clean brush between strokes to avoid muddy glazes",
            ],
            "fur_strokes": [
                "Vary stroke length and direction for natural look",
                "Work from dark underlayer to light top hairs",
                "Don't try to paint every hair - suggest texture instead",
            ],
        }
        return tips.get(technique, [])

    def _get_acrylic_warnings(self, technique: str) -> List[str]:
        """Get acrylic-specific warnings for a technique."""
        warnings = {
            "wet_blend": [
                "Don't overwork - stop when blended or you'll lift paint",
                "If paint starts dragging, stop and let it dry, then glaze over",
            ],
            "glazing": [
                "Glaze over fully dried paint only - wet paint will lift",
                "Don't apply thick glazes - they lose transparency",
            ],
            "impasto": [
                "Thick acrylic may take 24+ hours to dry completely",
                "Don't layer thick paint over thin - it can crack",
            ],
        }
        return warnings.get(technique, [])

    def _generate_finishing_checklist(self) -> List[str]:
        """Generate finishing checklist for acrylic painting."""
        return [
            "Step back and view painting from a distance - does it read well?",
            "Check for any areas that need final highlights or adjustments",
            "Ensure all edges are clean and intentional",
            "Sign your work once fully satisfied",
            "Allow painting to cure fully (24-48 hours) before varnishing",
            "Consider adding isolation coat before varnish (optional)",
        ]

    def _generate_care_instructions(self) -> List[str]:
        """Generate care instructions for finished acrylic painting."""
        return [
            "Acrylic paintings are water-resistant when dry but not waterproof",
            "Apply varnish after 2-4 weeks for protection (removable varnish recommended)",
            "Clean surface with slightly damp cloth - avoid solvents",
            "Store away from direct sunlight and extreme temperatures",
            "Handle by edges to avoid fingerprints on surface",
        ]

    def _generate_next_steps(self, lesson: LessonPlan) -> List[str]:
        """Generate suggested next steps for learning."""
        steps = [
            "Practice the techniques from this lesson on smaller studies",
            "Try the same subject with a different color palette",
            "Experiment with different surface textures",
        ]

        if lesson.skill_level == "beginner":
            steps.extend([
                "Work on color mixing exercises to build confidence",
                "Try painting simple objects from life to train observation",
            ])
        elif lesson.skill_level == "intermediate":
            steps.extend([
                "Challenge yourself with more complex compositions",
                "Experiment with mixed media techniques",
            ])

        return steps
