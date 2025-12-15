"""
Base Instruction Renderer - Abstract base for medium-specific renderers.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

from artisan.paint.planning.lesson_plan import LessonPlan, LessonStep, LessonPhase
from ..core.construction import Technique, Tool, Material


@dataclass
class RenderedInstruction:
    """A fully rendered instruction for a specific medium."""
    step_number: int
    title: str

    # Detailed instruction text
    instruction_text: str
    sub_instructions: List[str] = field(default_factory=list)

    # Medium-specific details
    paint_consistency: str = ""
    brush_load: str = ""
    stroke_type: str = ""

    # Materials for this step
    paints: List[str] = field(default_factory=list)
    brushes: List[str] = field(default_factory=list)
    other_tools: List[str] = field(default_factory=list)

    # Timing
    working_time: str = ""
    drying_time: str = ""

    # Tips
    tips: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Checkpoints
    checkpoints: List[str] = field(default_factory=list)


@dataclass
class RenderedPhase:
    """A fully rendered phase with medium-specific instructions."""
    phase_number: int
    name: str
    overview: str
    key_concepts: List[str]
    instructions: List[RenderedInstruction]

    # Phase-level guidance
    setup_notes: str = ""
    cleanup_notes: str = ""

    # Timing
    estimated_duration: str = ""
    recommended_breaks: List[str] = field(default_factory=list)


@dataclass
class RenderedLesson:
    """Complete rendered lesson for a specific medium."""
    title: str
    medium: str
    style: str
    skill_level: str

    # Overview
    introduction: str = ""
    learning_objectives: List[str] = field(default_factory=list)

    # Materials
    supply_list: List[Dict[str, Any]] = field(default_factory=list)
    color_mixing_guide: List[Dict[str, Any]] = field(default_factory=list)

    # Setup
    workspace_setup: List[str] = field(default_factory=list)
    preparation_steps: List[str] = field(default_factory=list)

    # Content
    phases: List[RenderedPhase] = field(default_factory=list)

    # Conclusion
    finishing_checklist: List[str] = field(default_factory=list)
    care_instructions: List[str] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)


class InstructionRenderer(ABC):
    """
    Abstract base class for medium-specific instruction rendering.

    Each medium has unique characteristics that affect how instructions
    are written. For example:
    - Acrylic: Fast drying, requires working quickly, can layer immediately
    - Oil: Slow drying, fat-over-lean rule, can blend for hours
    - Watercolor: Light-to-dark, preserve whites, reactivates with water
    """

    @property
    @abstractmethod
    def medium_name(self) -> str:
        """Name of the medium."""
        pass

    @property
    @abstractmethod
    def medium_characteristics(self) -> Dict[str, Any]:
        """Key characteristics that affect instruction rendering."""
        pass

    @abstractmethod
    def render_lesson(self, lesson: LessonPlan) -> RenderedLesson:
        """Render a complete lesson for this medium."""
        pass

    @abstractmethod
    def render_step(self, step: LessonStep) -> RenderedInstruction:
        """Render a single step for this medium."""
        pass

    def get_technique_instruction(self, technique_name: str) -> str:
        """Get medium-specific instruction for a technique."""
        # Override in subclasses for medium-specific guidance
        return f"Apply {technique_name}"

    def get_tool_recommendation(self, abstract_tool: str) -> str:
        """Map abstract tool to medium-specific recommendation."""
        # Override in subclasses
        return abstract_tool

    def get_consistency_guide(self, technique_name: str) -> str:
        """Get paint consistency guide for a technique."""
        return "Medium consistency"

    def _generate_introduction(self, lesson: LessonPlan) -> str:
        """Generate lesson introduction."""
        return (
            f"Welcome to this {lesson.skill_level} {lesson.target_style} "
            f"{self.medium_name} painting lesson. "
            f"In this lesson, you'll learn to create a {lesson.subject_domain} "
            f"using {self.medium_name} techniques. "
            f"The estimated time is {lesson.total_duration_minutes // 60} hours "
            f"across {lesson.recommended_sessions} session(s)."
        )

    def _format_duration(self, minutes: int) -> str:
        """Format duration in human-readable form."""
        if minutes < 60:
            return f"{minutes} minutes"
        hours = minutes // 60
        remaining = minutes % 60
        if remaining == 0:
            return f"{hours} hour{'s' if hours > 1 else ''}"
        return f"{hours} hour{'s' if hours > 1 else ''} {remaining} minutes"
