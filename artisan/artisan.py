"""
Artisan - Main orchestrator for semantic-aware art construction.

This is the primary interface for generating art lessons and instruction sets.
It combines:
- Image perception (segmentation, subject detection)
- Art principles (construction philosophy, technique selection)
- Lesson planning (pedagogical structure)
- Medium-specific rendering

Usage:
    from artisan import Artisan
    from artisan.core.constraints import ArtConstraints, Medium, Style

    artisan = Artisan()
    constraints = ArtConstraints.from_simple(
        image_path="dog.jpg",
        medium="acrylic",
        style="painterly",
        skill="intermediate"
    )

    lesson = artisan.create_lesson(constraints)
    lesson.save("lesson.json")
"""

from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image
import json

from .core.constraints import (
    ArtConstraints, Medium, Style, SkillLevel, SubjectDomain,
    PRESETS
)
from .core.scene_graph import SceneGraph, Entity, EntityType
from .core.art_principles import ArtPrinciplesEngine, ConstructionPhilosophy
from .core.construction import ConstructionPlan

from .perception.semantic import SemanticSegmenter, SegmentationResult
from .perception.scene_builder import SceneGraphBuilder
from .perception.subject_detector import SubjectDetector, SubjectAnalysis

from .planning.lesson_plan import LessonPlan, LessonPlanGenerator


class Artisan:
    """
    Main orchestrator for the Artisan art construction system.

    Artisan takes an image and user constraints, analyzes the image semantically,
    and generates a comprehensive lesson plan for recreating the artwork in the
    specified medium and style.
    """

    def __init__(
        self,
        use_sam: bool = True,
        sam_model: str = "vit_h",
        sam_checkpoint: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize Artisan.

        Args:
            use_sam: Whether to use SAM for segmentation (falls back to classical if unavailable)
            sam_model: SAM model variant ("vit_h", "vit_l", "vit_b")
            sam_checkpoint: Path to SAM checkpoint (downloads if None)
            device: "cuda", "cpu", or "auto"
        """
        self.use_sam = use_sam
        self.sam_model = sam_model
        self.sam_checkpoint = sam_checkpoint
        self.device = device

        # Initialize components lazily
        self._segmenter: Optional[SemanticSegmenter] = None
        self._scene_builder: Optional[SceneGraphBuilder] = None
        self._subject_detector: Optional[SubjectDetector] = None
        self._art_engine: Optional[ArtPrinciplesEngine] = None
        self._lesson_generator: Optional[LessonPlanGenerator] = None

    @property
    def segmenter(self) -> SemanticSegmenter:
        if self._segmenter is None:
            self._segmenter = SemanticSegmenter(
                model_type=self.sam_model,
                checkpoint_path=self.sam_checkpoint,
                device=self.device
            )
        return self._segmenter

    @property
    def scene_builder(self) -> SceneGraphBuilder:
        if self._scene_builder is None:
            self._scene_builder = SceneGraphBuilder()
        return self._scene_builder

    @property
    def subject_detector(self) -> SubjectDetector:
        if self._subject_detector is None:
            self._subject_detector = SubjectDetector()
        return self._subject_detector

    @property
    def art_engine(self) -> ArtPrinciplesEngine:
        if self._art_engine is None:
            self._art_engine = ArtPrinciplesEngine()
        return self._art_engine

    @property
    def lesson_generator(self) -> LessonPlanGenerator:
        if self._lesson_generator is None:
            self._lesson_generator = LessonPlanGenerator()
        return self._lesson_generator

    def analyze_image(
        self,
        image: np.ndarray,
        detailed: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze an image without generating a lesson plan.

        Useful for understanding what's in an image before committing to constraints.

        Args:
            image: RGB image as numpy array
            detailed: Whether to run full segmentation

        Returns:
            Dictionary with analysis results
        """
        results = {}

        # Detect subject domain
        subject_analysis = self.subject_detector.analyze(image)
        results["domain"] = subject_analysis.primary_domain.value
        results["complexity"] = subject_analysis.scene_complexity
        results["recommended_approach"] = subject_analysis.recommended_approach

        if detailed:
            # Run segmentation
            segmentation = self.segmenter.segment(image)
            results["segmentation"] = {
                "method": segmentation.method,
                "num_segments": segmentation.total_segments,
                "num_background": len(segmentation.background_segments),
                "num_subjects": len(segmentation.subject_segments),
                "num_details": len(segmentation.detail_segments),
            }

            # Full subject analysis with segmentation
            subject_analysis = self.subject_detector.analyze(image, segmentation)

        # Subject information
        results["subjects"] = [
            {
                "category": s.category.value,
                "domain": s.domain.value,
                "confidence": s.confidence,
                "coverage": float(np.sum(s.mask) / (image.shape[0] * image.shape[1])),
                "techniques": s.recommended_techniques,
                "challenges": s.key_challenges,
                "focus_areas": s.focus_areas,
            }
            for s in subject_analysis.subjects[:5]  # Top 5
        ]

        return results

    def create_lesson(
        self,
        constraints: ArtConstraints,
        output_dir: Optional[Path] = None
    ) -> LessonPlan:
        """
        Create a complete lesson plan from constraints.

        This is the main entry point for generating art lessons.

        Args:
            constraints: User-defined constraints
            output_dir: Optional directory to save lesson outputs

        Returns:
            Complete LessonPlan
        """
        # Ensure we have an image
        if constraints.source_image is None:
            raise ValueError("Constraints must include a source image")

        image = constraints.source_image

        # Step 1: Segment image
        print("Analyzing image...")
        segmentation = self.segmenter.segment(image)
        print(f"  Found {segmentation.total_segments} regions ({segmentation.method})")

        # Step 2: Build scene graph
        print("Building scene graph...")
        scene_graph = self.scene_builder.build(segmentation, image)
        print(f"  Created {len(scene_graph.entities)} entities")

        # Step 3: Detect subjects
        print("Detecting subjects...")
        subject_analysis = self.subject_detector.analyze(image, segmentation)

        # Auto-detect domain if not specified
        if constraints.subject_domain is None:
            constraints.subject_domain = subject_analysis.primary_domain
            print(f"  Detected domain: {constraints.subject_domain.value}")

        # Step 4: Generate lesson plan
        print("Generating lesson plan...")
        lesson = self.lesson_generator.generate(constraints, scene_graph)

        # Enhance with subject-specific guidance
        self._enhance_lesson_with_subjects(lesson, subject_analysis)

        print(f"  Created {len(lesson.phases)} phases, {lesson.get_total_steps()} steps")
        print(f"  Estimated time: {lesson.total_duration_minutes} minutes")

        # Save if output directory specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Save lesson JSON
            lesson_path = output_dir / "lesson_plan.json"
            lesson.save(lesson_path)
            print(f"  Saved lesson plan to {lesson_path}")

            # Generate and save reference images
            self._generate_reference_images(
                image, segmentation, scene_graph, lesson, output_dir
            )

        return lesson

    def _enhance_lesson_with_subjects(
        self,
        lesson: LessonPlan,
        subject_analysis: SubjectAnalysis
    ):
        """Enhance lesson with subject-specific guidance."""
        # Add subject-specific learning objectives
        for subject in subject_analysis.subjects:
            if subject.key_challenges:
                for challenge in subject.key_challenges:
                    obj = f"Handle {subject.category.value} challenge: {challenge}"
                    if obj not in lesson.learning_objectives:
                        lesson.learning_objectives.append(obj)

        # Update description with approach
        if subject_analysis.recommended_approach:
            lesson.description += f"\n\nRecommended approach: {subject_analysis.recommended_approach}"

    def _generate_reference_images(
        self,
        image: np.ndarray,
        segmentation: SegmentationResult,
        scene_graph: SceneGraph,
        lesson: LessonPlan,
        output_dir: Path
    ):
        """Generate reference images for the lesson."""
        import cv2

        # Save original
        original_path = output_dir / "original.png"
        Image.fromarray(image).save(original_path)
        lesson.reference_images["original"] = str(original_path)

        # Create segmentation visualization
        seg_viz = self._visualize_segmentation(image, segmentation)
        seg_path = output_dir / "segmentation.png"
        Image.fromarray(seg_viz).save(seg_path)
        lesson.reference_images["segmentation"] = str(seg_path)

        # Create entity-highlighted versions for each phase
        steps_dir = output_dir / "steps"
        steps_dir.mkdir(exist_ok=True)

        for phase in lesson.phases:
            for step in phase.steps:
                if step.focus_region_key:
                    entity = scene_graph.get_entity(step.focus_region_key)
                    if entity:
                        # Create highlighted image
                        highlighted = self._highlight_entity(image, entity)
                        step_path = steps_dir / f"step_{step.step_number:03d}.png"
                        Image.fromarray(highlighted).save(step_path)
                        lesson.reference_images[f"step_{step.step_number}"] = str(step_path)

    def _visualize_segmentation(
        self,
        image: np.ndarray,
        segmentation: SegmentationResult
    ) -> np.ndarray:
        """Create a visualization of the segmentation."""
        import cv2

        viz = image.copy()

        # Assign random colors to segments
        np.random.seed(42)
        colors = np.random.randint(0, 255, (len(segmentation.segments), 3))

        for i, segment in enumerate(segmentation.segments):
            color = colors[i]
            mask = segment.mask

            # Create overlay
            overlay = viz.copy()
            overlay[mask] = overlay[mask] * 0.5 + color * 0.5

            viz = overlay.astype(np.uint8)

            # Draw boundary
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(viz, contours, -1, (255, 255, 255), 1)

        return viz

    def _highlight_entity(
        self,
        image: np.ndarray,
        entity: Entity
    ) -> np.ndarray:
        """Create an image with an entity highlighted."""
        import cv2

        viz = image.copy()

        # Dim non-entity areas
        non_mask = ~entity.mask
        viz[non_mask] = (viz[non_mask] * 0.3).astype(np.uint8)

        # Draw boundary
        contours, _ = cv2.findContours(
            entity.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(viz, contours, -1, (255, 200, 0), 2)

        return viz

    @classmethod
    def from_preset(cls, preset_name: str) -> "ArtConstraints":
        """Get constraints from a preset."""
        if preset_name not in PRESETS:
            available = ", ".join(PRESETS.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")
        return PRESETS[preset_name]

    def quick_lesson(
        self,
        image_path: str,
        medium: str = "acrylic",
        style: str = "painterly",
        skill: str = "intermediate",
        output_dir: Optional[str] = None
    ) -> LessonPlan:
        """
        Quick way to generate a lesson from minimal inputs.

        Args:
            image_path: Path to source image
            medium: Art medium (acrylic, oil, watercolor, etc.)
            style: Art style (painterly, realism, impressionism, etc.)
            skill: Skill level (beginner, intermediate, advanced, expert)
            output_dir: Optional output directory

        Returns:
            Complete LessonPlan
        """
        constraints = ArtConstraints.from_simple(
            image_path=image_path,
            medium=medium,
            style=style,
            skill=skill
        )

        output_path = Path(output_dir) if output_dir else None
        return self.create_lesson(constraints, output_path)


# Convenience function for quick usage
def create_lesson(
    image_path: str,
    medium: str = "acrylic",
    style: str = "painterly",
    skill: str = "intermediate",
    output_dir: Optional[str] = None
) -> LessonPlan:
    """
    Convenience function to create a lesson with minimal code.

    Example:
        from artisan import create_lesson
        lesson = create_lesson("dog.jpg", output_dir="./output")
    """
    artisan = Artisan()
    return artisan.quick_lesson(image_path, medium, style, skill, output_dir)
