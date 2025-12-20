#!/usr/bin/env python3
"""
Artisan Lesson Generator CLI

Generate comprehensive art lessons from images using semantic understanding.

Usage:
    python -m artisan.cli.lesson <image_path> [options]

Examples:
    # Basic usage - generates acrylic lesson
    python -m artisan.cli.lesson dog.jpg

    # Specify medium and style
    python -m artisan.cli.lesson portrait.jpg --medium oil --style realism --skill advanced

    # Output to specific directory
    python -m artisan.cli.lesson landscape.jpg -o ./my_lesson

    # Analyze image without generating lesson
    python -m artisan.cli.lesson photo.jpg --analyze-only
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic-aware art lessons from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s dog.jpg
  %(prog)s portrait.jpg --medium oil --style realism
  %(prog)s landscape.jpg -o ./output --skill beginner
  %(prog)s photo.jpg --analyze-only
        """
    )

    parser.add_argument(
        "image",
        help="Path to source image"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory (default: ./output/<image_name>)",
        default=None
    )

    parser.add_argument(
        "--medium",
        choices=["acrylic", "oil", "watercolor", "gouache", "colored_pencil", "graphite"],
        default="acrylic",
        help="Art medium (default: acrylic)"
    )

    parser.add_argument(
        "--style",
        choices=["painterly", "realism", "impressionism", "loose", "tight", "photorealism"],
        default="painterly",
        help="Art style (default: painterly)"
    )

    parser.add_argument(
        "--skill",
        choices=["beginner", "intermediate", "advanced", "expert"],
        default="intermediate",
        help="Skill level (default: intermediate)"
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze image, don't generate lesson"
    )

    parser.add_argument(
        "--use-sam",
        action="store_true",
        default=False,
        help="Use SAM for segmentation (requires torch + segment-anything)"
    )

    parser.add_argument(
        "--render",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output format (default: both)"
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Validate input
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)

    # Set output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("./output") / image_path.stem

    print(f"Artisan Lesson Generator")
    print(f"{'='*50}")
    print(f"Image: {image_path}")
    print(f"Medium: {args.medium}")
    print(f"Style: {args.style}")
    print(f"Skill: {args.skill}")
    print(f"Output: {output_dir}")
    print()

    # Import artisan components
    try:
        from artisan.orchestrator import Artisan
        from artisan.core.constraints import ArtConstraints, Medium, Style, SkillLevel
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running from the artisan directory")
        sys.exit(1)

    # Create artisan instance
    artisan = Artisan(use_sam=args.use_sam)

    # Load image
    from PIL import Image
    import numpy as np
    image = np.array(Image.open(image_path).convert("RGB"))

    if args.analyze_only:
        # Just analyze and print results
        print("Analyzing image...")
        analysis = artisan.analyze_image(image, detailed=True)

        print(f"\nAnalysis Results")
        print(f"{'-'*40}")
        print(f"Detected Domain: {analysis['domain']}")
        print(f"Scene Complexity: {analysis['complexity']:.2f}")
        print(f"\nRecommended Approach:")
        print(f"  {analysis['recommended_approach']}")

        if "segmentation" in analysis:
            seg = analysis["segmentation"]
            print(f"\nSegmentation ({seg['method']}):")
            print(f"  Total segments: {seg['num_segments']}")
            print(f"  Background: {seg['num_background']}")
            print(f"  Subjects: {seg['num_subjects']}")
            print(f"  Details: {seg['num_details']}")

        print(f"\nDetected Subjects:")
        for subject in analysis["subjects"]:
            print(f"  - {subject['category']} ({subject['domain']})")
            print(f"    Coverage: {subject['coverage']*100:.1f}%")
            print(f"    Techniques: {', '.join(subject['techniques'])}")
            print(f"    Focus areas: {', '.join(subject['focus_areas'])}")

        return

    # Create constraints
    constraints = ArtConstraints(
        source_image=image,
        source_path=image_path,
        medium=Medium(args.medium),
        style=Style(args.style),
        skill_level=SkillLevel(args.skill),
    )

    # Generate lesson
    print("Generating lesson plan...")
    try:
        lesson = artisan.create_lesson(constraints, output_dir)
    except Exception as e:
        print(f"Error generating lesson: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Render output
    if args.render in ["json", "both"]:
        json_path = output_dir / "lesson_plan.json"
        print(f"\nSaved JSON: {json_path}")

    if args.render in ["markdown", "both"]:
        md_path = output_dir / "lesson.md"
        render_markdown(lesson, md_path)
        print(f"Saved Markdown: {md_path}")

    # Print summary
    print(f"\nLesson Summary")
    print(f"{'-'*40}")
    print(f"Title: {lesson.title}")
    print(f"Domain: {lesson.subject_domain}")
    print(f"Phases: {len(lesson.phases)}")
    print(f"Total Steps: {lesson.get_total_steps()}")
    print(f"Estimated Time: {lesson.total_duration_minutes} minutes")
    print(f"Sessions: {lesson.recommended_sessions}")
    print(f"\nLearning Objectives:")
    for obj in lesson.learning_objectives[:5]:
        print(f"  - {obj}")

    print(f"\n{'='*50}")
    print(f"Lesson generated successfully!")
    print(f"View full lesson at: {output_dir}")


def render_markdown(lesson, path: Path):
    """Render lesson to markdown format."""
    md = []

    md.append(f"# {lesson.title}\n")
    md.append(f"*{lesson.description}*\n")

    md.append(f"\n## Overview\n")
    md.append(f"- **Medium:** {lesson.target_medium.replace('_', ' ').title()}")
    md.append(f"- **Style:** {lesson.target_style.replace('_', ' ').title()}")
    md.append(f"- **Skill Level:** {lesson.skill_level.replace('_', ' ').title()}")
    md.append(f"- **Subject:** {lesson.subject_domain.replace('_', ' ').title()}")
    md.append(f"- **Estimated Time:** {lesson.total_duration_minutes} minutes")
    md.append(f"- **Recommended Sessions:** {lesson.recommended_sessions}")

    md.append(f"\n## Learning Objectives\n")
    for obj in lesson.learning_objectives:
        md.append(f"- {obj}")

    md.append(f"\n## Materials Needed\n")
    for material in lesson.materials:
        optional = " *(optional)*" if material.optional else ""
        md.append(f"- **{material.name}** ({material.quantity}){optional}")
        md.append(f"  - Purpose: {material.purpose}")
        if material.alternatives:
            md.append(f"  - Alternatives: {', '.join(material.alternatives)}")

    if lesson.preparation_checklist:
        md.append(f"\n## Preparation Checklist\n")
        for item in lesson.preparation_checklist:
            md.append(f"- [ ] {item}")

    if lesson.workspace_setup:
        md.append(f"\n## Workspace Setup\n")
        for item in lesson.workspace_setup:
            md.append(f"- {item}")

    md.append(f"\n---\n")
    md.append(f"\n# Lesson Content\n")

    for phase in lesson.phases:
        md.append(f"\n## Phase {phase.phase_number}: {phase.name}\n")
        md.append(f"*{phase.description}*\n")
        md.append(f"**Estimated time:** {phase.estimated_duration_minutes} minutes\n")

        if phase.overview:
            md.append(f"\n### Overview\n{phase.overview}\n")

        if phase.key_concepts:
            md.append(f"\n**Key Concepts:** {', '.join(phase.key_concepts)}\n")

        for step in phase.steps:
            md.append(f"\n### Step {step.step_number}: {step.title}\n")
            md.append(f"**Objective:** {step.objective}\n")
            md.append(f"**Technique:** {step.technique}\n")
            md.append(f"**Duration:** {step.duration_minutes} minutes\n")
            md.append(f"\n{step.instruction}\n")

            if step.tips:
                md.append(f"\n**Tips:**")
                for tip in step.tips:
                    md.append(f"- {tip}")

            if step.common_mistakes:
                md.append(f"\n**Common Mistakes to Avoid:**")
                for mistake in step.common_mistakes:
                    md.append(f"- {mistake}")

            if step.checkpoints:
                md.append(f"\n**Check before continuing:**")
                for checkpoint in step.checkpoints:
                    md.append(f"- [ ] {checkpoint}")

        if phase.rest_after:
            md.append(f"\n> **Take a break!** This is a good stopping point.\n")

    md.append(f"\n---\n")
    md.append(f"\n*Generated with Artisan - Semantic-Aware Art Construction System*\n")

    # Write file
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(md))


if __name__ == "__main__":
    main()
