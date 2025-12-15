"""
Painting Planner

Uses scene analysis to generate optimal painting substeps for each region.
Determines the correct value progression based on the region's lighting role.

This replaces the simple dark-to-light approach with context-aware strategies.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from artisan.vision.analysis.scene_analyzer import (
    SceneAnalyzer, SceneAnalysisResult, RegionAnalysis,
    LightingRole, ValueProgression, DepthLayer, SceneType
)
from artisan.vision.analysis.layering_strategies import LayerSubstep, SubjectType, classify_subject


@dataclass
class PaintingSubstepPlan:
    """A planned painting substep with all necessary information."""
    region_name: str
    substep_name: str
    substep_order: int  # Global order across all regions
    technique: str
    brush: str
    strokes: str
    mask_method: str
    mask_params: Dict
    instruction: str
    tips: List[str]
    value_range: Tuple[float, float]  # (min, max) as percentiles
    is_focal: bool
    lighting_role: LightingRole
    depth_layer: DepthLayer


class PaintingPlanner:
    """
    Generates optimal painting plans based on scene analysis.

    For each region, creates substeps with:
    - Correct value progression (based on lighting role)
    - Appropriate techniques
    - Scene-aware tips and instructions
    """

    def __init__(self, scene_analysis: SceneAnalysisResult):
        self.analysis = scene_analysis

    def generate_plan(
        self,
        regions: Dict[str, np.ndarray],
        subject_types: Dict[str, SubjectType],
        num_substeps_per_region: int = 4
    ) -> List[PaintingSubstepPlan]:
        """
        Generate complete painting plan for all regions.

        Args:
            regions: Dictionary mapping region names to masks
            subject_types: Dictionary mapping region names to subject types
            num_substeps_per_region: Number of substeps per region

        Returns:
            List of PaintingSubstepPlan in painting order
        """
        all_substeps = []
        global_order = 1

        # Process regions in painting order
        for region_name in self.analysis.painting_order:
            if region_name not in regions:
                continue

            region_analysis = self.analysis.region_analyses.get(region_name)
            if region_analysis is None:
                continue

            subject_type = subject_types.get(region_name, SubjectType.UNKNOWN)
            is_focal = region_analysis.depth_layer == DepthLayer.FOCAL

            # Generate substeps for this region
            region_substeps = self._generate_region_substeps(
                region_name=region_name,
                region_analysis=region_analysis,
                subject_type=subject_type,
                is_focal=is_focal,
                num_substeps=num_substeps_per_region if not is_focal else num_substeps_per_region + 1
            )

            # Assign global order
            for substep in region_substeps:
                substep.substep_order = global_order
                global_order += 1

            all_substeps.extend(region_substeps)

        return all_substeps

    def _generate_region_substeps(
        self,
        region_name: str,
        region_analysis: RegionAnalysis,
        subject_type: SubjectType,
        is_focal: bool,
        num_substeps: int
    ) -> List[PaintingSubstepPlan]:
        """Generate substeps for a single region based on its lighting role."""

        value_progression = region_analysis.value_progression
        lighting_role = region_analysis.lighting_role
        depth_layer = region_analysis.depth_layer

        # Get value ranges based on progression type
        value_ranges = self._get_value_ranges(value_progression, num_substeps)

        # Get substep names and techniques based on subject and progression
        substep_configs = self._get_substep_configs(
            subject_type, value_progression, lighting_role, num_substeps
        )

        substeps = []

        for i, (value_range, config) in enumerate(zip(value_ranges, substep_configs)):
            substep = PaintingSubstepPlan(
                region_name=region_name,
                substep_name=f"{region_name} - {config['name']}",
                substep_order=0,  # Will be set later
                technique=config['technique'],
                brush=config['brush'],
                strokes=config['strokes'],
                mask_method=config['mask_method'],
                mask_params={'range': value_range},
                instruction=self._generate_instruction(
                    region_name, config, value_progression, lighting_role, i, num_substeps
                ),
                tips=self._generate_tips(
                    config, lighting_role, depth_layer, i, num_substeps
                ),
                value_range=value_range,
                is_focal=is_focal,
                lighting_role=lighting_role,
                depth_layer=depth_layer
            )
            substeps.append(substep)

        return substeps

    def _get_value_ranges(
        self,
        progression: ValueProgression,
        num_substeps: int
    ) -> List[Tuple[float, float]]:
        """Get value ranges for each substep based on progression type."""

        if progression == ValueProgression.DARK_TO_LIGHT:
            # Standard: dark values first, light values last
            # Each substep covers ~25% of values with overlap
            ranges = []
            step_size = 1.0 / num_substeps
            for i in range(num_substeps):
                start = i * step_size
                end = min(1.0, (i + 1) * step_size + 0.1)  # 10% overlap
                ranges.append((start, end))
            return ranges

        elif progression == ValueProgression.LIGHT_TO_DARK:
            # Emission: light values first (glow), then darker edges
            ranges = []
            step_size = 1.0 / num_substeps
            for i in range(num_substeps):
                # Reverse order: start from bright
                start = 1.0 - (i + 1) * step_size
                end = 1.0 - i * step_size + 0.1
                ranges.append((max(0, start), min(1.0, end)))
            return ranges

        elif progression == ValueProgression.GLOW_THEN_EDGES:
            # Aurora/fire: bright core first, then mid, then dark edges
            if num_substeps == 3:
                return [(0.7, 1.0), (0.35, 0.75), (0.0, 0.4)]
            elif num_substeps == 4:
                return [(0.75, 1.0), (0.5, 0.8), (0.25, 0.55), (0.0, 0.3)]
            else:
                # Default to light-to-dark
                return self._get_value_ranges(ValueProgression.LIGHT_TO_DARK, num_substeps)

        elif progression == ValueProgression.SILHOUETTE_RIM:
            # Silhouette: dark mass first (most of it), then rim highlights
            if num_substeps == 3:
                return [(0.0, 0.3), (0.25, 0.6), (0.7, 1.0)]  # Dark, mid, rim
            elif num_substeps == 4:
                return [(0.0, 0.25), (0.2, 0.45), (0.4, 0.7), (0.75, 1.0)]
            else:
                return [(0.0, 0.5), (0.5, 1.0)]  # Simple: dark mass, then rim

        elif progression == ValueProgression.REFLECTION:
            # Reflections: base, then reflected image
            if num_substeps >= 3:
                return [(0.0, 0.35), (0.3, 0.65), (0.6, 1.0)]
            else:
                return [(0.0, 0.5), (0.4, 1.0)]

        elif progression == ValueProgression.MUTED_FLAT:
            # Shadow areas: compressed value range
            # All substeps work in a narrower range
            if num_substeps >= 3:
                return [(0.0, 0.35), (0.25, 0.55), (0.45, 0.7)]
            else:
                return [(0.0, 0.4), (0.3, 0.6)]

        # Default fallback
        return self._get_value_ranges(ValueProgression.DARK_TO_LIGHT, num_substeps)

    def _get_substep_configs(
        self,
        subject_type: SubjectType,
        progression: ValueProgression,
        lighting_role: LightingRole,
        num_substeps: int
    ) -> List[Dict]:
        """Get configuration for each substep based on subject and progression."""

        # Base configurations by progression type
        if progression == ValueProgression.LIGHT_TO_DARK:
            return self._configs_light_to_dark(subject_type, num_substeps)

        elif progression == ValueProgression.GLOW_THEN_EDGES:
            return self._configs_glow_then_edges(subject_type, num_substeps)

        elif progression == ValueProgression.SILHOUETTE_RIM:
            return self._configs_silhouette_rim(subject_type, num_substeps)

        elif progression == ValueProgression.REFLECTION:
            return self._configs_reflection(subject_type, num_substeps)

        elif progression == ValueProgression.MUTED_FLAT:
            return self._configs_muted_flat(subject_type, num_substeps)

        else:  # DARK_TO_LIGHT (standard)
            return self._configs_dark_to_light(subject_type, num_substeps)

    def _configs_dark_to_light(self, subject_type: SubjectType, n: int) -> List[Dict]:
        """Standard dark-to-light substep configs."""
        if n == 3:
            return [
                {'name': 'Dark Values', 'technique': 'blocking', 'brush': '1-inch brush',
                 'strokes': 'follow the form', 'mask_method': 'luminosity'},
                {'name': 'Mid Values', 'technique': 'layering', 'brush': 'appropriate brush',
                 'strokes': 'build up form', 'mask_method': 'luminosity'},
                {'name': 'Light Values', 'technique': 'highlighting', 'brush': 'fan brush',
                 'strokes': 'light touches', 'mask_method': 'luminosity'},
            ]
        else:  # 4 or more
            return [
                {'name': 'Dark Values', 'technique': 'blocking', 'brush': '1-inch brush',
                 'strokes': 'establish shadows', 'mask_method': 'luminosity'},
                {'name': 'Shadow Mid-tones', 'technique': 'layering', 'brush': '1-inch brush',
                 'strokes': 'build form', 'mask_method': 'luminosity'},
                {'name': 'Light Mid-tones', 'technique': 'layering', 'brush': 'filbert',
                 'strokes': 'blend values', 'mask_method': 'luminosity'},
                {'name': 'Highlights', 'technique': 'highlighting', 'brush': 'fan brush',
                 'strokes': 'sparingly', 'mask_method': 'luminosity'},
            ][:n]

    def _configs_light_to_dark(self, subject_type: SubjectType, n: int) -> List[Dict]:
        """Light-to-dark substep configs (for emitters like sky)."""
        if n == 3:
            return [
                {'name': 'Bright Glow', 'technique': 'blocking', 'brush': '2-inch brush',
                 'strokes': 'soft blending', 'mask_method': 'luminosity'},
                {'name': 'Mid Tones', 'technique': 'layering', 'brush': '1-inch brush',
                 'strokes': 'gradient blending', 'mask_method': 'luminosity'},
                {'name': 'Dark Accents', 'technique': 'detailing', 'brush': 'filbert',
                 'strokes': 'define edges', 'mask_method': 'luminosity'},
            ]
        else:
            return [
                {'name': 'Core Glow', 'technique': 'blocking', 'brush': '2-inch brush',
                 'strokes': 'soft coverage', 'mask_method': 'luminosity'},
                {'name': 'Light Falloff', 'technique': 'layering', 'brush': '2-inch brush',
                 'strokes': 'blend outward', 'mask_method': 'luminosity'},
                {'name': 'Mid Values', 'technique': 'layering', 'brush': '1-inch brush',
                 'strokes': 'transition zones', 'mask_method': 'luminosity'},
                {'name': 'Edge Definition', 'technique': 'detailing', 'brush': 'filbert',
                 'strokes': 'crisp where needed', 'mask_method': 'luminosity'},
            ][:n]

    def _configs_glow_then_edges(self, subject_type: SubjectType, n: int) -> List[Dict]:
        """Glow-then-edges configs (for aurora, fire, neon)."""
        if n == 3:
            return [
                {'name': 'Bright Core', 'technique': 'blocking', 'brush': 'fan brush',
                 'strokes': 'soft dabbing', 'mask_method': 'luminosity'},
                {'name': 'Glow Falloff', 'technique': 'layering', 'brush': 'fan brush',
                 'strokes': 'feathering outward', 'mask_method': 'luminosity'},
                {'name': 'Dark Edges', 'technique': 'detailing', 'brush': '1-inch brush',
                 'strokes': 'define boundaries', 'mask_method': 'luminosity'},
            ]
        else:
            return [
                {'name': 'Brightest Core', 'technique': 'blocking', 'brush': 'fan brush',
                 'strokes': 'soft center glow', 'mask_method': 'luminosity'},
                {'name': 'Inner Glow', 'technique': 'layering', 'brush': 'fan brush',
                 'strokes': 'blend from core', 'mask_method': 'luminosity'},
                {'name': 'Outer Glow', 'technique': 'layering', 'brush': '1-inch brush',
                 'strokes': 'feather edges', 'mask_method': 'luminosity'},
                {'name': 'Dark Boundary', 'technique': 'detailing', 'brush': 'filbert',
                 'strokes': 'define shape', 'mask_method': 'luminosity'},
            ][:n]

    def _configs_silhouette_rim(self, subject_type: SubjectType, n: int) -> List[Dict]:
        """Silhouette with rim light configs (for backlit subjects)."""
        if n == 3:
            return [
                {'name': 'Dark Mass', 'technique': 'blocking', 'brush': '1-inch brush',
                 'strokes': 'fill shape', 'mask_method': 'luminosity'},
                {'name': 'Form Shadows', 'technique': 'layering', 'brush': 'filbert',
                 'strokes': 'subtle variation', 'mask_method': 'luminosity'},
                {'name': 'Rim Light', 'technique': 'highlighting', 'brush': 'script liner',
                 'strokes': 'edge highlights', 'mask_method': 'luminosity'},
            ]
        else:
            return [
                {'name': 'Core Shadow', 'technique': 'blocking', 'brush': '1-inch brush',
                 'strokes': 'establish mass', 'mask_method': 'luminosity'},
                {'name': 'Shadow Variation', 'technique': 'layering', 'brush': 'filbert',
                 'strokes': 'subtle form', 'mask_method': 'luminosity'},
                {'name': 'Edge Transition', 'technique': 'layering', 'brush': 'fan brush',
                 'strokes': 'soft edges', 'mask_method': 'luminosity'},
                {'name': 'Rim Highlights', 'technique': 'highlighting', 'brush': 'script liner',
                 'strokes': 'backlight glow', 'mask_method': 'luminosity'},
            ][:n]

    def _configs_reflection(self, subject_type: SubjectType, n: int) -> List[Dict]:
        """Reflection configs (for water, glass, metal)."""
        if n == 3:
            return [
                {'name': 'Base Color', 'technique': 'blocking', 'brush': '2-inch brush',
                 'strokes': 'horizontal for water', 'mask_method': 'luminosity'},
                {'name': 'Reflected Image', 'technique': 'layering', 'brush': '1-inch brush',
                 'strokes': 'muted, inverted', 'mask_method': 'luminosity'},
                {'name': 'Surface Highlights', 'technique': 'highlighting', 'brush': 'palette knife',
                 'strokes': 'broken sparkle', 'mask_method': 'luminosity'},
            ]
        else:
            return [
                {'name': 'Deep Base', 'technique': 'blocking', 'brush': '2-inch brush',
                 'strokes': 'horizontal coverage', 'mask_method': 'luminosity'},
                {'name': 'Reflection Darks', 'technique': 'layering', 'brush': '1-inch brush',
                 'strokes': 'inverted image', 'mask_method': 'luminosity'},
                {'name': 'Reflection Lights', 'technique': 'layering', 'brush': '1-inch brush',
                 'strokes': 'muted copy', 'mask_method': 'luminosity'},
                {'name': 'Sparkle', 'technique': 'highlighting', 'brush': 'palette knife',
                 'strokes': 'broken horizontal', 'mask_method': 'luminosity'},
            ][:n]

    def _configs_muted_flat(self, subject_type: SubjectType, n: int) -> List[Dict]:
        """Muted/flat configs (for shadow areas)."""
        if n >= 3:
            return [
                {'name': 'Shadow Core', 'technique': 'blocking', 'brush': '1-inch brush',
                 'strokes': 'soft coverage', 'mask_method': 'luminosity'},
                {'name': 'Shadow Variation', 'technique': 'layering', 'brush': 'filbert',
                 'strokes': 'subtle changes', 'mask_method': 'luminosity'},
                {'name': 'Shadow Edge', 'technique': 'blending', 'brush': 'fan brush',
                 'strokes': 'soft transitions', 'mask_method': 'luminosity'},
            ][:n]
        else:
            return [
                {'name': 'Shadow Base', 'technique': 'blocking', 'brush': '1-inch brush',
                 'strokes': 'even coverage', 'mask_method': 'luminosity'},
                {'name': 'Shadow Detail', 'technique': 'layering', 'brush': 'filbert',
                 'strokes': 'minimal variation', 'mask_method': 'luminosity'},
            ]

    def _generate_instruction(
        self,
        region_name: str,
        config: Dict,
        progression: ValueProgression,
        lighting_role: LightingRole,
        step_index: int,
        total_steps: int
    ) -> str:
        """Generate painting instruction for a substep."""

        base_instruction = ""

        if step_index == 0:
            # First step
            if progression == ValueProgression.LIGHT_TO_DARK:
                base_instruction = f"Start with the brightest values in {region_name}. This establishes the light source - the glow that everything else relates to."
            elif progression == ValueProgression.GLOW_THEN_EDGES:
                base_instruction = f"Begin with the glowing core of {region_name}. Use soft strokes to establish where the light is strongest."
            elif progression == ValueProgression.SILHOUETTE_RIM:
                base_instruction = f"Block in the dark mass of {region_name}. This is your silhouette shape against the light."
            else:
                base_instruction = f"Start with the darkest values in {region_name}. Establish your shadows first - they create the foundation."

        elif step_index == total_steps - 1:
            # Last step
            if lighting_role == LightingRole.SILHOUETTE:
                base_instruction = f"Add rim lighting to {region_name}. These edge highlights show the backlight wrapping around the form."
            elif lighting_role == LightingRole.EMITTER:
                base_instruction = f"Define the edges of {region_name}. Keep them soft where the glow fades into darkness."
            else:
                base_instruction = f"Add final highlights to {region_name}. Use sparingly - these are the brightest points where light directly hits."

        else:
            # Middle steps
            base_instruction = f"Build up the {config['name'].lower()} in {region_name}. Work {config['strokes']}."

        return base_instruction

    def _generate_tips(
        self,
        config: Dict,
        lighting_role: LightingRole,
        depth_layer: DepthLayer,
        step_index: int,
        total_steps: int
    ) -> List[str]:
        """Generate tips for a substep."""
        tips = []

        # Lighting role specific tips
        if lighting_role == LightingRole.EMITTER:
            if step_index == 0:
                tips.append("Establish the glow first - this is your light source")
            tips.append("Keep edges soft where light fades")

        elif lighting_role == LightingRole.SILHOUETTE:
            if step_index == 0:
                tips.append("Dark shapes against light - keep values low")
            if step_index == total_steps - 1:
                tips.append("Rim light is warm where backlight hits edges")

        elif lighting_role == LightingRole.REFLECTOR:
            tips.append("Reflections are darker and less saturated than source")
            tips.append("Keep strokes horizontal for water")

        elif lighting_role == LightingRole.SHADOW:
            tips.append("Shadows have color - usually cool blues/purples")
            tips.append("Keep contrast low in shadow areas")

        # Depth layer tips
        if depth_layer == DepthLayer.BACKGROUND:
            tips.append("Less detail - atmospheric perspective softens distant elements")
        elif depth_layer == DepthLayer.FOCAL:
            tips.append("This is your focal point - give it the most attention")

        # Technique tips
        if config['technique'] == 'blocking':
            tips.append("Work quickly to cover the area - don't overwork")
        elif config['technique'] == 'highlighting':
            tips.append("Less is more - highlights are the finishing touch")

        return tips[:3]  # Limit to 3 tips


def create_painting_plan(
    image: np.ndarray,
    regions: Dict[str, np.ndarray],
    subject_types: Dict[str, SubjectType],
    scene_context=None,
    num_substeps: int = 4
) -> Tuple[SceneAnalysisResult, List[PaintingSubstepPlan]]:
    """
    Convenience function to create a complete painting plan.

    Args:
        image: RGB image
        regions: Dictionary mapping region names to masks
        subject_types: Dictionary mapping region names to subject types
        scene_context: Optional pre-computed scene context
        num_substeps: Number of substeps per region

    Returns:
        Tuple of (SceneAnalysisResult, List of PaintingSubstepPlan)
    """
    # Analyze the scene
    analyzer = SceneAnalyzer(image, scene_context)
    analysis = analyzer.analyze(regions)

    # Generate painting plan
    planner = PaintingPlanner(analysis)
    plan = planner.generate_plan(regions, subject_types, num_substeps)

    return analysis, plan
