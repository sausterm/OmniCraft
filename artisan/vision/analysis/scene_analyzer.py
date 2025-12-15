"""
Advanced Scene Analyzer

Analyzes images to determine:
1. Scene type (landscape, seascape, cityscape, night, aurora, portrait, still life)
2. Light sources (diffuse, point, emission) and their locations
3. Depth layers (background, mid-ground, foreground)
4. Region lighting roles (emitter, receiver, silhouette, reflector, shadow)

This information drives the optimal painting strategy for any scene.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum

from .scene_context import SceneContext, TimeOfDay, Weather, LightingType


class SceneType(Enum):
    """Types of scenes we can detect and handle."""
    LANDSCAPE = "landscape"
    SEASCAPE = "seascape"
    CITYSCAPE_DAY = "cityscape_day"
    CITYSCAPE_NIGHT = "cityscape_night"
    NIGHT_SKY = "night_sky"
    AURORA = "aurora"
    PORTRAIT = "portrait"
    STILL_LIFE = "still_life"
    INTERIOR = "interior"
    ABSTRACT = "abstract"
    UNKNOWN = "unknown"


class LightSourceType(Enum):
    """Types of light sources."""
    DIFFUSE = "diffuse"          # Large, soft (sky, overcast, window)
    POINT = "point"              # Small, intense (sun, lamp, candle)
    EMISSION = "emission"        # Glowing (aurora, neon, fire, screen)
    AMBIENT = "ambient"          # Overall fill light


class LightingRole(Enum):
    """How a region relates to light in the scene."""
    EMITTER = "emitter"          # IS a light source - paint glow first
    RECEIVER_LIT = "receiver_lit"  # Directly lit - dark to light
    RECEIVER_PARTIAL = "receiver_partial"  # Partially lit
    SILHOUETTE = "silhouette"    # Backlit, dark against light
    REFLECTOR = "reflector"      # Reflects light (water, glass, metal)
    SHADOW = "shadow"            # In shadow, blocked from light
    NEUTRAL = "neutral"          # Standard lighting


class DepthLayer(Enum):
    """Spatial depth categories."""
    BACKGROUND = "background"      # Furthest (sky, distant mountains)
    FAR_MIDGROUND = "far_midground"  # Far middle (distant trees, buildings)
    NEAR_MIDGROUND = "near_midground"  # Near middle
    FOREGROUND = "foreground"      # Closest to viewer
    FOCAL = "focal"                # Main subject (painted last with most detail)


class ValueProgression(Enum):
    """How to progress through values when painting a region."""
    DARK_TO_LIGHT = "dark_to_light"      # Standard: shadows first, highlights last
    LIGHT_TO_DARK = "light_to_dark"      # Emission: glow first, then edges
    GLOW_THEN_EDGES = "glow_then_edges"  # Aurora/fire: bright core, dark edges
    SILHOUETTE_RIM = "silhouette_rim"    # Dark mass, then rim highlights
    REFLECTION = "reflection"             # Base, then reflected image (muted)
    MUTED_FLAT = "muted_flat"            # Shadow areas: low contrast


@dataclass
class LightSource:
    """Detected light source in the scene."""
    type: LightSourceType
    position: Tuple[int, int]  # Center position (y, x)
    region_mask: Optional[np.ndarray]  # Mask of the light source region
    intensity: float  # 0-1
    color_temperature: float  # 0=cool, 1=warm
    direction_from_center: Tuple[float, float]  # Normalized direction vector
    is_primary: bool = False


@dataclass
class RegionAnalysis:
    """Analysis results for a single region."""
    name: str
    mask: np.ndarray
    depth_layer: DepthLayer
    lighting_role: LightingRole
    value_progression: ValueProgression
    avg_luminosity: float
    is_light_source: bool = False
    light_source_type: Optional[LightSourceType] = None
    receives_light_from: List[str] = field(default_factory=list)
    casts_shadow_on: List[str] = field(default_factory=list)
    painting_priority: int = 0  # Lower = paint first

    # Additional properties for painting
    suggested_techniques: List[str] = field(default_factory=list)
    edge_treatment: str = "soft"  # soft, hard, mixed
    color_notes: List[str] = field(default_factory=list)


@dataclass
class SceneAnalysisResult:
    """Complete scene analysis results."""
    scene_type: SceneType
    scene_context: SceneContext
    light_sources: List[LightSource]
    primary_light_direction: Tuple[float, float]  # Normalized (y, x) vector
    is_backlit: bool
    depth_layers: Dict[DepthLayer, List[str]]  # Layer -> region names
    region_analyses: Dict[str, RegionAnalysis]  # Region name -> analysis
    painting_order: List[str]  # Ordered list of region names
    notes: List[str] = field(default_factory=list)


class SceneAnalyzer:
    """
    Analyzes scenes to determine optimal painting strategies.

    Works for any scene type: landscapes, seascapes, cityscapes,
    night scenes, aurora, portraits, still life, etc.
    """

    def __init__(self, image: np.ndarray, scene_context: Optional[SceneContext] = None):
        """
        Initialize with an image.

        Args:
            image: RGB image as numpy array
            scene_context: Optional pre-computed scene context
        """
        self.image = image
        self.height, self.width = image.shape[:2]
        self.center = (self.height // 2, self.width // 2)

        # Compute basic image properties
        self.gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.luminosity = self.gray.astype(float) / 255.0
        self.hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Scene context (detect if not provided)
        if scene_context is None:
            from .scene_context import analyze_scene_context
            self.scene_context = analyze_scene_context(image)
        else:
            self.scene_context = scene_context

        # Will be populated during analysis
        self.scene_type: Optional[SceneType] = None
        self.light_sources: List[LightSource] = []
        self.region_analyses: Dict[str, RegionAnalysis] = {}

    def analyze(self, regions: Dict[str, np.ndarray]) -> SceneAnalysisResult:
        """
        Perform complete scene analysis.

        Args:
            regions: Dictionary mapping region names to binary masks

        Returns:
            SceneAnalysisResult with all analysis data
        """
        # Step 1: Detect scene type
        self.scene_type = self._detect_scene_type(regions)

        # Step 2: Detect light sources
        self.light_sources = self._detect_light_sources(regions)

        # Step 3: Determine primary light direction
        primary_direction = self._compute_primary_light_direction()
        is_backlit = self._is_scene_backlit(primary_direction)

        # Step 4: Classify depth layers for each region
        depth_assignments = self._assign_depth_layers(regions)

        # Step 5: Analyze each region's lighting role
        for name, mask in regions.items():
            self.region_analyses[name] = self._analyze_region(
                name, mask, depth_assignments.get(name, DepthLayer.FOREGROUND),
                primary_direction, is_backlit
            )

        # Step 6: Determine painting order
        painting_order = self._compute_painting_order()

        # Step 7: Compile depth layers dict
        depth_layers = {}
        for name, analysis in self.region_analyses.items():
            layer = analysis.depth_layer
            if layer not in depth_layers:
                depth_layers[layer] = []
            depth_layers[layer].append(name)

        return SceneAnalysisResult(
            scene_type=self.scene_type,
            scene_context=self.scene_context,
            light_sources=self.light_sources,
            primary_light_direction=primary_direction,
            is_backlit=is_backlit,
            depth_layers=depth_layers,
            region_analyses=self.region_analyses,
            painting_order=painting_order,
            notes=self._generate_scene_notes()
        )

    def _detect_scene_type(self, regions: Dict[str, np.ndarray]) -> SceneType:
        """Detect the type of scene from image features and regions."""

        region_names_lower = [n.lower() for n in regions.keys()]

        # Check for aurora (distinctive green/purple in upper portion)
        if self._has_aurora_colors():
            return SceneType.AURORA

        # Check for night scene
        if self.scene_context.time_of_day == TimeOfDay.NIGHT:
            if self._has_city_lights():
                return SceneType.CITYSCAPE_NIGHT
            return SceneType.NIGHT_SKY

        # Check for portrait (face detection or person-dominant)
        if any('person' in n or 'face' in n for n in region_names_lower):
            person_coverage = sum(
                np.sum(mask) for name, mask in regions.items()
                if 'person' in name.lower()
            ) / (self.height * self.width)
            if person_coverage > 0.15:
                return SceneType.PORTRAIT

        # Check for water (seascape)
        if any('water' in n or 'ocean' in n or 'sea' in n or 'lake' in n
               for n in region_names_lower):
            return SceneType.SEASCAPE

        # Check for buildings (cityscape)
        if any('building' in n or 'house' in n or 'road' in n
               for n in region_names_lower):
            return SceneType.CITYSCAPE_DAY

        # Check for sky in upper portion (landscape)
        if any('sky' in n for n in region_names_lower):
            return SceneType.LANDSCAPE

        # Check for indoor/still life indicators
        if self._appears_indoor():
            if self._has_distinct_objects():
                return SceneType.STILL_LIFE
            return SceneType.INTERIOR

        # Default to landscape for outdoor scenes
        if self.luminosity.mean() > 0.3:
            return SceneType.LANDSCAPE

        return SceneType.UNKNOWN

    def _has_aurora_colors(self) -> bool:
        """Check for aurora-like colors (green/cyan/purple bands in dark sky)."""
        # Check upper third of image
        upper_third = self.hsv[:self.height // 3, :, :]

        # Aurora greens are around hue 60-90 (in OpenCV's 0-180 scale: 30-45)
        # Aurora purples/pinks are around hue 270-300 (135-150)
        hue = upper_third[:, :, 0]
        sat = upper_third[:, :, 1]
        val = upper_third[:, :, 2]

        # Look for saturated greens or purples in dark areas
        green_aurora = ((hue >= 30) & (hue <= 50) & (sat > 100) & (val > 50) & (val < 200))
        purple_aurora = ((hue >= 130) & (hue <= 160) & (sat > 80) & (val > 50) & (val < 200))

        aurora_pixels = np.sum(green_aurora) + np.sum(purple_aurora)
        total_pixels = upper_third.shape[0] * upper_third.shape[1]

        return aurora_pixels / total_pixels > 0.05  # 5% threshold

    def _has_city_lights(self) -> bool:
        """Check for point light sources typical of city at night."""
        # Find bright spots in dark image
        if self.luminosity.mean() > 0.3:
            return False

        bright_spots = self.luminosity > 0.7
        num_bright = np.sum(bright_spots)

        # Use connected components to count distinct lights
        bright_uint8 = (bright_spots * 255).astype(np.uint8)
        num_labels, _ = cv2.connectedComponents(bright_uint8)

        # City scenes have many small bright spots
        return num_labels > 10 and num_bright / (self.height * self.width) < 0.1

    def _appears_indoor(self) -> bool:
        """Heuristic check if scene appears to be indoors."""
        # Indoor scenes typically have less sky, more uniform lighting
        upper_luminosity = self.luminosity[:self.height // 4, :].mean()
        overall_luminosity = self.luminosity.mean()

        # Sky typically brighter than rest of scene
        # Indoor: upper portion similar to rest
        return abs(upper_luminosity - overall_luminosity) < 0.1

    def _has_distinct_objects(self) -> bool:
        """Check if scene has distinct foreground objects (still life)."""
        # Use edge detection to find object boundaries
        edges = cv2.Canny(self.gray, 50, 150)
        edge_density = np.sum(edges > 0) / (self.height * self.width)

        # Still life has moderate edge density (distinct objects)
        return 0.02 < edge_density < 0.15

    def _detect_light_sources(self, regions: Dict[str, np.ndarray]) -> List[LightSource]:
        """Detect light sources in the scene."""
        light_sources = []

        # 1. Check for diffuse sky light
        for name, mask in regions.items():
            if 'sky' in name.lower():
                if np.sum(mask) > 0:
                    sky_luminosity = self.luminosity[mask].mean()
                    if sky_luminosity > 0.4:  # Bright enough to be light source
                        # Find center of sky region
                        ys, xs = np.where(mask)
                        center = (int(np.mean(ys)), int(np.mean(xs)))

                        # Color temperature from sky color
                        sky_color = self.image[mask].mean(axis=0)
                        color_temp = self._estimate_color_temperature(sky_color)

                        light_sources.append(LightSource(
                            type=LightSourceType.DIFFUSE,
                            position=center,
                            region_mask=mask.copy(),
                            intensity=sky_luminosity,
                            color_temperature=color_temp,
                            direction_from_center=self._direction_from_center(center),
                            is_primary=True
                        ))

        # 2. Check for emission sources (aurora, neon, fire)
        if self.scene_type == SceneType.AURORA:
            aurora_mask = self._find_aurora_regions()
            if aurora_mask is not None and np.sum(aurora_mask) > 0:
                ys, xs = np.where(aurora_mask)
                center = (int(np.mean(ys)), int(np.mean(xs)))
                light_sources.append(LightSource(
                    type=LightSourceType.EMISSION,
                    position=center,
                    region_mask=aurora_mask,
                    intensity=0.7,
                    color_temperature=0.3,  # Cool (green/purple)
                    direction_from_center=self._direction_from_center(center),
                    is_primary=True
                ))

        # 3. Check for point light sources (bright spots)
        point_lights = self._find_point_lights()
        light_sources.extend(point_lights)

        # 4. If no light sources found, assume ambient
        if not light_sources:
            light_sources.append(LightSource(
                type=LightSourceType.AMBIENT,
                position=self.center,
                region_mask=None,
                intensity=self.luminosity.mean(),
                color_temperature=0.5,
                direction_from_center=(0, 0),
                is_primary=True
            ))

        # Mark the brightest as primary if not already set
        if not any(ls.is_primary for ls in light_sources):
            brightest = max(light_sources, key=lambda ls: ls.intensity)
            brightest.is_primary = True

        return light_sources

    def _find_aurora_regions(self) -> Optional[np.ndarray]:
        """Find aurora-colored regions."""
        hue = self.hsv[:, :, 0]
        sat = self.hsv[:, :, 1]
        val = self.hsv[:, :, 2]

        # Aurora greens and purples
        green = (hue >= 30) & (hue <= 50) & (sat > 100)
        purple = (hue >= 130) & (hue <= 160) & (sat > 80)

        aurora = green | purple

        if np.sum(aurora) > 100:
            return aurora
        return None

    def _find_point_lights(self) -> List[LightSource]:
        """Find small, bright point light sources."""
        lights = []

        # Find very bright spots
        very_bright = self.luminosity > 0.85

        if np.sum(very_bright) == 0:
            return lights

        # Find connected components
        bright_uint8 = (very_bright * 255).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bright_uint8)

        for i in range(1, num_labels):  # Skip background (0)
            area = stats[i, cv2.CC_STAT_AREA]

            # Point lights are small
            if area < (self.height * self.width * 0.02):  # Less than 2% of image
                center = (int(centroids[i][1]), int(centroids[i][0]))
                mask = labels == i

                intensity = self.luminosity[mask].mean()
                color = self.image[mask].mean(axis=0)
                color_temp = self._estimate_color_temperature(color)

                lights.append(LightSource(
                    type=LightSourceType.POINT,
                    position=center,
                    region_mask=mask,
                    intensity=intensity,
                    color_temperature=color_temp,
                    direction_from_center=self._direction_from_center(center),
                    is_primary=False
                ))

        return lights

    def _estimate_color_temperature(self, rgb: np.ndarray) -> float:
        """Estimate color temperature from RGB. 0=cool/blue, 1=warm/orange."""
        r, g, b = rgb[0], rgb[1], rgb[2]

        # Simple heuristic: more red = warmer, more blue = cooler
        if r + g + b == 0:
            return 0.5

        warmth = (r - b) / (r + g + b + 1)
        return np.clip((warmth + 1) / 2, 0, 1)

    def _direction_from_center(self, pos: Tuple[int, int]) -> Tuple[float, float]:
        """Compute normalized direction vector from image center to position."""
        dy = pos[0] - self.center[0]
        dx = pos[1] - self.center[1]

        magnitude = np.sqrt(dy**2 + dx**2)
        if magnitude == 0:
            return (0.0, 0.0)

        return (dy / magnitude, dx / magnitude)

    def _compute_primary_light_direction(self) -> Tuple[float, float]:
        """Compute the primary light direction from detected sources."""
        if not self.light_sources:
            return (0.0, 0.0)

        # Weight by intensity and primary status
        total_dy, total_dx = 0.0, 0.0
        total_weight = 0.0

        for ls in self.light_sources:
            weight = ls.intensity * (2.0 if ls.is_primary else 1.0)
            total_dy += ls.direction_from_center[0] * weight
            total_dx += ls.direction_from_center[1] * weight
            total_weight += weight

        if total_weight == 0:
            return (0.0, 0.0)

        dy = total_dy / total_weight
        dx = total_dx / total_weight

        # Normalize
        magnitude = np.sqrt(dy**2 + dx**2)
        if magnitude == 0:
            return (0.0, 0.0)

        return (dy / magnitude, dx / magnitude)

    def _is_scene_backlit(self, light_direction: Tuple[float, float]) -> bool:
        """Determine if the scene is backlit (light coming from behind subjects)."""
        # Backlit = light source is in the upper portion of the image
        # (behind the subjects, toward the camera)

        # Check if primary light sources are in upper half
        for ls in self.light_sources:
            if ls.is_primary:
                # If light is above center and bright, it's likely backlit
                if ls.position[0] < self.height // 2 and ls.intensity > 0.5:
                    return True

        # Also check scene context
        return self.scene_context.lighting == LightingType.BACK_LIT

    def _assign_depth_layers(self, regions: Dict[str, np.ndarray]) -> Dict[str, DepthLayer]:
        """Assign depth layers to each region based on position and characteristics."""
        assignments = {}

        for name, mask in regions.items():
            if np.sum(mask) == 0:
                assignments[name] = DepthLayer.FOREGROUND
                continue

            name_lower = name.lower()

            # Get vertical position (higher = further back in most scenes)
            ys, _ = np.where(mask)
            avg_y = np.mean(ys)
            relative_y = avg_y / self.height

            # Get coverage
            coverage = np.sum(mask) / (self.height * self.width)

            # Rule-based assignment with semantic hints
            if 'sky' in name_lower:
                assignments[name] = DepthLayer.BACKGROUND
            elif 'mountain' in name_lower or 'distant' in name_lower:
                assignments[name] = DepthLayer.FAR_MIDGROUND
            elif 'person' in name_lower or 'dog' in name_lower or 'cat' in name_lower:
                # Animals/people are often focal points
                assignments[name] = DepthLayer.FOCAL
            elif 'water' in name_lower:
                # Water can be mid or foreground depending on position
                if relative_y < 0.5:
                    assignments[name] = DepthLayer.FAR_MIDGROUND
                else:
                    assignments[name] = DepthLayer.FOREGROUND
            elif relative_y < 0.33:
                # Upper third = background
                assignments[name] = DepthLayer.BACKGROUND
            elif relative_y < 0.55:
                # Upper-middle = far midground
                assignments[name] = DepthLayer.FAR_MIDGROUND
            elif relative_y < 0.75:
                # Lower-middle = near midground
                assignments[name] = DepthLayer.NEAR_MIDGROUND
            else:
                # Bottom = foreground
                assignments[name] = DepthLayer.FOREGROUND

        return assignments

    def _analyze_region(
        self,
        name: str,
        mask: np.ndarray,
        depth_layer: DepthLayer,
        light_direction: Tuple[float, float],
        is_backlit: bool
    ) -> RegionAnalysis:
        """Analyze a single region's lighting role and painting requirements."""

        if np.sum(mask) == 0:
            return RegionAnalysis(
                name=name,
                mask=mask,
                depth_layer=depth_layer,
                lighting_role=LightingRole.NEUTRAL,
                value_progression=ValueProgression.DARK_TO_LIGHT,
                avg_luminosity=0.0
            )

        avg_luminosity = self.luminosity[mask].mean()
        name_lower = name.lower()

        # Check if this region is a light source
        is_light_source = False
        light_source_type = None

        for ls in self.light_sources:
            if ls.region_mask is not None:
                overlap = np.sum(mask & ls.region_mask) / max(1, np.sum(mask))
                if overlap > 0.5:
                    is_light_source = True
                    light_source_type = ls.type
                    break

        # Determine lighting role
        lighting_role = self._determine_lighting_role(
            name_lower, mask, depth_layer, avg_luminosity,
            is_light_source, is_backlit
        )

        # Determine value progression based on lighting role
        value_progression = self._determine_value_progression(lighting_role, light_source_type)

        # Determine painting priority (lower = paint first)
        priority = self._compute_painting_priority(depth_layer, lighting_role, is_light_source)

        # Generate technique suggestions
        techniques = self._suggest_techniques(name_lower, lighting_role, depth_layer)

        # Determine edge treatment
        edge_treatment = self._determine_edge_treatment(name_lower, lighting_role)

        # Color notes
        color_notes = self._generate_color_notes(lighting_role, is_backlit, depth_layer)

        return RegionAnalysis(
            name=name,
            mask=mask,
            depth_layer=depth_layer,
            lighting_role=lighting_role,
            value_progression=value_progression,
            avg_luminosity=avg_luminosity,
            is_light_source=is_light_source,
            light_source_type=light_source_type,
            painting_priority=priority,
            suggested_techniques=techniques,
            edge_treatment=edge_treatment,
            color_notes=color_notes
        )

    def _determine_lighting_role(
        self,
        name_lower: str,
        mask: np.ndarray,
        depth_layer: DepthLayer,
        avg_luminosity: float,
        is_light_source: bool,
        is_backlit: bool
    ) -> LightingRole:
        """Determine how a region relates to light in the scene."""

        # Emitter: IS the light source
        if is_light_source:
            return LightingRole.EMITTER

        # Check for reflective surfaces
        if any(word in name_lower for word in ['water', 'lake', 'ocean', 'glass', 'mirror', 'metal']):
            return LightingRole.REFLECTOR

        # In backlit scenes
        if is_backlit:
            if depth_layer == DepthLayer.BACKGROUND:
                # Background in backlit = usually the light source or near it
                if avg_luminosity > 0.5:
                    return LightingRole.EMITTER
            elif depth_layer in [DepthLayer.FOREGROUND, DepthLayer.NEAR_MIDGROUND, DepthLayer.FOCAL]:
                # Foreground in backlit = silhouette
                return LightingRole.SILHOUETTE
            else:
                # Midground in backlit = partial silhouette
                if avg_luminosity < 0.4:
                    return LightingRole.SILHOUETTE
                return LightingRole.RECEIVER_PARTIAL

        # Standard lighting analysis
        overall_luminosity = self.luminosity.mean()

        if avg_luminosity < overall_luminosity * 0.6:
            return LightingRole.SHADOW
        elif avg_luminosity > overall_luminosity * 1.2:
            return LightingRole.RECEIVER_LIT
        else:
            return LightingRole.RECEIVER_PARTIAL

    def _determine_value_progression(
        self,
        lighting_role: LightingRole,
        light_source_type: Optional[LightSourceType]
    ) -> ValueProgression:
        """Determine how to progress through values when painting."""

        if lighting_role == LightingRole.EMITTER:
            if light_source_type == LightSourceType.EMISSION:
                return ValueProgression.GLOW_THEN_EDGES
            return ValueProgression.LIGHT_TO_DARK

        elif lighting_role == LightingRole.SILHOUETTE:
            return ValueProgression.SILHOUETTE_RIM

        elif lighting_role == LightingRole.REFLECTOR:
            return ValueProgression.REFLECTION

        elif lighting_role == LightingRole.SHADOW:
            return ValueProgression.MUTED_FLAT

        else:
            return ValueProgression.DARK_TO_LIGHT

    def _compute_painting_priority(
        self,
        depth_layer: DepthLayer,
        lighting_role: LightingRole,
        is_light_source: bool
    ) -> int:
        """Compute painting priority (lower = paint first)."""

        # Base priority from depth
        depth_priority = {
            DepthLayer.BACKGROUND: 0,
            DepthLayer.FAR_MIDGROUND: 100,
            DepthLayer.NEAR_MIDGROUND: 200,
            DepthLayer.FOREGROUND: 300,
            DepthLayer.FOCAL: 400
        }.get(depth_layer, 200)

        # Adjust for lighting role
        # Light sources painted first within their depth layer
        if is_light_source or lighting_role == LightingRole.EMITTER:
            depth_priority -= 50

        # Focal subjects always last
        if depth_layer == DepthLayer.FOCAL:
            depth_priority += 100

        return depth_priority

    def _suggest_techniques(
        self,
        name_lower: str,
        lighting_role: LightingRole,
        depth_layer: DepthLayer
    ) -> List[str]:
        """Suggest painting techniques based on region characteristics."""
        techniques = []

        # Based on lighting role
        if lighting_role == LightingRole.EMITTER:
            techniques.append("Establish glow with soft edges first")
            techniques.append("Build from center of light outward")

        elif lighting_role == LightingRole.SILHOUETTE:
            techniques.append("Block in dark mass as a shape")
            techniques.append("Add rim lighting on edges last")

        elif lighting_role == LightingRole.REFLECTOR:
            techniques.append("Paint base color first")
            techniques.append("Add reflection as muted version of source")
            techniques.append("Keep reflection slightly darker and less saturated")

        # Based on depth
        if depth_layer == DepthLayer.BACKGROUND:
            techniques.append("Keep edges soft and values close")
            techniques.append("Less detail - atmospheric perspective")

        elif depth_layer == DepthLayer.FOCAL:
            techniques.append("Most detail and contrast here")
            techniques.append("Sharp edges where appropriate")

        return techniques

    def _determine_edge_treatment(
        self,
        name_lower: str,
        lighting_role: LightingRole
    ) -> str:
        """Determine how to treat edges of this region."""

        # Hard edges for architecture
        if any(word in name_lower for word in ['building', 'house', 'road', 'vehicle']):
            return "hard"

        # Soft edges for atmosphere, glow
        if lighting_role == LightingRole.EMITTER:
            return "soft"

        # Mixed for silhouettes (soft against light, defined against dark)
        if lighting_role == LightingRole.SILHOUETTE:
            return "mixed"

        # Default soft for organic forms
        return "soft"

    def _generate_color_notes(
        self,
        lighting_role: LightingRole,
        is_backlit: bool,
        depth_layer: DepthLayer
    ) -> List[str]:
        """Generate color guidance notes."""
        notes = []

        if lighting_role == LightingRole.SHADOW:
            notes.append("Use cool colors (blue/purple) in shadows")
            notes.append("Reduce saturation in shadow areas")

        if lighting_role == LightingRole.SILHOUETTE:
            notes.append("Keep values dark, minimal color variation")
            notes.append("Warm rim highlights from backlight")

        if is_backlit and depth_layer == DepthLayer.BACKGROUND:
            notes.append("Warm colors around light source")
            notes.append("Gradient from warm (center) to cool (edges)")

        if depth_layer in [DepthLayer.BACKGROUND, DepthLayer.FAR_MIDGROUND]:
            notes.append("Cooler, bluer colors for atmospheric perspective")
            notes.append("Lower saturation and contrast")

        return notes

    def _compute_painting_order(self) -> List[str]:
        """Compute optimal painting order for all regions."""

        # Sort by painting priority
        sorted_regions = sorted(
            self.region_analyses.items(),
            key=lambda x: x[1].painting_priority
        )

        return [name for name, _ in sorted_regions]

    def _generate_scene_notes(self) -> List[str]:
        """Generate overall notes about the scene."""
        notes = []

        notes.append(f"Scene type: {self.scene_type.value}")

        if self.scene_type == SceneType.AURORA:
            notes.append("Aurora: Paint dark sky base first, then build up the glow")

        if any(ls.type == LightSourceType.POINT for ls in self.light_sources):
            notes.append("Multiple point light sources - consider each light's influence")

        if self._is_scene_backlit((0, 0)):
            notes.append("Backlit scene: Foreground subjects will be silhouetted")
            notes.append("Establish bright background first, then dark silhouettes")

        return notes
