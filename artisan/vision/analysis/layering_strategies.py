"""
Dynamic Layering Strategies

Defines how each type of subject should be painted based on:
1. What the subject IS (dog, tree, grass, sky, building, water, etc.)
2. The scene context (day/night, weather, mood, lighting)
3. Spatial relationships (back-to-front within the region)

Each strategy returns a list of painting substeps that are context-appropriate.
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .scene_context import SceneContext, TimeOfDay, Weather, LightingType, Mood


@dataclass
class LayerSubstep:
    """A single painting substep within a semantic region."""
    name: str
    technique: str
    mask_method: str  # How to create the mask: "luminosity", "spatial", "edge", "custom"
    mask_params: Dict  # Parameters for mask creation
    brush: str
    stroke: str
    description: str
    tips: List[str]
    priority: int  # Lower = paint first


class SubjectType(Enum):
    """Types of subjects we can detect and have strategies for."""
    DOG = "dog"
    CAT = "cat"
    PERSON = "person"
    BIRD = "bird"
    HORSE = "horse"
    COW = "cow"
    SHEEP = "sheep"
    ANIMAL_OTHER = "animal_other"

    TREE = "tree"
    GRASS = "grass"
    FLOWER = "flower"
    FOLIAGE = "foliage"

    SKY = "sky"
    CLOUD = "cloud"
    WATER = "water"
    MOUNTAIN = "mountain"
    ROCK = "rock"
    GROUND = "ground"
    SAND = "sand"
    SNOW = "snow"

    BUILDING = "building"
    VEHICLE = "vehicle"
    ROAD = "road"

    UNKNOWN = "unknown"


def classify_subject(yolo_class: str) -> SubjectType:
    """Map YOLO class name to our subject type."""
    mapping = {
        "dog": SubjectType.DOG,
        "cat": SubjectType.CAT,
        "person": SubjectType.PERSON,
        "bird": SubjectType.BIRD,
        "horse": SubjectType.HORSE,
        "cow": SubjectType.COW,
        "sheep": SubjectType.SHEEP,

        "tree": SubjectType.TREE,
        "trees": SubjectType.TREE,
        "grass": SubjectType.GRASS,
        "foliage": SubjectType.FOLIAGE,
        "flower": SubjectType.FLOWER,

        "sky": SubjectType.SKY,
        "cloud": SubjectType.CLOUD,
        "clouds": SubjectType.CLOUD,
        "water": SubjectType.WATER,
        "ocean": SubjectType.WATER,
        "lake": SubjectType.WATER,
        "river": SubjectType.WATER,
        "mountain": SubjectType.MOUNTAIN,
        "mountains": SubjectType.MOUNTAIN,
        "rock": SubjectType.ROCK,
        "ground": SubjectType.GROUND,
        "sand": SubjectType.SAND,
        "snow": SubjectType.SNOW,

        "building": SubjectType.BUILDING,
        "house": SubjectType.BUILDING,
        "car": SubjectType.VEHICLE,
        "truck": SubjectType.VEHICLE,
        "bus": SubjectType.VEHICLE,
        "road": SubjectType.ROAD,
    }

    key = yolo_class.lower().strip()
    return mapping.get(key, SubjectType.UNKNOWN)


class LayeringStrategyEngine:
    """
    Generates context-aware painting substeps for any subject type.

    Strategies are organized by:
    1. Subject type (what is it)
    2. Scene context (when/where)
    3. Spatial analysis (back-to-front within region)
    """

    def __init__(self, context: SceneContext):
        self.context = context

    def get_strategy(
        self,
        subject_type: SubjectType,
        region_mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool = False
    ) -> List[LayerSubstep]:
        """
        Get painting substeps for a subject given the context.

        Args:
            subject_type: What type of subject this is
            region_mask: Binary mask for this region
            image: Full RGB image
            is_focal: Whether this is a focal point

        Returns:
            List of LayerSubstep objects in painting order
        """
        # Get the strategy function for this subject type
        strategy_fn = self._get_strategy_function(subject_type)

        # Generate substeps
        substeps = strategy_fn(region_mask, image, is_focal)

        # Adjust for context (pass is_focal for backlit handling)
        substeps = self._adjust_for_context(substeps, subject_type, is_focal)

        return substeps

    def _get_strategy_function(self, subject_type: SubjectType) -> Callable:
        """Get the strategy function for a subject type."""
        strategies = {
            # Animals - fur/feather-based, attention to form
            SubjectType.DOG: self._strategy_animal_fur,
            SubjectType.CAT: self._strategy_animal_fur,
            SubjectType.HORSE: self._strategy_animal_fur,
            SubjectType.COW: self._strategy_animal_fur,
            SubjectType.SHEEP: self._strategy_animal_wool,
            SubjectType.BIRD: self._strategy_bird,
            SubjectType.PERSON: self._strategy_person,
            SubjectType.ANIMAL_OTHER: self._strategy_animal_fur,

            # Vegetation - back-to-front foliage
            SubjectType.TREE: self._strategy_tree,
            SubjectType.GRASS: self._strategy_grass,
            SubjectType.FOLIAGE: self._strategy_foliage,
            SubjectType.FLOWER: self._strategy_flower,

            # Sky/atmosphere
            SubjectType.SKY: self._strategy_sky,
            SubjectType.CLOUD: self._strategy_cloud,

            # Terrain
            SubjectType.WATER: self._strategy_water,
            SubjectType.MOUNTAIN: self._strategy_mountain,
            SubjectType.ROCK: self._strategy_rock,
            SubjectType.GROUND: self._strategy_ground,
            SubjectType.SAND: self._strategy_sand,
            SubjectType.SNOW: self._strategy_snow,

            # Man-made
            SubjectType.BUILDING: self._strategy_building,
            SubjectType.VEHICLE: self._strategy_vehicle,
            SubjectType.ROAD: self._strategy_road,

            SubjectType.UNKNOWN: self._strategy_default,
        }

        return strategies.get(subject_type, self._strategy_default)

    def _adjust_for_context(
        self,
        substeps: List[LayerSubstep],
        subject_type: SubjectType,
        is_focal: bool = False
    ) -> List[LayerSubstep]:
        """Adjust substeps based on scene context."""

        # Night adjustments
        if self.context.time_of_day == TimeOfDay.NIGHT:
            for step in substeps:
                step.tips.append("Keep values darker overall - it's a night scene")
                if "highlight" in step.technique:
                    step.tips.append("Use highlights sparingly - only where moonlight hits")

        # Foggy/overcast adjustments
        if self.context.weather in [Weather.FOGGY, Weather.OVERCAST]:
            for step in substeps:
                step.tips.append("Reduce contrast - the atmosphere softens everything")
                step.tips.append("Edges should be softer in foggy conditions")

        # Snowy adjustments
        if self.context.weather == Weather.SNOWY:
            for step in substeps:
                step.tips.append("Watch your values - snow reflects a lot of light")
                step.tips.append("Snow has subtle color - often blue/purple in shadows")

        # Golden hour adjustments
        if self.context.time_of_day == TimeOfDay.GOLDEN_HOUR:
            for step in substeps:
                step.tips.append("Warm up your colors - golden light touches everything")

        # Backlit adjustments - ONLY for focal subjects that would be silhouettes
        # Environment regions (sky, grass, etc.) should still paint dark-to-light
        if self.context.lighting == LightingType.BACK_LIT and is_focal:
            # Reverse the typical order - silhouette first for focal subjects
            substeps = sorted(substeps, key=lambda s: -s.priority)
            for step in substeps:
                step.tips.append("Subject is backlit - establish the silhouette first")

        return substeps

    # ==================== ANIMAL STRATEGIES ====================

    def _strategy_animal_fur(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for furry animals (dogs, cats, horses)."""
        n_steps = 5 if is_focal else 4

        if n_steps == 5:
            return [
                LayerSubstep(
                    name="Base Form",
                    technique="blocking",
                    mask_method="luminosity",
                    mask_params={"range": (0.0, 0.3)},
                    brush="1-inch landscape brush",
                    stroke="follow the body contours",
                    description="Block in the overall form with the darkest fur tones",
                    tips=["Establish the basic shape", "Think about where shadows fall naturally"],
                    priority=1,
                ),
                LayerSubstep(
                    name="Shadow Fur",
                    technique="layering",
                    mask_method="luminosity",
                    mask_params={"range": (0.2, 0.45)},
                    brush="#6 filbert brush",
                    stroke="short strokes following fur direction",
                    description="Build up the shadowed areas of fur",
                    tips=["Follow the fur growth direction", "Shadows define muscle structure"],
                    priority=2,
                ),
                LayerSubstep(
                    name="Mid-Tone Fur",
                    technique="layering",
                    mask_method="luminosity",
                    mask_params={"range": (0.4, 0.65)},
                    brush="#6 filbert brush",
                    stroke="medium strokes, still following fur",
                    description="Add the main body colors of the fur",
                    tips=["This is the largest area", "Build up texture with varied strokes"],
                    priority=3,
                ),
                LayerSubstep(
                    name="Light Fur",
                    technique="blending",
                    mask_method="luminosity",
                    mask_params={"range": (0.6, 0.85)},
                    brush="fan brush",
                    stroke="light feathering strokes",
                    description="Add lighter fur tones where light catches",
                    tips=["Light catches on top of head, back, haunches", "Keep strokes soft"],
                    priority=4,
                ),
                LayerSubstep(
                    name="Highlights & Details",
                    technique="detailing",
                    mask_method="luminosity",
                    mask_params={"range": (0.8, 1.0)},
                    brush="script liner",
                    stroke="tiny precise strokes",
                    description="Add eye shine, nose highlights, whisker hints",
                    tips=["Eyes are the soul - make them shine", "Less is more with highlights"],
                    priority=5,
                ),
            ]
        else:
            return [
                LayerSubstep(
                    name="Dark Form",
                    technique="blocking",
                    mask_method="luminosity",
                    mask_params={"range": (0.0, 0.35)},
                    brush="1-inch brush",
                    stroke="follow body shape",
                    description="Establish the dark areas and form",
                    tips=["Block in shadows first"],
                    priority=1,
                ),
                LayerSubstep(
                    name="Main Fur",
                    technique="layering",
                    mask_method="luminosity",
                    mask_params={"range": (0.3, 0.65)},
                    brush="#6 filbert",
                    stroke="follow fur direction",
                    description="Build up the main fur tones",
                    tips=["Work in the direction fur grows"],
                    priority=2,
                ),
                LayerSubstep(
                    name="Light Areas",
                    technique="blending",
                    mask_method="luminosity",
                    mask_params={"range": (0.6, 0.85)},
                    brush="fan brush",
                    stroke="soft feathering",
                    description="Add lighter tones where light hits",
                    tips=["Top of back, head, wherever sun hits"],
                    priority=3,
                ),
                LayerSubstep(
                    name="Highlights",
                    technique="highlighting",
                    mask_method="luminosity",
                    mask_params={"range": (0.8, 1.0)},
                    brush="script liner",
                    stroke="tiny touches",
                    description="Final highlights and eye shine",
                    tips=["Sparingly on eyes and brightest spots"],
                    priority=4,
                ),
            ]

    def _strategy_animal_wool(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for woolly animals (sheep)."""
        return [
            LayerSubstep(
                name="Wool Shadows",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.4)},
                brush="fan brush",
                stroke="circular dabbing for wool texture",
                description="Block in the shadowed wool areas",
                tips=["Use circular motions to suggest wool", "Darker in folds and underneath"],
                priority=1,
            ),
            LayerSubstep(
                name="Main Wool",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.35, 0.7)},
                brush="fan brush",
                stroke="small circular dabs",
                description="Build up the fluffy wool texture",
                tips=["Overlap your dabs", "Create clumpy, fluffy texture"],
                priority=2,
            ),
            LayerSubstep(
                name="Bright Wool",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.65, 1.0)},
                brush="fan brush",
                stroke="light touches",
                description="Add the brightest wool highlights",
                tips=["Top of the fleece catches most light", "Don't overwork"],
                priority=3,
            ),
        ]

    def _strategy_bird(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for birds."""
        return [
            LayerSubstep(
                name="Body Shadow",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.4)},
                brush="#6 filbert",
                stroke="follow feather direction",
                description="Block in the darker areas",
                tips=["Feathers overlap like shingles", "Work from body outward"],
                priority=1,
            ),
            LayerSubstep(
                name="Main Plumage",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.35, 0.7)},
                brush="#4 filbert",
                stroke="short strokes suggesting feathers",
                description="Build up the main feather colors",
                tips=["Each stroke can be a feather", "Vary your colors slightly"],
                priority=2,
            ),
            LayerSubstep(
                name="Highlights",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.65, 1.0)},
                brush="script liner",
                stroke="fine detail strokes",
                description="Add feather highlights and eye detail",
                tips=["Wing tips catch light", "Eye sparkle is essential"],
                priority=3,
            ),
        ]

    def _strategy_person(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for people/portraits."""
        n_steps = 5 if is_focal else 4

        steps = [
            LayerSubstep(
                name="Shadow Shapes",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.35)},
                brush="1-inch brush",
                stroke="follow face planes",
                description="Block in the shadow shapes on face/figure",
                tips=["Shadows define structure", "Think about planes of the face"],
                priority=1,
            ),
            LayerSubstep(
                name="Skin Mid-tones",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.3, 0.6)},
                brush="#6 filbert",
                stroke="soft blending strokes",
                description="Build up the flesh tones",
                tips=["Skin has many colors - look carefully", "Blend edges softly"],
                priority=2,
            ),
            LayerSubstep(
                name="Light Skin",
                technique="blending",
                mask_method="luminosity",
                mask_params={"range": (0.55, 0.8)},
                brush="fan brush",
                stroke="gentle blending",
                description="Add lighter areas where light falls",
                tips=["Forehead, nose, cheekbones catch light", "Keep transitions soft"],
                priority=3,
            ),
            LayerSubstep(
                name="Highlights & Features",
                technique="detailing",
                mask_method="luminosity",
                mask_params={"range": (0.75, 1.0)},
                brush="script liner",
                stroke="precise small strokes",
                description="Add final highlights and facial features",
                tips=["Eyes and lips need care", "Nose tip highlight"],
                priority=4,
            ),
        ]

        return steps[:n_steps] if not is_focal else steps

    # ==================== VEGETATION STRATEGIES ====================

    def _strategy_tree(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """
        Strategy for trees - back-to-front foliage, not just dark-to-light.
        """
        if self.context.time_of_day == TimeOfDay.NIGHT:
            # Night trees - silhouette first
            return [
                LayerSubstep(
                    name="Tree Silhouette",
                    technique="blocking",
                    mask_method="luminosity",
                    mask_params={"range": (0.0, 0.5)},
                    brush="2-inch brush",
                    stroke="tap and push",
                    description="Block in the dark tree mass against the sky",
                    tips=["Trees at night are mostly silhouette", "Just hints of form"],
                    priority=1,
                ),
                LayerSubstep(
                    name="Moonlit Edges",
                    technique="highlighting",
                    mask_method="luminosity",
                    mask_params={"range": (0.4, 1.0)},
                    brush="fan brush",
                    stroke="light touches on edges",
                    description="Add subtle moonlight on outer foliage",
                    tips=["Moonlight is cool/blue", "Very sparingly"],
                    priority=2,
                ),
            ]
        else:
            # Daytime trees - back-to-front foliage
            return [
                LayerSubstep(
                    name="Back Foliage",
                    technique="blocking",
                    mask_method="spatial_back",
                    mask_params={"depth": 0.33},
                    brush="2-inch brush",
                    stroke="push and tap for foliage texture",
                    description="Start with the deepest, furthest foliage",
                    tips=["Darkest values, furthest back", "This is the interior shadow"],
                    priority=1,
                ),
                LayerSubstep(
                    name="Middle Foliage",
                    technique="layering",
                    mask_method="spatial_mid",
                    mask_params={"depth": (0.33, 0.66)},
                    brush="1-inch brush",
                    stroke="varied tapping motions",
                    description="Build up the middle layer of leaves",
                    tips=["Medium values here", "Start to show leaf clusters"],
                    priority=2,
                ),
                LayerSubstep(
                    name="Front Foliage",
                    technique="layering",
                    mask_method="spatial_front",
                    mask_params={"depth": 0.66},
                    brush="fan brush",
                    stroke="light tapping",
                    description="Add the front-most, sun-touched leaves",
                    tips=["Brightest greens", "Individual leaf suggestions"],
                    priority=3,
                ),
                LayerSubstep(
                    name="Highlights",
                    technique="highlighting",
                    mask_method="luminosity",
                    mask_params={"range": (0.75, 1.0)},
                    brush="fan brush",
                    stroke="gentle touches",
                    description="Final highlights where sun directly hits",
                    tips=["Sun catches the very outer leaves", "Yellow-green tints"],
                    priority=4,
                ),
            ]

    def _strategy_grass(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for grass - back-to-front depth."""
        return [
            LayerSubstep(
                name="Distant Grass",
                technique="blocking",
                mask_method="spatial_top",
                mask_params={"portion": 0.4},
                brush="2-inch brush",
                stroke="horizontal strokes",
                description="Block in the furthest grass area",
                tips=["More muted, bluer greens in distance", "Less detail"],
                priority=1,
            ),
            LayerSubstep(
                name="Middle Grass",
                technique="layering",
                mask_method="spatial_middle",
                mask_params={"portion": (0.3, 0.7)},
                brush="fan brush",
                stroke="upward flicking strokes",
                description="Build up the middle ground grass",
                tips=["Start suggesting individual grass blades", "More color variation"],
                priority=2,
            ),
            LayerSubstep(
                name="Foreground Grass",
                technique="detailing",
                mask_method="spatial_bottom",
                mask_params={"portion": 0.4},
                brush="fan brush",
                stroke="strong upward strokes",
                description="Add the closest, most detailed grass",
                tips=["Tallest, most distinct blades", "Highest contrast"],
                priority=3,
            ),
            LayerSubstep(
                name="Sun-touched Tips",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.8, 1.0)},
                brush="fan brush",
                stroke="quick upward flicks",
                description="Add bright tips where sun catches",
                tips=["Yellow-green highlights", "Just the tips of blades"],
                priority=4,
            ),
        ]

    def _strategy_foliage(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for generic foliage/bushes."""
        return self._strategy_tree(mask, image, is_focal)

    def _strategy_flower(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for flowers."""
        return [
            LayerSubstep(
                name="Flower Centers",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.4)},
                brush="#4 filbert",
                stroke="dabbing",
                description="Start with flower centers and shadows",
                tips=["Centers are often darker", "Creates depth in the bloom"],
                priority=1,
            ),
            LayerSubstep(
                name="Petal Shadows",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.35, 0.6)},
                brush="#6 filbert",
                stroke="petal-shaped strokes",
                description="Add the shadowed parts of petals",
                tips=["Each stroke can be a petal", "Overlap for fullness"],
                priority=2,
            ),
            LayerSubstep(
                name="Bright Petals",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.55, 1.0)},
                brush="script liner or small filbert",
                stroke="delicate petal strokes",
                description="Add the bright, sun-lit petals",
                tips=["Petals facing the sun are brightest", "Keep colors clean"],
                priority=3,
            ),
        ]

    # ==================== SKY/ATMOSPHERE STRATEGIES ====================

    def _strategy_sky(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for sky - varies dramatically by time of day."""

        if self.context.time_of_day == TimeOfDay.NIGHT:
            return [
                LayerSubstep(
                    name="Night Sky Base",
                    technique="blocking",
                    mask_method="full",
                    mask_params={},
                    brush="2-inch brush",
                    stroke="crisscross blending",
                    description="Cover the sky with deep blue-black",
                    tips=["Night sky isn't pure black - deep blue", "Darker at top"],
                    priority=1,
                ),
                LayerSubstep(
                    name="Stars/Moon",
                    technique="highlighting",
                    mask_method="luminosity",
                    mask_params={"range": (0.7, 1.0)},
                    brush="script liner",
                    stroke="tiny dots and glows",
                    description="Add stars and any moon glow",
                    tips=["Vary star sizes", "Moon has a soft glow around it"],
                    priority=2,
                ),
            ]

        elif self.context.time_of_day in [TimeOfDay.GOLDEN_HOUR, TimeOfDay.DUSK, TimeOfDay.DAWN]:
            return [
                LayerSubstep(
                    name="Horizon Glow",
                    technique="blocking",
                    mask_method="spatial_bottom",
                    mask_params={"portion": 0.4},
                    brush="2-inch brush",
                    stroke="horizontal blending",
                    description="Start with warm colors at the horizon",
                    tips=["Warmest, brightest near horizon", "Oranges, yellows, pinks"],
                    priority=1,
                ),
                LayerSubstep(
                    name="Upper Sky",
                    technique="blending",
                    mask_method="spatial_top",
                    mask_params={"portion": 0.5},
                    brush="2-inch brush",
                    stroke="blend into horizon colors",
                    description="Blend the cooler upper sky into the warm horizon",
                    tips=["Purple, blue transitioning to warm", "Seamless gradient"],
                    priority=2,
                ),
                LayerSubstep(
                    name="Sky Details",
                    technique="highlighting",
                    mask_method="luminosity",
                    mask_params={"range": (0.85, 1.0)},
                    brush="fan brush",
                    stroke="soft touches",
                    description="Add the brightest points of light",
                    tips=["Where sun is strongest", "Don't overdo it"],
                    priority=3,
                ),
            ]

        else:  # DAY
            return [
                LayerSubstep(
                    name="Upper Sky",
                    technique="blocking",
                    mask_method="spatial_top",
                    mask_params={"portion": 0.5},
                    brush="2-inch brush",
                    stroke="crisscross for smooth coverage",
                    description="Start with the deepest blue at the top",
                    tips=["Richest blue at zenith", "Thin, even coverage"],
                    priority=1,
                ),
                LayerSubstep(
                    name="Lower Sky",
                    technique="blending",
                    mask_method="spatial_bottom",
                    mask_params={"portion": 0.6},
                    brush="2-inch brush",
                    stroke="blend upward into darker blue",
                    description="Lighter blue near the horizon",
                    tips=["Atmosphere makes horizon lighter", "Seamless gradient"],
                    priority=2,
                ),
            ]

    def _strategy_cloud(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for clouds."""
        return [
            LayerSubstep(
                name="Cloud Shadows",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.5)},
                brush="1-inch brush",
                stroke="circular, fluffy motions",
                description="Block in the shadow side of clouds",
                tips=["Cloud shadows are gray-blue, not pure gray", "Soft edges"],
                priority=1,
            ),
            LayerSubstep(
                name="Cloud Body",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.4, 0.75)},
                brush="fan brush",
                stroke="gentle circular dabs",
                description="Build up the main cloud mass",
                tips=["Clouds are not pure white", "Work in layers for fluffiness"],
                priority=2,
            ),
            LayerSubstep(
                name="Cloud Highlights",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.7, 1.0)},
                brush="fan brush",
                stroke="very light touches",
                description="Add the bright tops of clouds",
                tips=["Sun hits the tops", "Pure white used sparingly"],
                priority=3,
            ),
        ]

    # ==================== TERRAIN STRATEGIES ====================

    def _strategy_water(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for water."""
        return [
            LayerSubstep(
                name="Deep Water",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.4)},
                brush="2-inch brush",
                stroke="horizontal strokes",
                description="Block in the darkest water areas",
                tips=["Water is darkest away from reflections", "Keep strokes horizontal"],
                priority=1,
            ),
            LayerSubstep(
                name="Water Body",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.35, 0.7)},
                brush="2-inch brush",
                stroke="long horizontal strokes",
                description="Build up the main water color",
                tips=["Water is always horizontal", "Reflect sky colors"],
                priority=2,
            ),
            LayerSubstep(
                name="Reflections & Highlights",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.65, 1.0)},
                brush="palette knife or fan brush",
                stroke="broken horizontal strokes",
                description="Add sparkles and reflections",
                tips=["Broken strokes for sparkle", "Reflections are less intense than source"],
                priority=3,
            ),
        ]

    def _strategy_mountain(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for mountains."""
        return [
            LayerSubstep(
                name="Mountain Shadow",
                technique="blocking",
                mask_method="spatial_left" if self.context.light_direction.endswith("right") else "spatial_right",
                mask_params={"portion": 0.5},
                brush="palette knife",
                stroke="diagonal strokes following slope",
                description="Block in the shadow side of mountains",
                tips=["Shadow side opposite the light", "Cool, muted colors"],
                priority=1,
            ),
            LayerSubstep(
                name="Mountain Light Side",
                technique="layering",
                mask_method="spatial_right" if self.context.light_direction.endswith("right") else "spatial_left",
                mask_params={"portion": 0.5},
                brush="palette knife",
                stroke="pull down strokes",
                description="Add the light side of mountains",
                tips=["Warmer where sun hits", "Follow the mountain form"],
                priority=2,
            ),
            LayerSubstep(
                name="Snow/Highlights",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.8, 1.0)},
                brush="palette knife edge",
                stroke="scrape for snow texture",
                description="Add snow caps or bright highlights",
                tips=["Snow on peaks and ridges", "Use knife edge for crisp lines"],
                priority=3,
            ),
        ]

    def _strategy_rock(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for rocks."""
        return [
            LayerSubstep(
                name="Rock Shadows",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.4)},
                brush="palette knife",
                stroke="angular, faceted strokes",
                description="Block in the dark crevices and shadows",
                tips=["Rocks have hard angles", "Deepest shadows in cracks"],
                priority=1,
            ),
            LayerSubstep(
                name="Rock Body",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.35, 0.7)},
                brush="palette knife",
                stroke="varied angular strokes",
                description="Build up the main rock surfaces",
                tips=["Vary your colors - rocks aren't uniform", "Show the planes"],
                priority=2,
            ),
            LayerSubstep(
                name="Rock Highlights",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.65, 1.0)},
                brush="palette knife edge",
                stroke="scrape highlights",
                description="Add highlights on edges and tops",
                tips=["Light catches edges", "Scraping creates texture"],
                priority=3,
            ),
        ]

    def _strategy_ground(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for generic ground."""
        return self._strategy_grass(mask, image, is_focal)

    def _strategy_sand(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for sand/beach."""
        return [
            LayerSubstep(
                name="Wet/Shadow Sand",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.45)},
                brush="2-inch brush",
                stroke="horizontal blending",
                description="Block in wet or shadowed sand areas",
                tips=["Wet sand is much darker", "Near water line"],
                priority=1,
            ),
            LayerSubstep(
                name="Dry Sand",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.4, 0.75)},
                brush="fan brush",
                stroke="light texture strokes",
                description="Build up the dry sand areas",
                tips=["Sand has many colors - not just tan", "Subtle variations"],
                priority=2,
            ),
            LayerSubstep(
                name="Bright Sand",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.7, 1.0)},
                brush="fan brush",
                stroke="dry brush for sparkle",
                description="Add highlights where sun hits directly",
                tips=["Sand sparkles in sun", "Very light touches"],
                priority=3,
            ),
        ]

    def _strategy_snow(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for snow."""
        return [
            LayerSubstep(
                name="Snow Shadows",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.5)},
                brush="2-inch brush",
                stroke="soft blending",
                description="Block in the shadow areas of snow",
                tips=["Snow shadows are blue/purple, not gray", "Very subtle"],
                priority=1,
            ),
            LayerSubstep(
                name="Snow Mid-tones",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.45, 0.8)},
                brush="fan brush",
                stroke="gentle blending",
                description="Build up the main snow surface",
                tips=["Snow reflects sky color", "Keep values high overall"],
                priority=2,
            ),
            LayerSubstep(
                name="Snow Sparkle",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.75, 1.0)},
                brush="script liner",
                stroke="tiny dots and touches",
                description="Add sparkle where sun hits snow crystals",
                tips=["Snow sparkles like glitter", "Use pure white sparingly"],
                priority=3,
            ),
        ]

    # ==================== MAN-MADE STRATEGIES ====================

    def _strategy_building(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for buildings."""
        return [
            LayerSubstep(
                name="Shadow Side",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.4)},
                brush="1-inch brush",
                stroke="straight, architectural strokes",
                description="Block in the shadow faces of buildings",
                tips=["Buildings have hard edges", "Keep lines straight"],
                priority=1,
            ),
            LayerSubstep(
                name="Main Surfaces",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.35, 0.7)},
                brush="1-inch brush",
                stroke="clean, flat strokes",
                description="Fill in the main building surfaces",
                tips=["Walls are relatively flat in value", "Watch your perspective"],
                priority=2,
            ),
            LayerSubstep(
                name="Windows & Details",
                technique="detailing",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.3)},
                brush="#4 filbert",
                stroke="precise rectangles",
                description="Add windows and architectural details",
                tips=["Windows are usually dark", "Keep them consistent"],
                priority=3,
            ),
            LayerSubstep(
                name="Highlights",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.7, 1.0)},
                brush="script liner",
                stroke="clean lines",
                description="Add highlights and reflections",
                tips=["Window reflections", "Edge highlights where sun hits"],
                priority=4,
            ),
        ]

    def _strategy_vehicle(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for vehicles."""
        return self._strategy_building(mask, image, is_focal)

    def _strategy_road(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Strategy for roads."""
        return [
            LayerSubstep(
                name="Road Base",
                technique="blocking",
                mask_method="full",
                mask_params={},
                brush="2-inch brush",
                stroke="long horizontal strokes",
                description="Block in the road surface",
                tips=["Roads are darker than you think", "Follow the perspective"],
                priority=1,
            ),
            LayerSubstep(
                name="Road Details",
                technique="detailing",
                mask_method="luminosity",
                mask_params={"range": (0.6, 1.0)},
                brush="script liner",
                stroke="dashed lines",
                description="Add road markings and highlights",
                tips=["Center lines, edge lines", "Breaks get closer in distance"],
                priority=2,
            ),
        ]

    # ==================== DEFAULT STRATEGY ====================

    def _strategy_default(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        is_focal: bool
    ) -> List[LayerSubstep]:
        """Default strategy when subject type is unknown."""
        n_steps = 4 if is_focal else 3

        steps = [
            LayerSubstep(
                name="Dark Values",
                technique="blocking",
                mask_method="luminosity",
                mask_params={"range": (0.0, 0.35)},
                brush="1-inch brush",
                stroke="follow the form",
                description="Block in the darkest areas first",
                tips=["Establish your darks", "This creates depth"],
                priority=1,
            ),
            LayerSubstep(
                name="Mid Values",
                technique="layering",
                mask_method="luminosity",
                mask_params={"range": (0.3, 0.65)},
                brush="appropriate brush for size",
                stroke="work with the form",
                description="Build up the middle values",
                tips=["This is the bulk of your work", "Take your time"],
                priority=2,
            ),
            LayerSubstep(
                name="Light Values",
                technique="highlighting",
                mask_method="luminosity",
                mask_params={"range": (0.6, 1.0)},
                brush="fan brush or script liner",
                stroke="lighter touch",
                description="Add lighter values and highlights",
                tips=["Highlights last", "Use sparingly"],
                priority=3,
            ),
        ]

        if n_steps == 4:
            steps.insert(2, LayerSubstep(
                name="Light Mid Values",
                technique="blending",
                mask_method="luminosity",
                mask_params={"range": (0.5, 0.75)},
                brush="fan brush",
                stroke="gentle blending",
                description="Transition between mid and light values",
                tips=["Smooth transitions", "Blend edges"],
                priority=2.5,
            ))

        return sorted(steps, key=lambda s: s.priority)
