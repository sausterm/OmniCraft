"""
Subject Detector - Identifies and classifies subjects in images.

This module detects what kind of subject is in an image (portrait, pet, landscape)
and identifies key regions within subjects that need special handling
(eyes, fur, skin, etc.).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import cv2

from ..core.constraints import SubjectDomain
from .semantic import SegmentationResult, Segment, SegmentType


class SubjectCategory(Enum):
    """Fine-grained subject categories."""
    # Portrait
    HUMAN_FACE = "human_face"
    HUMAN_FIGURE = "human_figure"

    # Animals
    DOG = "dog"
    CAT = "cat"
    BIRD = "bird"
    HORSE = "horse"
    OTHER_ANIMAL = "other_animal"

    # Landscape elements
    SKY = "sky"
    WATER = "water"
    MOUNTAIN = "mountain"
    TREE = "tree"
    FOLIAGE = "foliage"
    GRASS = "grass"

    # Still life
    FLOWER = "flower"
    FOOD = "food"
    OBJECT = "object"

    # General
    UNKNOWN = "unknown"


@dataclass
class SubjectRegion:
    """A region within a subject that needs special handling."""
    name: str
    category: str                       # "eyes", "fur", "skin", etc.
    mask: np.ndarray
    importance: float                   # For painting order/focus
    special_techniques: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DetectedSubject:
    """A detected subject in the image."""
    id: str
    category: SubjectCategory
    domain: SubjectDomain
    confidence: float

    # Spatial info
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]     # (x, y, w, h)
    centroid: Tuple[float, float]

    # Sub-regions
    regions: List[SubjectRegion] = field(default_factory=list)

    # Painting guidance
    recommended_techniques: List[str] = field(default_factory=list)
    key_challenges: List[str] = field(default_factory=list)
    focus_areas: List[str] = field(default_factory=list)


@dataclass
class SubjectAnalysis:
    """Complete subject analysis for an image."""
    primary_domain: SubjectDomain
    subjects: List[DetectedSubject]
    dominant_subject_id: Optional[str] = None
    scene_complexity: float = 0.5       # 0=simple, 1=complex
    recommended_approach: str = ""


class SubjectDetector:
    """
    Detects and analyzes subjects in images.

    Uses a combination of:
    - Color/luminosity analysis
    - Position heuristics
    - Shape analysis
    - Optional ML models (when available)
    """

    def __init__(self):
        # Color ranges for common subjects (in HSV)
        self.color_hints = {
            "sky": {"h_range": (90, 130), "s_range": (20, 255), "v_range": (100, 255)},
            "grass": {"h_range": (35, 85), "s_range": (30, 255), "v_range": (20, 200)},
            "skin": {"h_range": (0, 25), "s_range": (20, 200), "v_range": (100, 255)},
            "water": {"h_range": (90, 130), "s_range": (30, 255), "v_range": (50, 200)},
        }

        # Position hints
        self.position_hints = {
            "sky": {"y_range": (0.0, 0.4), "min_coverage": 0.1},
            "ground": {"y_range": (0.6, 1.0), "min_coverage": 0.1},
            "subject": {"y_range": (0.2, 0.8), "x_range": (0.15, 0.85)},
        }

    def analyze(
        self,
        image: np.ndarray,
        segmentation: Optional[SegmentationResult] = None
    ) -> SubjectAnalysis:
        """
        Analyze image to detect and classify subjects.

        Args:
            image: RGB image
            segmentation: Optional pre-computed segmentation

        Returns:
            SubjectAnalysis with detected subjects
        """
        h, w = image.shape[:2]

        # Detect primary domain
        domain, domain_confidence = self._detect_domain(image)

        subjects = []

        # If we have segmentation, analyze segments
        if segmentation:
            subjects = self._analyze_segments(image, segmentation, domain)
        else:
            # Basic analysis without segmentation
            subjects = self._basic_subject_detection(image, domain)

        # Find dominant subject
        dominant_id = None
        if subjects:
            # Dominant = largest subject-type detection
            subject_subjects = [s for s in subjects if s.category not in
                              [SubjectCategory.SKY, SubjectCategory.GRASS, SubjectCategory.WATER]]
            if subject_subjects:
                dominant = max(subject_subjects, key=lambda s: np.sum(s.mask))
                dominant_id = dominant.id

        # Calculate scene complexity
        complexity = self._calculate_complexity(image, subjects)

        # Generate recommended approach
        approach = self._recommend_approach(domain, subjects, complexity)

        return SubjectAnalysis(
            primary_domain=domain,
            subjects=subjects,
            dominant_subject_id=dominant_id,
            scene_complexity=complexity,
            recommended_approach=approach,
        )

    def _detect_domain(
        self,
        image: np.ndarray
    ) -> Tuple[SubjectDomain, float]:
        """Detect the primary domain of the image."""
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        scores = {
            SubjectDomain.LANDSCAPE: 0.0,
            SubjectDomain.PORTRAIT: 0.0,
            SubjectDomain.ANIMAL: 0.0,
            SubjectDomain.STILL_LIFE: 0.0,
            SubjectDomain.BOTANICAL: 0.0,
        }

        # Check for sky in upper portion
        upper_region = hsv[:int(h*0.3), :, :]
        sky_mask = self._color_in_range(upper_region, self.color_hints["sky"])
        if np.mean(sky_mask) > 0.3:
            scores[SubjectDomain.LANDSCAPE] += 0.3

        # Check for skin tones (portrait indicator)
        skin_mask = self._color_in_range(hsv, self.color_hints["skin"])
        skin_ratio = np.mean(skin_mask)
        if skin_ratio > 0.05:
            scores[SubjectDomain.PORTRAIT] += skin_ratio * 2

        # Check for green (landscape/botanical indicator)
        grass_mask = self._color_in_range(hsv, self.color_hints["grass"])
        green_ratio = np.mean(grass_mask)
        if green_ratio > 0.1:
            scores[SubjectDomain.LANDSCAPE] += green_ratio
            scores[SubjectDomain.BOTANICAL] += green_ratio * 0.5

        # Check image aspect ratio (portraits often vertical)
        if h > w * 1.2:
            scores[SubjectDomain.PORTRAIT] += 0.1

        # Get best match
        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        # Normalize confidence
        total = sum(scores.values())
        confidence = best_score / total if total > 0 else 0.5

        # Default to mixed if uncertain
        if confidence < 0.3:
            return SubjectDomain.MIXED, 0.5

        return best_domain, confidence

    def _color_in_range(
        self,
        hsv_image: np.ndarray,
        color_hint: Dict
    ) -> np.ndarray:
        """Check how much of image matches a color range."""
        h_min, h_max = color_hint.get("h_range", (0, 180))
        s_min, s_max = color_hint.get("s_range", (0, 255))
        v_min, v_max = color_hint.get("v_range", (0, 255))

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])

        return cv2.inRange(hsv_image, lower, upper) > 0

    def _analyze_segments(
        self,
        image: np.ndarray,
        segmentation: SegmentationResult,
        domain: SubjectDomain
    ) -> List[DetectedSubject]:
        """Analyze segments to detect subjects."""
        subjects = []
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        for i, segment in enumerate(segmentation.segments):
            # Skip very small segments
            if segment.coverage < 0.01:
                continue

            # Classify segment
            category = self._classify_segment(segment, image, hsv, domain)

            if category == SubjectCategory.UNKNOWN:
                continue

            # Map to domain
            segment_domain = self._category_to_domain(category)

            # Detect sub-regions for subjects
            regions = []
            if segment.is_focal or segment.segment_type == SegmentType.SUBJECT:
                regions = self._detect_subject_regions(segment, image, category)

            # Generate techniques and challenges
            techniques = self._get_recommended_techniques(category)
            challenges = self._get_key_challenges(category)
            focus = self._get_focus_areas(category)

            subject = DetectedSubject(
                id=f"subject_{i:04d}",
                category=category,
                domain=segment_domain,
                confidence=segment.stability_score,
                mask=segment.mask,
                bbox=segment.bbox,
                centroid=segment.centroid,
                regions=regions,
                recommended_techniques=techniques,
                key_challenges=challenges,
                focus_areas=focus,
            )
            subjects.append(subject)

        return subjects

    def _classify_segment(
        self,
        segment: Segment,
        image: np.ndarray,
        hsv: np.ndarray,
        domain: SubjectDomain
    ) -> SubjectCategory:
        """Classify a segment into a subject category."""
        h, w = image.shape[:2]

        # Get segment properties
        norm_y = segment.centroid[1] / h
        norm_x = segment.centroid[0] / w
        coverage = segment.coverage
        luminosity = segment.avg_luminosity

        # Get color stats for segment
        masked_hsv = hsv[segment.mask]
        masked_rgb = image[segment.mask]
        if len(masked_hsv) == 0:
            return SubjectCategory.UNKNOWN

        avg_h = np.mean(masked_hsv[:, 0])
        avg_s = np.mean(masked_hsv[:, 1])
        avg_v = np.mean(masked_hsv[:, 2])

        # Color variance (high variance = textured/complex, like animal fur)
        color_var = np.std(masked_rgb, axis=0).mean() if len(masked_rgb) > 0 else 0

        # Check for sky (top of image, blue, bright)
        if norm_y < 0.35 and luminosity > 0.5 and coverage > 0.1:
            if 90 < avg_h < 130:
                return SubjectCategory.SKY

        # Check for potential ANIMAL/SUBJECT (key addition)
        # Animals tend to have:
        # - Brown/tan/black colors (not blue, not bright green)
        # - Higher color variance (fur texture)
        # - Position in middle/lower frame (not just at top)
        # - Not extremely high saturation (unlike grass)
        is_animal_color = (
            (avg_h < 35 or avg_h > 140) and  # Not green, not blue
            avg_s < 200 and                   # Not hyper-saturated
            avg_v < 200 and                   # Not pure white/bright
            color_var > 15                    # Has texture variation
        )

        is_subject_position = (
            0.15 < norm_x < 0.85 and          # Not at extreme edges
            0.2 < norm_y < 0.9                # Not at very top, allows bottom
        )

        # Animal detection: brown/black/tan regions with texture
        if is_animal_color and is_subject_position and coverage > 0.02:
            # Additional check: is it dark brown/black (typical dog/animal colors)?
            is_brown_black = (avg_h < 25 or avg_h > 160) and avg_v < 150
            is_tan_light = avg_h < 30 and avg_s < 150 and avg_v > 100

            if is_brown_black or is_tan_light:
                # High confidence animal region
                if domain == SubjectDomain.ANIMAL or segment.is_focal:
                    return SubjectCategory.DOG
                # Even without domain hint, if it looks like an animal
                elif color_var > 25 and coverage > 0.03:
                    return SubjectCategory.DOG  # Likely animal

        # Check for grass/ground (green, lower in frame)
        if norm_y > 0.5 and coverage > 0.05:  # Lowered position threshold
            if 35 < avg_h < 85 and avg_s > 30:
                # But NOT if it has high texture variance in center of frame (could be animal)
                if not (is_subject_position and color_var > 30):
                    return SubjectCategory.GRASS

        # Check for trees/foliage (green, but upper/middle)
        if norm_y < 0.6 and 35 < avg_h < 85:
            if luminosity < 0.4:  # Darker green = likely trees
                return SubjectCategory.TREE
            else:
                return SubjectCategory.FOLIAGE

        # For focal segments, use domain hints
        if segment.is_focal:
            if domain == SubjectDomain.PORTRAIT:
                return SubjectCategory.HUMAN_FACE
            elif domain == SubjectDomain.ANIMAL:
                return SubjectCategory.DOG
            elif domain == SubjectDomain.BOTANICAL:
                return SubjectCategory.FLOWER
            else:
                # Check if it looks like an animal by color
                if is_animal_color:
                    return SubjectCategory.DOG

        # Check for water (dark, blue-ish)
        if luminosity < 0.5 and 90 < avg_h < 130:
            return SubjectCategory.WATER

        # Last chance: if in subject position with animal-like colors
        if is_subject_position and is_animal_color and coverage > 0.01:
            return SubjectCategory.OTHER_ANIMAL

        return SubjectCategory.UNKNOWN

    def _category_to_domain(self, category: SubjectCategory) -> SubjectDomain:
        """Map category to domain."""
        mapping = {
            SubjectCategory.HUMAN_FACE: SubjectDomain.PORTRAIT,
            SubjectCategory.HUMAN_FIGURE: SubjectDomain.PORTRAIT,
            SubjectCategory.DOG: SubjectDomain.ANIMAL,
            SubjectCategory.CAT: SubjectDomain.ANIMAL,
            SubjectCategory.BIRD: SubjectDomain.ANIMAL,
            SubjectCategory.HORSE: SubjectDomain.ANIMAL,
            SubjectCategory.OTHER_ANIMAL: SubjectDomain.ANIMAL,
            SubjectCategory.SKY: SubjectDomain.LANDSCAPE,
            SubjectCategory.WATER: SubjectDomain.LANDSCAPE,
            SubjectCategory.MOUNTAIN: SubjectDomain.LANDSCAPE,
            SubjectCategory.TREE: SubjectDomain.LANDSCAPE,
            SubjectCategory.FOLIAGE: SubjectDomain.LANDSCAPE,
            SubjectCategory.GRASS: SubjectDomain.LANDSCAPE,
            SubjectCategory.FLOWER: SubjectDomain.BOTANICAL,
            SubjectCategory.FOOD: SubjectDomain.STILL_LIFE,
            SubjectCategory.OBJECT: SubjectDomain.STILL_LIFE,
        }
        return mapping.get(category, SubjectDomain.MIXED)

    def _detect_subject_regions(
        self,
        segment: Segment,
        image: np.ndarray,
        category: SubjectCategory
    ) -> List[SubjectRegion]:
        """Detect sub-regions within a subject."""
        regions = []

        # Domain-specific region detection
        if category in [SubjectCategory.DOG, SubjectCategory.CAT]:
            regions = self._detect_animal_regions(segment, image)
        elif category == SubjectCategory.HUMAN_FACE:
            regions = self._detect_face_regions(segment, image)

        return regions

    def _detect_animal_regions(
        self,
        segment: Segment,
        image: np.ndarray
    ) -> List[SubjectRegion]:
        """Detect regions in an animal subject (eyes, fur, etc.)."""
        regions = []
        h, w = image.shape[:2]

        # Get bounding box of segment
        y_coords, x_coords = np.where(segment.mask)
        if len(y_coords) == 0:
            return regions

        y_min, y_max = np.min(y_coords), np.max(y_coords)
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        seg_h = y_max - y_min
        seg_w = x_max - x_min

        # Estimate eye region (upper third, usually)
        eye_region_y = (y_min, y_min + seg_h // 3)
        eye_mask = segment.mask.copy()
        eye_mask[:eye_region_y[0], :] = False
        eye_mask[eye_region_y[1]:, :] = False

        if np.any(eye_mask):
            regions.append(SubjectRegion(
                name="eye_region",
                category="eyes",
                mask=eye_mask,
                importance=0.9,
                special_techniques=["fine_detail", "highlight"],
                notes="Eyes are the focal point - pay extra attention to catchlights and expression",
            ))

        # Body/fur region (main mass)
        body_mask = segment.mask.copy()
        body_mask[eye_region_y[0]:eye_region_y[1], :] = False  # Exclude eye region

        if np.any(body_mask):
            regions.append(SubjectRegion(
                name="fur_body",
                category="fur",
                mask=body_mask,
                importance=0.7,
                special_techniques=["dry_brush", "fur_strokes"],
                notes="Build up fur texture with directional strokes following fur growth",
            ))

        return regions

    def _detect_face_regions(
        self,
        segment: Segment,
        image: np.ndarray
    ) -> List[SubjectRegion]:
        """Detect regions in a face (eyes, nose, mouth, skin)."""
        regions = []

        # Basic face region detection based on position
        y_coords, x_coords = np.where(segment.mask)
        if len(y_coords) == 0:
            return regions

        y_min, y_max = np.min(y_coords), np.max(y_coords)
        seg_h = y_max - y_min

        # Eye region (upper third)
        eye_mask = segment.mask.copy()
        eye_mask[:y_min + seg_h // 4, :] = False
        eye_mask[y_min + seg_h // 2:, :] = False

        if np.any(eye_mask):
            regions.append(SubjectRegion(
                name="eye_region",
                category="eyes",
                mask=eye_mask,
                importance=0.95,
                special_techniques=["fine_detail", "glazing"],
                notes="Eyes are critical - build up layers, add catchlights last",
            ))

        # Skin region (whole face minus features)
        regions.append(SubjectRegion(
            name="skin",
            category="skin",
            mask=segment.mask,
            importance=0.7,
            special_techniques=["wet_blend", "glazing", "scumbling"],
            notes="Mix skin tones carefully, work warm to cool, preserve translucency",
        ))

        return regions

    def _basic_subject_detection(
        self,
        image: np.ndarray,
        domain: SubjectDomain
    ) -> List[DetectedSubject]:
        """Basic subject detection without segmentation."""
        subjects = []
        h, w = image.shape[:2]

        # Simple center-based subject detection
        center_mask = np.zeros((h, w), dtype=bool)
        margin_x = w // 4
        margin_y = h // 4
        center_mask[margin_y:h-margin_y, margin_x:w-margin_x] = True

        subject = DetectedSubject(
            id="subject_0000",
            category=SubjectCategory.UNKNOWN,
            domain=domain,
            confidence=0.5,
            mask=center_mask,
            bbox=(margin_x, margin_y, w - 2*margin_x, h - 2*margin_y),
            centroid=(w/2, h/2),
            recommended_techniques=self._get_recommended_techniques_for_domain(domain),
            key_challenges=self._get_key_challenges_for_domain(domain),
        )
        subjects.append(subject)

        return subjects

    def _get_recommended_techniques(self, category: SubjectCategory) -> List[str]:
        """Get recommended techniques for a subject category."""
        techniques = {
            SubjectCategory.HUMAN_FACE: ["wet_blend", "glazing", "fine_detail"],
            SubjectCategory.DOG: ["dry_brush", "fur_strokes", "wet_blend"],
            SubjectCategory.CAT: ["dry_brush", "fur_strokes", "fine_detail"],
            SubjectCategory.SKY: ["wet_blend", "gradient", "feathering"],
            SubjectCategory.WATER: ["wet_blend", "glazing", "horizontal_strokes"],
            SubjectCategory.GRASS: ["stippling", "dry_brush", "vertical_strokes"],
            SubjectCategory.FLOWER: ["fine_detail", "layering", "glazing"],
        }
        return techniques.get(category, ["layering", "blocking"])

    def _get_key_challenges(self, category: SubjectCategory) -> List[str]:
        """Get key challenges for a subject category."""
        challenges = {
            SubjectCategory.HUMAN_FACE: [
                "Accurate skin tone mixing",
                "Subtle value transitions",
                "Eye rendering",
            ],
            SubjectCategory.DOG: [
                "Fur texture and direction",
                "Capturing expression in eyes",
                "Warm/cool color temperature in fur",
            ],
            SubjectCategory.CAT: [
                "Fine fur texture",
                "Reflective eyes",
                "Subtle markings",
            ],
            SubjectCategory.SKY: [
                "Smooth gradients",
                "Avoiding streaks",
                "Cloud edges",
            ],
        }
        return challenges.get(category, ["Value accuracy", "Edge control"])

    def _get_focus_areas(self, category: SubjectCategory) -> List[str]:
        """Get focus areas for a subject category."""
        focus = {
            SubjectCategory.HUMAN_FACE: ["eyes", "skin tones", "expression"],
            SubjectCategory.DOG: ["eyes", "fur texture", "nose"],
            SubjectCategory.CAT: ["eyes", "whiskers", "fur pattern"],
            SubjectCategory.SKY: ["color gradient", "cloud edges"],
        }
        return focus.get(category, ["overall composition"])

    def _get_recommended_techniques_for_domain(self, domain: SubjectDomain) -> List[str]:
        """Get techniques for a domain."""
        domain_techniques = {
            SubjectDomain.PORTRAIT: ["wet_blend", "glazing", "fine_detail"],
            SubjectDomain.ANIMAL: ["dry_brush", "fur_strokes", "wet_blend"],
            SubjectDomain.LANDSCAPE: ["wet_blend", "dry_brush", "gradient"],
            SubjectDomain.BOTANICAL: ["fine_detail", "glazing", "layering"],
            SubjectDomain.STILL_LIFE: ["blocking", "glazing", "highlighting"],
        }
        return domain_techniques.get(domain, ["layering", "blocking"])

    def _get_key_challenges_for_domain(self, domain: SubjectDomain) -> List[str]:
        """Get challenges for a domain."""
        challenges = {
            SubjectDomain.PORTRAIT: ["Skin tone accuracy", "Likeness"],
            SubjectDomain.ANIMAL: ["Texture rendering", "Capturing character"],
            SubjectDomain.LANDSCAPE: ["Atmospheric perspective", "Color harmony"],
            SubjectDomain.BOTANICAL: ["Delicate details", "Color accuracy"],
        }
        return challenges.get(domain, ["Value accuracy"])

    def _calculate_complexity(
        self,
        image: np.ndarray,
        subjects: List[DetectedSubject]
    ) -> float:
        """Calculate overall scene complexity."""
        # Factors: number of subjects, diversity of categories, detail level

        complexity = 0.3  # Base

        # More subjects = more complex
        complexity += min(0.3, len(subjects) * 0.05)

        # Diverse categories = more complex
        categories = set(s.category for s in subjects)
        complexity += min(0.2, len(categories) * 0.05)

        # Image variance = more complex
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = np.var(gray) / 10000
        complexity += min(0.2, variance)

        return min(1.0, complexity)

    def _recommend_approach(
        self,
        domain: SubjectDomain,
        subjects: List[DetectedSubject],
        complexity: float
    ) -> str:
        """Generate recommended painting approach."""
        approaches = {
            SubjectDomain.PORTRAIT: (
                "Start with a warm-toned ground. Block in shadow shapes first, "
                "then work from general to specific. Save highlights and details for last."
            ),
            SubjectDomain.ANIMAL: (
                "Establish the overall shape and value structure first. "
                "Build fur texture in the direction of growth using dry brush. "
                "Focus on the eyes early - they carry the expression."
            ),
            SubjectDomain.LANDSCAPE: (
                "Work from back to front, sky first if present. "
                "Establish atmospheric perspective through value and color temperature changes. "
                "Keep background soft, add detail progressively toward foreground."
            ),
            SubjectDomain.BOTANICAL: (
                "Study the structure carefully before starting. "
                "Build up layers from darkest areas to lightest. "
                "Pay attention to how light passes through petals and leaves."
            ),
            SubjectDomain.STILL_LIFE: (
                "Establish the value structure with an underpainting. "
                "Observe how light falls across different surfaces. "
                "Build up form through careful value transitions."
            ),
        }

        base_approach = approaches.get(domain, "Work from general to specific, background to foreground.")

        # Adjust for complexity
        if complexity > 0.7:
            base_approach += " Take your time - this is a complex composition."
        elif complexity < 0.3:
            base_approach += " This is a straightforward composition - focus on execution."

        return base_approach
