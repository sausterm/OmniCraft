"""
Semantic Segmentation - Understanding what's in an image.

This module provides semantic segmentation using SAM (Segment Anything Model)
when available, with fallback to classical methods.

SAM provides:
- Zero-shot segmentation (no training needed)
- High-quality masks for any object
- Ability to segment based on points, boxes, or automatic

Our system uses SAM to:
1. Automatically segment all distinct regions
2. Classify regions by type (subject, background, etc.)
3. Build a scene graph for construction planning
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np
import cv2
from pathlib import Path

# Check for SAM availability
SAM_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = TORCH_AVAILABLE
except ImportError:
    SAM_AVAILABLE = False


class SegmentType(Enum):
    """Classification of segment by role in scene."""
    BACKGROUND = "background"
    ENVIRONMENT = "environment"
    SUBJECT = "subject"
    DETAIL = "detail"
    UNKNOWN = "unknown"


@dataclass
class Segment:
    """A segmented region of the image."""
    id: str
    mask: np.ndarray                    # Binary mask (H x W)
    bbox: Tuple[int, int, int, int]     # (x, y, w, h)
    area: int                           # Pixel count
    stability_score: float              # SAM confidence
    predicted_iou: float                # SAM predicted IoU

    # Derived properties
    segment_type: SegmentType = SegmentType.UNKNOWN
    centroid: Tuple[float, float] = (0.0, 0.0)
    coverage: float = 0.0               # Fraction of image

    # Color analysis
    dominant_color: Optional[Tuple[int, int, int]] = None
    color_variance: float = 0.0
    avg_luminosity: float = 0.5

    # Semantic hints
    is_focal: bool = False              # Likely subject/focus
    depth_estimate: float = 0.5         # 0=far, 1=near
    texture_type: str = "smooth"

    # Relationships
    parent_id: Optional[str] = None
    contained_ids: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived properties."""
        if self.mask is not None:
            h, w = self.mask.shape
            self.coverage = self.area / (h * w)

            # Calculate centroid
            y_coords, x_coords = np.where(self.mask)
            if len(y_coords) > 0:
                self.centroid = (float(np.mean(x_coords)), float(np.mean(y_coords)))


@dataclass
class SegmentationResult:
    """Complete segmentation result for an image."""
    image_shape: Tuple[int, int]        # (height, width)
    segments: List[Segment]
    method: str                         # "sam" or "fallback"

    # Hierarchical organization
    background_segments: List[str] = field(default_factory=list)
    subject_segments: List[str] = field(default_factory=list)
    detail_segments: List[str] = field(default_factory=list)

    # Analysis metadata
    total_segments: int = 0
    merged_segments: int = 0            # Before merge
    processing_time_ms: float = 0.0

    def get_segment(self, segment_id: str) -> Optional[Segment]:
        """Get segment by ID."""
        for seg in self.segments:
            if seg.id == segment_id:
                return seg
        return None

    def get_segments_by_type(self, seg_type: SegmentType) -> List[Segment]:
        """Get all segments of a given type."""
        return [s for s in self.segments if s.segment_type == seg_type]

    def get_focal_segments(self) -> List[Segment]:
        """Get segments marked as focal/subject."""
        return [s for s in self.segments if s.is_focal]


class SemanticSegmenter:
    """
    Main segmentation interface.

    Uses SAM when available, falls back to classical methods otherwise.
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: str = "auto"
    ):
        """
        Initialize segmenter.

        Args:
            model_type: SAM model variant ("vit_h", "vit_l", "vit_b")
            checkpoint_path: Path to SAM checkpoint (downloads if None)
            device: "cuda", "cpu", or "auto"
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path
        self.sam_loaded = False
        self.sam = None
        self.mask_generator = None
        self.predictor = None

        # Determine device
        if device == "auto":
            if TORCH_AVAILABLE:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device

    def load_sam(self) -> bool:
        """Load SAM model if available."""
        if not SAM_AVAILABLE:
            print("SAM not available. Install with: pip install segment-anything torch torchvision")
            return False

        if self.sam_loaded:
            return True

        try:
            import torch

            # Default checkpoint paths
            if self.checkpoint_path is None:
                checkpoint_dir = Path(__file__).parent.parent / "models" / "sam"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                checkpoint_names = {
                    "vit_h": "sam_vit_h_4b8939.pth",
                    "vit_l": "sam_vit_l_0b3195.pth",
                    "vit_b": "sam_vit_b_01ec64.pth",
                }
                self.checkpoint_path = str(checkpoint_dir / checkpoint_names[self.model_type])

            if not Path(self.checkpoint_path).exists():
                print(f"SAM checkpoint not found at {self.checkpoint_path}")
                print("Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
                return False

            # Load model
            self.sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
            self.sam.to(device=self.device)

            # Create mask generator for automatic segmentation
            self.mask_generator = SamAutomaticMaskGenerator(
                model=self.sam,
                points_per_side=32,
                pred_iou_thresh=0.88,
                stability_score_thresh=0.95,
                crop_n_layers=1,
                crop_n_points_downscale_factor=2,
                min_mask_region_area=100,
            )

            # Create predictor for point/box prompts
            self.predictor = SamPredictor(self.sam)

            self.sam_loaded = True
            print(f"SAM loaded successfully ({self.model_type} on {self.device})")
            return True

        except Exception as e:
            print(f"Failed to load SAM: {e}")
            return False

    def segment(
        self,
        image: np.ndarray,
        min_area_ratio: float = 0.005,
        max_segments: int = 50,
        merge_similar: bool = True
    ) -> SegmentationResult:
        """
        Segment an image into distinct regions.

        Args:
            image: RGB image as numpy array (H, W, 3)
            min_area_ratio: Minimum segment size as fraction of image
            max_segments: Maximum number of segments to return
            merge_similar: Whether to merge similar adjacent segments

        Returns:
            SegmentationResult with all segments
        """
        import time
        start_time = time.time()

        h, w = image.shape[:2]
        min_area = int(h * w * min_area_ratio)

        # Try SAM first
        if SAM_AVAILABLE and self.load_sam():
            segments = self._segment_with_sam(image, min_area, max_segments)
            method = "sam"
        else:
            segments = self._segment_fallback(image, min_area, max_segments)
            method = "fallback"

        # Analyze and classify segments
        segments = self._analyze_segments(image, segments)

        # Merge similar if requested
        original_count = len(segments)
        if merge_similar:
            segments = self._merge_similar_segments(segments, image)

        # Classify segment types
        segments = self._classify_segments(segments, image.shape[:2])

        # Build result
        result = SegmentationResult(
            image_shape=(h, w),
            segments=segments,
            method=method,
            total_segments=len(segments),
            merged_segments=original_count,
            processing_time_ms=(time.time() - start_time) * 1000,
        )

        # Organize by type
        for seg in segments:
            if seg.segment_type == SegmentType.BACKGROUND:
                result.background_segments.append(seg.id)
            elif seg.segment_type == SegmentType.SUBJECT:
                result.subject_segments.append(seg.id)
            elif seg.segment_type == SegmentType.DETAIL:
                result.detail_segments.append(seg.id)

        return result

    def _segment_with_sam(
        self,
        image: np.ndarray,
        min_area: int,
        max_segments: int
    ) -> List[Segment]:
        """Segment using SAM automatic mask generator."""
        masks = self.mask_generator.generate(image)

        segments = []
        for i, mask_data in enumerate(masks):
            if mask_data['area'] < min_area:
                continue

            seg = Segment(
                id=f"seg_{i:04d}",
                mask=mask_data['segmentation'],
                bbox=mask_data['bbox'],
                area=mask_data['area'],
                stability_score=mask_data['stability_score'],
                predicted_iou=mask_data['predicted_iou'],
            )
            segments.append(seg)

        # Sort by area (largest first) and limit
        segments.sort(key=lambda s: s.area, reverse=True)
        return segments[:max_segments]

    def _segment_fallback(
        self,
        image: np.ndarray,
        min_area: int,
        max_segments: int
    ) -> List[Segment]:
        """Fallback segmentation using classical methods."""
        h, w = image.shape[:2]

        # Convert to LAB for better color clustering
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Use superpixels + watershed
        # First, create superpixels using SLIC-like approach with k-means
        pixels = lab.reshape(-1, 3).astype(np.float32)

        # Determine optimal number of clusters
        n_clusters = min(max_segments, max(8, int(np.sqrt(h * w) / 20)))

        # K-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            pixels, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )

        label_map = labels.reshape(h, w)

        # Create segments from clusters
        segments = []
        for i in range(n_clusters):
            mask = (label_map == i).astype(np.uint8)

            # Clean up mask with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            area = np.sum(mask)
            if area < min_area:
                continue

            # Get bounding box
            y_coords, x_coords = np.where(mask)
            if len(y_coords) == 0:
                continue

            bbox = (
                int(np.min(x_coords)),
                int(np.min(y_coords)),
                int(np.max(x_coords) - np.min(x_coords)),
                int(np.max(y_coords) - np.min(y_coords))
            )

            seg = Segment(
                id=f"seg_{i:04d}",
                mask=mask.astype(bool),
                bbox=bbox,
                area=int(area),
                stability_score=0.8,  # Estimated
                predicted_iou=0.7,    # Estimated
            )
            segments.append(seg)

        segments.sort(key=lambda s: s.area, reverse=True)
        return segments[:max_segments]

    def _analyze_segments(
        self,
        image: np.ndarray,
        segments: List[Segment]
    ) -> List[Segment]:
        """Analyze each segment for color, texture, and other properties."""
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        h, w = image.shape[:2]

        for seg in segments:
            # Get masked region
            masked_pixels = image[seg.mask]

            if len(masked_pixels) == 0:
                continue

            # Dominant color (mode of RGB values)
            seg.dominant_color = tuple(np.median(masked_pixels, axis=0).astype(int))

            # Color variance
            seg.color_variance = float(np.std(masked_pixels))

            # Luminosity
            seg.avg_luminosity = float(np.mean(lab[seg.mask, 0]) / 255.0)

            # Texture analysis (using Laplacian variance)
            mask_gray = gray.copy()
            mask_gray[~seg.mask] = 0
            laplacian = cv2.Laplacian(mask_gray, cv2.CV_64F)
            laplacian_var = np.var(laplacian[seg.mask]) if np.any(seg.mask) else 0

            if laplacian_var < 100:
                seg.texture_type = "smooth"
            elif laplacian_var < 500:
                seg.texture_type = "medium"
            else:
                seg.texture_type = "rough"

            # Depth estimate based on position and size
            # Higher in image + larger = more likely background
            norm_y = seg.centroid[1] / h
            norm_size = seg.coverage

            # Simple heuristic: top of image = far, large areas = likely background
            if norm_y < 0.3 and norm_size > 0.1:
                seg.depth_estimate = 0.2  # Far background (sky)
            elif norm_y > 0.7:
                seg.depth_estimate = 0.8  # Foreground
            else:
                seg.depth_estimate = 0.5 + (norm_y - 0.5) * 0.4

        return segments

    def _classify_segments(
        self,
        segments: List[Segment],
        image_shape: Tuple[int, int]
    ) -> List[Segment]:
        """Classify segments by type (background, subject, detail)."""
        h, w = image_shape

        # Calculate image center and focal region
        center = (w / 2, h / 2)

        for seg in segments:
            # Distance from center
            dx = seg.centroid[0] - center[0]
            dy = seg.centroid[1] - center[1]
            dist_from_center = np.sqrt(dx*dx + dy*dy) / np.sqrt(center[0]**2 + center[1]**2)

            # Classify based on multiple factors
            if seg.coverage > 0.3:
                # Large area - likely background
                seg.segment_type = SegmentType.BACKGROUND
            elif seg.coverage < 0.01:
                # Small area - likely detail
                seg.segment_type = SegmentType.DETAIL
            elif dist_from_center < 0.4 and seg.coverage > 0.02:
                # Near center, reasonable size - likely subject
                seg.segment_type = SegmentType.SUBJECT
                seg.is_focal = True
            elif seg.depth_estimate < 0.3:
                # Far depth estimate - background/environment
                seg.segment_type = SegmentType.BACKGROUND
            elif seg.depth_estimate > 0.6 and seg.coverage > 0.05:
                # Near depth, decent size - could be subject
                seg.segment_type = SegmentType.SUBJECT
            else:
                seg.segment_type = SegmentType.ENVIRONMENT

        return segments

    def _merge_similar_segments(
        self,
        segments: List[Segment],
        image: np.ndarray
    ) -> List[Segment]:
        """Merge segments with similar colors that are adjacent."""
        if len(segments) < 2:
            return segments

        # Build adjacency and similarity
        merged = []
        used = set()

        for i, seg_a in enumerate(segments):
            if i in used:
                continue

            # Find similar neighbors to merge
            to_merge = [seg_a]
            used.add(i)

            for j, seg_b in enumerate(segments):
                if j in used or j == i:
                    continue

                # Check color similarity
                if seg_a.dominant_color and seg_b.dominant_color:
                    color_dist = np.sqrt(sum(
                        (a - b) ** 2
                        for a, b in zip(seg_a.dominant_color, seg_b.dominant_color)
                    ))

                    # Check adjacency (dilate mask and check overlap)
                    if color_dist < 30:  # Similar color
                        kernel = np.ones((5, 5), np.uint8)
                        dilated_a = cv2.dilate(seg_a.mask.astype(np.uint8), kernel)
                        if np.any(dilated_a & seg_b.mask.astype(np.uint8)):
                            # Adjacent and similar - merge
                            to_merge.append(seg_b)
                            used.add(j)

            # Create merged segment
            if len(to_merge) == 1:
                merged.append(seg_a)
            else:
                # Combine masks
                combined_mask = np.zeros_like(seg_a.mask)
                for seg in to_merge:
                    combined_mask = combined_mask | seg.mask

                # Create new segment
                y_coords, x_coords = np.where(combined_mask)
                bbox = (
                    int(np.min(x_coords)),
                    int(np.min(y_coords)),
                    int(np.max(x_coords) - np.min(x_coords)),
                    int(np.max(y_coords) - np.min(y_coords))
                )

                new_seg = Segment(
                    id=f"merged_{len(merged):04d}",
                    mask=combined_mask,
                    bbox=bbox,
                    area=int(np.sum(combined_mask)),
                    stability_score=max(s.stability_score for s in to_merge),
                    predicted_iou=max(s.predicted_iou for s in to_merge),
                    dominant_color=seg_a.dominant_color,
                    avg_luminosity=np.mean([s.avg_luminosity for s in to_merge]),
                )
                merged.append(new_seg)

        return merged

    def segment_with_prompt(
        self,
        image: np.ndarray,
        points: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        box: Optional[Tuple[int, int, int, int]] = None
    ) -> Optional[Segment]:
        """
        Segment with user prompt (points or box).

        Args:
            image: RGB image
            points: List of (x, y) points
            point_labels: 1 for foreground, 0 for background
            box: Bounding box (x1, y1, x2, y2)

        Returns:
            Segment or None if SAM not available
        """
        if not SAM_AVAILABLE or not self.load_sam():
            return None

        self.predictor.set_image(image)

        input_point = np.array(points) if points else None
        input_label = np.array(point_labels) if point_labels else None
        input_box = np.array(box) if box else None

        masks, scores, _ = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=True,
        )

        # Take best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx]

        y_coords, x_coords = np.where(mask)
        bbox = (
            int(np.min(x_coords)),
            int(np.min(y_coords)),
            int(np.max(x_coords) - np.min(x_coords)),
            int(np.max(y_coords) - np.min(y_coords))
        )

        return Segment(
            id="prompted_seg",
            mask=mask,
            bbox=bbox,
            area=int(np.sum(mask)),
            stability_score=float(scores[best_idx]),
            predicted_iou=float(scores[best_idx]),
        )
