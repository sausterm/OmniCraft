"""
YOLO-based Semantic Segmentation with Smart Background Analysis.

Uses YOLOv8-Seg for detecting known objects (dogs, people, etc.)
and color/depth analysis for environment regions (sky, grass, trees).

This gives us true semantic understanding:
- Dogs are labeled as "Dog" with high confidence
- Background is intelligently segmented by color/position
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path

# Check for YOLO availability
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SemanticRegion:
    """A semantically-identified region of the image."""
    id: str
    name: str                              # Human-readable name (e.g., "Dog 1", "Sky", "Grass")
    category: str                          # Category (subject, environment, background)
    mask: np.ndarray                       # Binary mask
    confidence: float                      # Detection confidence (1.0 for environment)

    # Bounding box
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h

    # Properties
    coverage: float = 0.0                  # Fraction of image
    depth_estimate: float = 0.5            # 0=far, 1=near
    dominant_color: Tuple[int, int, int] = (128, 128, 128)
    avg_luminosity: float = 0.5

    # For painting order
    paint_order: int = 0
    is_focal: bool = False


class YOLOSemanticSegmenter:
    """
    Hybrid semantic segmentation using YOLO + environment analysis.

    Pipeline:
    1. Run YOLO to detect known objects (dogs, people, cars, etc.)
    2. Create masks for detected objects
    3. Analyze remaining pixels for environment (sky, grass, trees, etc.)
    4. Return all regions with semantic labels
    """

    def __init__(self, model_size: str = "n"):
        """
        Initialize segmenter.

        Args:
            model_size: YOLO model size - "n" (nano), "s" (small), "m" (medium), "l" (large), "x" (xlarge)
        """
        self.model_size = model_size
        self.model = None

        if YOLO_AVAILABLE:
            model_name = f"yolov8{model_size}-seg.pt"
            self.model = YOLO(model_name)

    def segment(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        min_coverage: float = 0.01
    ) -> List[SemanticRegion]:
        """
        Segment image into semantic regions.

        Args:
            image: RGB image (H, W, 3)
            conf_threshold: Minimum confidence for YOLO detections
            min_coverage: Minimum coverage for environment regions

        Returns:
            List of SemanticRegion objects ordered by depth (back to front)
        """
        h, w = image.shape[:2]
        total_pixels = h * w
        regions = []

        # Track which pixels are assigned
        assigned_mask = np.zeros((h, w), dtype=bool)

        # Step 1: YOLO detection for known objects
        if self.model is not None:
            yolo_regions = self._detect_objects(image, conf_threshold)

            for region in yolo_regions:
                regions.append(region)
                assigned_mask |= region.mask

            print(f"  YOLO detected {len(yolo_regions)} objects")

        # Step 2: Analyze unassigned pixels for environment
        unassigned = ~assigned_mask
        unassigned_count = np.sum(unassigned)

        if unassigned_count > total_pixels * min_coverage:
            print(f"  Analyzing {100*unassigned_count/total_pixels:.1f}% unassigned pixels for environment...")
            env_regions = self._analyze_environment(image, unassigned, min_coverage)
            regions.extend(env_regions)
            print(f"  Found {len(env_regions)} environment regions")

        # Step 3: Order by depth (back to front)
        regions = self._order_by_depth(regions, h)

        # Step 4: Assign paint order
        for i, region in enumerate(regions):
            region.paint_order = i + 1

        return regions

    def _detect_objects(self, image: np.ndarray, conf_threshold: float) -> List[SemanticRegion]:
        """Run YOLO to detect known objects."""
        h, w = image.shape[:2]
        regions = []

        # Run inference
        results = self.model(image, verbose=False, conf=conf_threshold)

        for r in results:
            if r.masks is None:
                continue

            for i, (mask_data, box, cls, conf) in enumerate(zip(
                r.masks.data, r.boxes.xyxy, r.boxes.cls, r.boxes.conf
            )):
                # Get class name
                class_name = self.model.names[int(cls)]
                confidence = conf.item()

                # Convert mask to numpy and resize to image size
                mask = mask_data.cpu().numpy()
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
                mask = mask > 0.5

                if np.sum(mask) == 0:
                    continue

                # Get bounding box
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                bbox = (x1, y1, x2 - x1, y2 - y1)

                # Analyze region properties
                masked_pixels = image[mask]
                dominant_color = tuple(np.median(masked_pixels, axis=0).astype(int))

                # Estimate depth from y position (lower = closer)
                y_coords = np.where(mask)[0]
                y_center = np.mean(y_coords) / h
                depth_estimate = y_center  # 0=top (far), 1=bottom (near)

                # Calculate luminosity
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                avg_luminosity = np.mean(hsv[mask, 2]) / 255.0

                # Create friendly name
                # Count existing objects of this class
                existing_count = sum(1 for r in regions if class_name.lower() in r.name.lower())
                if existing_count > 0:
                    name = f"{class_name.title()} {existing_count + 1}"
                else:
                    name = class_name.title()

                region = SemanticRegion(
                    id=f"yolo_{class_name}_{i}",
                    name=name,
                    category="subject",
                    mask=mask,
                    confidence=confidence,
                    bbox=bbox,
                    coverage=np.sum(mask) / (h * w),
                    depth_estimate=depth_estimate,
                    dominant_color=dominant_color,
                    avg_luminosity=avg_luminosity,
                    is_focal=True,  # YOLO-detected objects are focal
                )
                regions.append(region)

        return regions

    def _analyze_environment(
        self,
        image: np.ndarray,
        unassigned: np.ndarray,
        min_coverage: float
    ) -> List[SemanticRegion]:
        """Analyze unassigned pixels for environment regions (sky, grass, trees, etc.)."""
        h, w = image.shape[:2]
        total_pixels = h * w
        regions = []

        # Convert to HSV for analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Get unassigned pixel coordinates
        unassigned_hsv = hsv.copy()
        unassigned_hsv[~unassigned] = 0

        # Strategy: Segment by color clustering within the unassigned area
        # Then classify each cluster by color/position

        # Use k-means on unassigned pixels
        unassigned_pixels = image[unassigned].reshape(-1, 3).astype(np.float32)

        if len(unassigned_pixels) < 100:
            return regions

        # Determine number of clusters based on color variance
        n_clusters = min(6, max(2, int(np.std(unassigned_pixels) / 20)))

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        # Subsample for speed
        if len(unassigned_pixels) > 10000:
            indices = np.random.choice(len(unassigned_pixels), 10000, replace=False)
            sample = unassigned_pixels[indices]
        else:
            sample = unassigned_pixels

        kmeans.fit(sample)

        # Predict all unassigned pixels
        labels_flat = kmeans.predict(unassigned_pixels)

        # Create label image
        label_image = np.full((h, w), -1, dtype=np.int32)
        label_image[unassigned] = labels_flat

        # Process each cluster
        for cluster_id in range(n_clusters):
            cluster_mask = (label_image == cluster_id)
            cluster_coverage = np.sum(cluster_mask) / total_pixels

            if cluster_coverage < min_coverage:
                continue

            # Analyze cluster properties
            cluster_pixels = image[cluster_mask]
            cluster_color = tuple(np.median(cluster_pixels, axis=0).astype(int))

            # Get position
            y_coords, x_coords = np.where(cluster_mask)
            y_center = np.mean(y_coords) / h
            x_center = np.mean(x_coords) / w
            y_top = np.min(y_coords) / h
            y_bottom = np.max(y_coords) / h

            # Luminosity
            cluster_hsv = hsv[cluster_mask]
            avg_luminosity = np.mean(cluster_hsv[:, 2]) / 255.0
            avg_saturation = np.mean(cluster_hsv[:, 1]) / 255.0
            avg_hue = np.mean(cluster_hsv[:, 0])

            # Classify by color and position
            name, category = self._classify_environment_region(
                cluster_color, y_center, y_top, y_bottom,
                avg_luminosity, avg_saturation, avg_hue
            )

            # Depth estimate (position-based for environment)
            depth_estimate = y_center * 0.7 + avg_saturation * 0.3

            region = SemanticRegion(
                id=f"env_{cluster_id}",
                name=name,
                category=category,
                mask=cluster_mask,
                confidence=1.0,  # Environment detection is deterministic
                coverage=cluster_coverage,
                depth_estimate=depth_estimate,
                dominant_color=cluster_color,
                avg_luminosity=avg_luminosity,
                is_focal=False,
            )
            regions.append(region)

        return regions

    def _classify_environment_region(
        self,
        color: Tuple[int, int, int],
        y_center: float,
        y_top: float,
        y_bottom: float,
        luminosity: float,
        saturation: float,
        hue: float
    ) -> Tuple[str, str]:
        """Classify an environment region by its properties."""
        r, g, b = color

        # Sky detection: top of image, blue-ish or bright
        if y_bottom < 0.4 and y_top < 0.2:
            if b > r and b > g:
                return "Sky", "background"
            elif luminosity > 0.7:
                return "Sky", "background"

        # Green detection (grass, trees, foliage)
        is_green = g > r * 1.1 and g > b * 1.1

        if is_green:
            if y_center > 0.6:
                return "Grass", "environment"
            elif y_center < 0.5:
                return "Trees", "environment"
            else:
                return "Foliage", "environment"

        # Dark regions (shadows, dark trees)
        if luminosity < 0.25:
            if y_center < 0.5:
                return "Dark Trees", "environment"
            else:
                return "Shadows", "environment"

        # Brown/earth tones
        if r > g and r > b and saturation < 0.5:
            return "Ground", "environment"

        # Bright regions
        if luminosity > 0.7:
            if y_center < 0.3:
                return "Bright Sky", "background"
            else:
                return "Highlights", "environment"

        # Default by position
        if y_center < 0.3:
            return "Background", "background"
        elif y_center < 0.6:
            return "Mid-ground", "environment"
        else:
            return "Foreground", "environment"

    def _order_by_depth(self, regions: List[SemanticRegion], image_height: int) -> List[SemanticRegion]:
        """Order regions by depth (back to front for painting)."""
        # Background first, then environment, then subjects
        category_order = {"background": 0, "environment": 1, "subject": 2}

        return sorted(regions, key=lambda r: (
            category_order.get(r.category, 1),
            r.depth_estimate
        ))


def segment_with_yolo(
    image_path: str,
    model_size: str = "n",
    conf_threshold: float = 0.5
) -> List[SemanticRegion]:
    """
    Convenience function to segment an image.

    Args:
        image_path: Path to image
        model_size: YOLO model size (n/s/m/l/x)
        conf_threshold: Minimum detection confidence

    Returns:
        List of SemanticRegion objects
    """
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    segmenter = YOLOSemanticSegmenter(model_size=model_size)
    return segmenter.segment(image, conf_threshold=conf_threshold)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python yolo_segmentation.py <image_path>")
        sys.exit(0)

    regions = segment_with_yolo(sys.argv[1])

    print(f"\nFound {len(regions)} semantic regions:")
    for r in regions:
        print(f"  {r.paint_order}. {r.name} ({r.category})")
        print(f"     Coverage: {r.coverage*100:.1f}%, Confidence: {r.confidence:.2f}")
