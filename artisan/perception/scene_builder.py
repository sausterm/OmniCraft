"""
Scene Graph Builder - Constructs scene graphs from segmentation results.

This module takes raw segmentation results and builds a structured scene graph
that represents the semantic understanding of the image.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import cv2

from ..core.scene_graph import (
    SceneGraph, Entity, EntityType, MaterialProperties,
    TextureType, RelationshipType, Color, BoundingBox
)
from .semantic import SegmentationResult, Segment, SegmentType


class SceneGraphBuilder:
    """
    Builds a SceneGraph from segmentation results.

    This class:
    1. Converts segments to entities
    2. Establishes hierarchical relationships (contains, part_of)
    3. Establishes spatial relationships (above, below, adjacent)
    4. Assigns material properties based on analysis
    """

    def __init__(self):
        self.texture_map = {
            "smooth": TextureType.SMOOTH,
            "medium": TextureType.ORGANIC,
            "rough": TextureType.ROUGH,
        }

    def build(
        self,
        segmentation: SegmentationResult,
        image: np.ndarray
    ) -> SceneGraph:
        """
        Build a scene graph from segmentation results.

        Args:
            segmentation: SegmentationResult from SemanticSegmenter
            image: Original RGB image

        Returns:
            Complete SceneGraph
        """
        scene = SceneGraph(image_shape=segmentation.image_shape)

        # Convert segments to entities
        entity_map = {}  # segment_id -> entity_id
        for segment in segmentation.segments:
            entity = self._segment_to_entity(segment, image, scene)
            entity_map[segment.id] = entity.id

        # Build relationships
        self._build_containment_relationships(segmentation.segments, scene, entity_map)
        self._build_spatial_relationships(segmentation.segments, scene, entity_map)

        return scene

    def _segment_to_entity(
        self,
        segment: Segment,
        image: np.ndarray,
        scene: SceneGraph
    ) -> Entity:
        """Convert a segment to an entity."""
        # Map segment type to entity type
        type_map = {
            SegmentType.BACKGROUND: EntityType.BACKGROUND,
            SegmentType.ENVIRONMENT: EntityType.ENVIRONMENT,
            SegmentType.SUBJECT: EntityType.SUBJECT,
            SegmentType.DETAIL: EntityType.DETAIL,
            SegmentType.UNKNOWN: EntityType.ENVIRONMENT,
        }
        entity_type = type_map[segment.segment_type]

        # Generate name
        name = self._generate_entity_name(segment, entity_type)

        # Build material properties
        properties = self._analyze_material_properties(segment, image)

        # Calculate importance based on multiple factors
        importance = self._calculate_importance(segment)

        # Create entity
        entity = scene.create_entity(
            name=name,
            entity_type=entity_type,
            mask=segment.mask,
            properties=properties,
            importance=importance,
            segment_id=segment.id,
            stability_score=segment.stability_score,
        )

        return entity

    def _generate_entity_name(self, segment: Segment, entity_type: EntityType) -> str:
        """Generate a descriptive name for an entity."""
        # Base name from type
        type_names = {
            EntityType.BACKGROUND: "background",
            EntityType.ENVIRONMENT: "environment",
            EntityType.SUBJECT: "subject",
            EntityType.DETAIL: "detail",
            EntityType.ACCENT: "accent",
        }
        base_name = type_names.get(entity_type, "region")

        # Add position descriptor
        h = segment.mask.shape[0]
        norm_y = segment.centroid[1] / h

        if norm_y < 0.25:
            position = "upper"
        elif norm_y < 0.5:
            position = "mid-upper"
        elif norm_y < 0.75:
            position = "mid-lower"
        else:
            position = "lower"

        # Add luminosity descriptor
        if segment.avg_luminosity > 0.7:
            tone = "light"
        elif segment.avg_luminosity < 0.3:
            tone = "dark"
        else:
            tone = "mid"

        return f"{position}_{tone}_{base_name}"

    def _analyze_material_properties(
        self,
        segment: Segment,
        image: np.ndarray
    ) -> MaterialProperties:
        """Analyze segment to determine material properties."""
        # Get colors in segment
        masked_pixels = image[segment.mask]

        if len(masked_pixels) == 0:
            return MaterialProperties(base_color=Color(128, 128, 128))

        # Base color (median for robustness)
        median_color = np.median(masked_pixels, axis=0).astype(int)
        base_color = Color(int(median_color[0]), int(median_color[1]), int(median_color[2]))

        # Color variance
        color_variance = float(np.std(masked_pixels))

        # Get all distinct colors (simplified - cluster to top N)
        unique_colors = self._get_distinct_colors(masked_pixels, n=5)
        colors = [Color.from_rgb(c) for c in unique_colors]

        # Texture type
        texture_type = self.texture_map.get(segment.texture_type, TextureType.ORGANIC)

        return MaterialProperties(
            base_color=base_color,
            color_variance=color_variance / 255.0,  # Normalize
            colors=colors,
            texture=texture_type,
            opacity=1.0,
            luminosity=segment.avg_luminosity,
            depth_hint=segment.depth_estimate,
            complexity=min(1.0, color_variance / 100.0),
        )

    def _get_distinct_colors(
        self,
        pixels: np.ndarray,
        n: int = 5
    ) -> List[Tuple[int, int, int]]:
        """Get N most distinct colors from pixels."""
        if len(pixels) < n:
            return [tuple(p) for p in pixels[:n]]

        # Use k-means to find distinct colors
        pixels_float = pixels.astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        try:
            _, _, centers = cv2.kmeans(
                pixels_float, n, None, criteria, 10, cv2.KMEANS_PP_CENTERS
            )
            return [tuple(c.astype(int)) for c in centers]
        except:
            # Fallback: just sample evenly
            indices = np.linspace(0, len(pixels) - 1, n, dtype=int)
            return [tuple(pixels[i]) for i in indices]

    def _calculate_importance(self, segment: Segment) -> float:
        """Calculate entity importance for construction ordering."""
        importance = 0.5  # Base

        # Focal segments are most important
        if segment.is_focal:
            importance += 0.3

        # Subjects are important
        if segment.segment_type == SegmentType.SUBJECT:
            importance += 0.2

        # Details get some importance
        if segment.segment_type == SegmentType.DETAIL:
            importance += 0.1

        # Very small or very large = less important (extremes)
        if segment.coverage < 0.01 or segment.coverage > 0.5:
            importance -= 0.1

        return max(0.0, min(1.0, importance))

    def _build_containment_relationships(
        self,
        segments: List[Segment],
        scene: SceneGraph,
        entity_map: Dict[str, str]
    ):
        """Build containment relationships (A contains B)."""
        for i, seg_a in enumerate(segments):
            for j, seg_b in enumerate(segments):
                if i == j:
                    continue

                # Check if A contains B (B is mostly inside A)
                if seg_a.area > seg_b.area:
                    # Check overlap
                    overlap = np.sum(seg_a.mask & seg_b.mask)
                    if overlap > seg_b.area * 0.8:  # B is 80%+ inside A
                        entity_a = entity_map[seg_a.id]
                        entity_b = entity_map[seg_b.id]

                        scene.add_relationship(
                            source_id=entity_a,
                            target_id=entity_b,
                            relationship_type=RelationshipType.CONTAINS,
                            strength=overlap / seg_b.area,
                        )

                        # Update parent/child
                        if entity_b_obj := scene.get_entity(entity_b):
                            entity_b_obj.parent_id = entity_a

    def _build_spatial_relationships(
        self,
        segments: List[Segment],
        scene: SceneGraph,
        entity_map: Dict[str, str]
    ):
        """Build spatial relationships (above, below, adjacent)."""
        for i, seg_a in enumerate(segments):
            for j, seg_b in enumerate(segments):
                if i >= j:  # Avoid duplicates
                    continue

                entity_a = entity_map[seg_a.id]
                entity_b = entity_map[seg_b.id]

                # Check adjacency (masks touch when dilated)
                kernel = np.ones((3, 3), np.uint8)
                dilated_a = cv2.dilate(seg_a.mask.astype(np.uint8), kernel)
                dilated_b = cv2.dilate(seg_b.mask.astype(np.uint8), kernel)

                if np.any(dilated_a & seg_b.mask.astype(np.uint8)) or \
                   np.any(dilated_b & seg_a.mask.astype(np.uint8)):

                    scene.add_relationship(
                        source_id=entity_a,
                        target_id=entity_b,
                        relationship_type=RelationshipType.ADJACENT,
                    )

                # Check above/below (based on centroid)
                y_diff = seg_a.centroid[1] - seg_b.centroid[1]
                if abs(y_diff) > seg_a.mask.shape[0] * 0.1:  # Significant vertical difference
                    if y_diff < 0:
                        scene.add_relationship(
                            source_id=entity_a,
                            target_id=entity_b,
                            relationship_type=RelationshipType.ABOVE,
                        )
                    else:
                        scene.add_relationship(
                            source_id=entity_a,
                            target_id=entity_b,
                            relationship_type=RelationshipType.BELOW,
                        )

                # Check depth relationship
                if abs(seg_a.depth_estimate - seg_b.depth_estimate) > 0.2:
                    if seg_a.depth_estimate > seg_b.depth_estimate:
                        scene.add_relationship(
                            source_id=entity_a,
                            target_id=entity_b,
                            relationship_type=RelationshipType.IN_FRONT,
                        )
                    else:
                        scene.add_relationship(
                            source_id=entity_a,
                            target_id=entity_b,
                            relationship_type=RelationshipType.BEHIND,
                        )

    def from_organic_layers(
        self,
        layers: List[Dict],
        image: np.ndarray
    ) -> SceneGraph:
        """
        Build scene graph from organic layer segmentation output.

        This provides compatibility with the existing organic segmentation.

        Args:
            layers: List of layer dicts from segment_into_painting_layers()
            image: Original RGB image

        Returns:
            SceneGraph
        """
        h, w = image.shape[:2]
        scene = SceneGraph(image_shape=(h, w))

        for layer in layers:
            mask = layer['mask']

            # Determine entity type
            if layer.get('priority', 0) <= 2:
                entity_type = EntityType.BACKGROUND
            elif 'foreground' in layer.get('name', '').lower():
                entity_type = EntityType.SUBJECT
            else:
                entity_type = EntityType.ENVIRONMENT

            # Get material properties
            masked_pixels = image[mask]
            if len(masked_pixels) > 0:
                median_color = np.median(masked_pixels, axis=0).astype(int)
                base_color = Color(int(median_color[0]), int(median_color[1]), int(median_color[2]))
            else:
                base_color = Color(128, 128, 128)

            properties = MaterialProperties(
                base_color=base_color,
                luminosity=layer.get('avg_luminosity', 0.5),
                depth_hint=1.0 - (layer.get('priority', 5) / 10.0),
            )

            # Calculate importance from priority (lower priority = higher importance for painting)
            importance = 0.3 + (layer.get('priority', 5) / 10.0) * 0.4

            scene.create_entity(
                name=layer.get('name', f"layer_{layer.get('priority', 0)}"),
                entity_type=entity_type,
                mask=mask,
                properties=properties,
                importance=importance,
                technique=layer.get('technique', 'layer'),
                coverage=layer.get('coverage', 0.0),
            )

        return scene
