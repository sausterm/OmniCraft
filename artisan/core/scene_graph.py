"""
Scene Graph - Core data structures for representing artwork scenes.

This module provides the fundamental abstractions for understanding and
representing any visual scene, regardless of the output medium.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from abc import ABC, abstractmethod


class EntityType(Enum):
    """Classification of entities by their role in the scene."""
    BACKGROUND = "background"      # Sky, distant elements, base layers
    ENVIRONMENT = "environment"    # Mid-ground elements (grass, trees, water)
    SUBJECT = "subject"            # Main focal points (people, animals, objects)
    DETAIL = "detail"              # Fine details within subjects (eyes, textures)
    ACCENT = "accent"              # Highlights, shadows, finishing touches


class TextureType(Enum):
    """Classification of surface textures."""
    SMOOTH = "smooth"              # Glass, water, skin
    ROUGH = "rough"                # Stone, bark, concrete
    SOFT = "soft"                  # Fur, fabric, clouds
    METALLIC = "metallic"          # Metal, reflective surfaces
    ORGANIC = "organic"            # Natural, irregular patterns
    GEOMETRIC = "geometric"        # Regular patterns, man-made
    TRANSLUCENT = "translucent"    # Semi-transparent materials
    MATTE = "matte"                # Non-reflective surfaces


class RelationshipType(Enum):
    """Types of spatial/semantic relationships between entities."""
    CONTAINS = "contains"          # A contains B (face contains eyes)
    ADJACENT = "adjacent"          # A is next to B
    OVERLAPS = "overlaps"          # A and B share space
    ABOVE = "above"                # A is above B
    BELOW = "below"                # A is below B
    IN_FRONT = "in_front"          # A is in front of B (depth)
    BEHIND = "behind"              # A is behind B (depth)
    PART_OF = "part_of"            # A is a component of B


@dataclass
class Color:
    """RGB color with optional alpha."""
    r: int
    g: int
    b: int
    a: int = 255

    @property
    def rgb(self) -> Tuple[int, int, int]:
        return (self.r, self.g, self.b)

    @property
    def rgba(self) -> Tuple[int, int, int, int]:
        return (self.r, self.g, self.b, self.a)

    @property
    def normalized(self) -> Tuple[float, float, float]:
        return (self.r / 255.0, self.g / 255.0, self.b / 255.0)

    @classmethod
    def from_rgb(cls, rgb: Tuple[int, int, int]) -> 'Color':
        return cls(r=rgb[0], g=rgb[1], b=rgb[2])

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Color':
        return cls(r=int(arr[0]), g=int(arr[1]), b=int(arr[2]))

    def distance_to(self, other: 'Color') -> float:
        """Euclidean distance in RGB space."""
        return np.sqrt(
            (self.r - other.r) ** 2 +
            (self.g - other.g) ** 2 +
            (self.b - other.b) ** 2
        )


@dataclass
class MaterialProperties:
    """
    Properties describing what something is made of and how it appears.

    These properties inform both perception (understanding the source)
    and construction (how to recreate it in a given medium).
    """
    # Visual properties
    base_color: Color
    color_variance: float = 0.0        # How much color varies within region
    colors: List[Color] = field(default_factory=list)  # All colors in region

    # Surface properties
    texture: TextureType = TextureType.SMOOTH
    texture_scale: float = 1.0         # Relative size of texture pattern
    texture_direction: Optional[float] = None  # Angle in degrees, if directional

    # Optical properties
    opacity: float = 1.0               # 0 = transparent, 1 = opaque
    reflectance: float = 0.0           # 0 = matte, 1 = mirror
    luminosity: float = 0.5            # 0 = dark, 1 = bright

    # Spatial hints
    depth_hint: float = 0.5            # 0 = far background, 1 = closest foreground
    edge_sharpness: float = 0.5        # 0 = blurry/soft, 1 = sharp edges

    # Semantic hints
    natural: bool = True               # Natural vs man-made
    complexity: float = 0.5            # 0 = simple, 1 = highly detailed


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min

    @property
    def center(self) -> Tuple[int, int]:
        return ((self.x_min + self.x_max) // 2, (self.y_min + self.y_max) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    @classmethod
    def from_mask(cls, mask: np.ndarray) -> 'BoundingBox':
        """Create bounding box from binary mask."""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        return cls(x_min=int(x_min), y_min=int(y_min),
                   x_max=int(x_max), y_max=int(y_max))


@dataclass
class Entity:
    """
    An entity is anything that exists in the scene - an object, region, or element.

    This is the fundamental unit of the scene graph. Entities can be nested
    (a dog contains eyes, nose, ears) and have relationships with each other.
    """
    id: str
    name: str
    entity_type: EntityType

    # Spatial extent
    mask: np.ndarray                   # Binary mask (H x W)
    bbox: Optional[BoundingBox] = None

    # Properties
    properties: Optional[MaterialProperties] = None

    # Hierarchy
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)

    # Importance for construction ordering
    importance: float = 0.5            # 0 = least important, 1 = focal point

    # Semantic label (from segmentation model)
    semantic_label: Optional[str] = None
    confidence: float = 1.0

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.bbox is None and self.mask is not None:
            self.bbox = BoundingBox.from_mask(self.mask)

    @property
    def coverage(self) -> float:
        """Percentage of image covered by this entity."""
        if self.mask is None:
            return 0.0
        return np.sum(self.mask) / self.mask.size

    @property
    def centroid(self) -> Tuple[float, float]:
        """Center of mass of the entity."""
        if self.mask is None:
            return (0.0, 0.0)
        y_coords, x_coords = np.where(self.mask)
        return (float(np.mean(x_coords)), float(np.mean(y_coords)))


@dataclass
class Relationship:
    """A relationship between two entities."""
    source_id: str
    target_id: str
    relationship_type: RelationshipType
    strength: float = 1.0              # How strong/certain is this relationship
    metadata: Dict[str, Any] = field(default_factory=dict)


class SceneGraph:
    """
    A graph representation of a visual scene.

    The scene graph contains all entities, their properties, and relationships.
    It provides the foundation for understanding and reconstructing artwork.
    """

    def __init__(self, image_shape: Tuple[int, int]):
        """
        Initialize scene graph.

        Args:
            image_shape: (height, width) of the source image
        """
        self.image_shape = image_shape
        self.entities: Dict[str, Entity] = {}
        self.relationships: List[Relationship] = []
        self._entity_counter = 0

    def add_entity(self, entity: Entity) -> str:
        """Add an entity to the scene graph."""
        self.entities[entity.id] = entity
        return entity.id

    def create_entity(
        self,
        name: str,
        entity_type: EntityType,
        mask: np.ndarray,
        properties: Optional[MaterialProperties] = None,
        semantic_label: Optional[str] = None,
        parent_id: Optional[str] = None,
        importance: float = 0.5,
        **metadata
    ) -> Entity:
        """Create and add a new entity."""
        self._entity_counter += 1
        entity_id = f"entity_{self._entity_counter:04d}"

        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            mask=mask,
            properties=properties,
            semantic_label=semantic_label,
            parent_id=parent_id,
            importance=importance,
            metadata=metadata
        )

        # Update parent's children list
        if parent_id and parent_id in self.entities:
            self.entities[parent_id].children_ids.append(entity_id)

        self.add_entity(entity)
        return entity

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: RelationshipType,
        strength: float = 1.0,
        **metadata
    ) -> Relationship:
        """Add a relationship between two entities."""
        rel = Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            strength=strength,
            metadata=metadata
        )
        self.relationships.append(rel)
        return rel

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get an entity by ID."""
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a given type."""
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_children(self, entity_id: str) -> List[Entity]:
        """Get all children of an entity."""
        entity = self.get_entity(entity_id)
        if not entity:
            return []
        return [self.entities[cid] for cid in entity.children_ids if cid in self.entities]

    def get_parent(self, entity_id: str) -> Optional[Entity]:
        """Get the parent of an entity."""
        entity = self.get_entity(entity_id)
        if not entity or not entity.parent_id:
            return None
        return self.entities.get(entity.parent_id)

    def get_relationships_for(self, entity_id: str) -> List[Relationship]:
        """Get all relationships involving an entity."""
        return [r for r in self.relationships
                if r.source_id == entity_id or r.target_id == entity_id]

    def get_root_entities(self) -> List[Entity]:
        """Get all entities with no parent."""
        return [e for e in self.entities.values() if e.parent_id is None]

    def get_leaf_entities(self) -> List[Entity]:
        """Get all entities with no children."""
        return [e for e in self.entities.values() if not e.children_ids]

    def get_depth_ordered_entities(self) -> List[Entity]:
        """Get entities ordered by depth (background to foreground)."""
        entities = list(self.entities.values())
        return sorted(entities, key=lambda e: e.properties.depth_hint if e.properties else 0.5)

    def get_importance_ordered_entities(self) -> List[Entity]:
        """Get entities ordered by importance (least to most)."""
        entities = list(self.entities.values())
        return sorted(entities, key=lambda e: e.importance)

    def to_dict(self) -> Dict:
        """Serialize scene graph to dictionary."""
        return {
            "image_shape": self.image_shape,
            "entities": {
                eid: {
                    "id": e.id,
                    "name": e.name,
                    "entity_type": e.entity_type.value,
                    "coverage": e.coverage,
                    "importance": e.importance,
                    "semantic_label": e.semantic_label,
                    "parent_id": e.parent_id,
                    "children_ids": e.children_ids,
                    "centroid": e.centroid,
                    "bbox": {
                        "x_min": e.bbox.x_min,
                        "y_min": e.bbox.y_min,
                        "x_max": e.bbox.x_max,
                        "y_max": e.bbox.y_max
                    } if e.bbox else None,
                    "properties": {
                        "texture": e.properties.texture.value,
                        "opacity": e.properties.opacity,
                        "depth_hint": e.properties.depth_hint,
                        "luminosity": e.properties.luminosity,
                        "complexity": e.properties.complexity
                    } if e.properties else None
                }
                for eid, e in self.entities.items()
            },
            "relationships": [
                {
                    "source": r.source_id,
                    "target": r.target_id,
                    "type": r.relationship_type.value,
                    "strength": r.strength
                }
                for r in self.relationships
            ]
        }

    def __repr__(self) -> str:
        return (f"SceneGraph(entities={len(self.entities)}, "
                f"relationships={len(self.relationships)})")
