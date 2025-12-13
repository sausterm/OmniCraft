"""
Base classes for style transfer engines
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from pathlib import Path
from PIL import Image
import time


@dataclass
class StyleResult:
    """Result from a style transfer operation"""
    image: Image.Image
    style_name: str
    engine_name: str
    metadata: Dict[str, Any]
    processing_time: float

    def save(self, output_path: str, **kwargs):
        """Save the styled image"""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        self.image.save(output_path, **kwargs)
        return output_path


class StyleTransferEngine(ABC):
    """
    Abstract base class for style transfer engines.

    Each engine implements a specific approach to style transfer:
    - Local algorithms (Britto, Van Gogh, etc.)
    - API-based services (Replicate, Stability AI, etc.)
    - ML models (StyleGAN, Neural Style Transfer, etc.)
    """

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._styles: Dict[str, Dict[str, Any]] = {}

    @abstractmethod
    def apply_style(
        self,
        image: Image.Image,
        style: str,
        **kwargs
    ) -> StyleResult:
        """
        Apply a style to an image.

        Args:
            image: Input PIL Image
            style: Name of the style to apply
            **kwargs: Style-specific parameters

        Returns:
            StyleResult with the styled image and metadata
        """
        pass

    @abstractmethod
    def get_available_styles(self) -> List[str]:
        """Return list of styles this engine can apply"""
        pass

    def get_style_info(self, style: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific style"""
        return self._styles.get(style)

    def register_style(self, style_name: str, **metadata):
        """Register a style with this engine"""
        self._styles[style_name] = metadata

    def supports_style(self, style: str) -> bool:
        """Check if this engine supports a given style"""
        return style in self.get_available_styles()

    def _create_result(
        self,
        image: Image.Image,
        style: str,
        metadata: Dict[str, Any],
        start_time: float
    ) -> StyleResult:
        """Helper to create a StyleResult"""
        return StyleResult(
            image=image,
            style_name=style,
            engine_name=self.name,
            metadata=metadata,
            processing_time=time.time() - start_time
        )


class LocalStyleEngine(StyleTransferEngine):
    """
    Base class for engines that process images locally
    (no external API calls)
    """

    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.requires_api = False


class APIStyleEngine(StyleTransferEngine):
    """
    Base class for engines that use external APIs
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None
    ):
        super().__init__(name, description)
        self.requires_api = True
        self.api_key = api_key
        self.api_base_url = api_base_url

    @abstractmethod
    def _validate_api_credentials(self) -> bool:
        """Validate that API credentials are properly configured"""
        pass

    def get_cost_estimate(self, style: str, image_size: tuple) -> Optional[float]:
        """
        Estimate the cost of applying a style (if applicable).

        Returns:
            Cost in USD, or None if not applicable
        """
        return None
