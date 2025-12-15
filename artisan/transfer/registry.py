"""
Global registry for style transfer engines
"""

from typing import Dict, List, Optional
from .base import StyleTransferEngine


class StyleRegistry:
    """
    Central registry for all style transfer engines.
    Allows discovery and selection of engines and styles.
    """

    def __init__(self):
        self._engines: Dict[str, StyleTransferEngine] = {}
        self._style_to_engines: Dict[str, List[str]] = {}

    def register_engine(self, engine: StyleTransferEngine):
        """Register a new style transfer engine"""
        self._engines[engine.name] = engine

        # Index styles for quick lookup
        for style in engine.get_available_styles():
            if style not in self._style_to_engines:
                self._style_to_engines[style] = []
            self._style_to_engines[style].append(engine.name)

    def get_engine(self, name: str) -> Optional[StyleTransferEngine]:
        """Get an engine by name"""
        return self._engines.get(name)

    def get_engines_for_style(self, style: str) -> List[StyleTransferEngine]:
        """Get all engines that support a given style"""
        engine_names = self._style_to_engines.get(style, [])
        return [self._engines[name] for name in engine_names]

    def get_all_engines(self) -> List[StyleTransferEngine]:
        """Get all registered engines"""
        return list(self._engines.values())

    def get_all_styles(self) -> List[str]:
        """Get all available styles across all engines"""
        return list(self._style_to_engines.keys())

    def list_styles_by_engine(self) -> Dict[str, List[str]]:
        """Get a mapping of engine names to their supported styles"""
        return {
            name: engine.get_available_styles()
            for name, engine in self._engines.items()
        }

    def find_best_engine(self, style: str, prefer_local: bool = True) -> Optional[StyleTransferEngine]:
        """
        Find the best engine for a given style.

        Args:
            style: The style to apply
            prefer_local: Prefer local engines over API engines

        Returns:
            The best matching engine, or None if no engine supports the style
        """
        engines = self.get_engines_for_style(style)
        if not engines:
            return None

        if prefer_local:
            # Prefer local engines
            local_engines = [e for e in engines if not e.requires_api]
            if local_engines:
                return local_engines[0]

        return engines[0]


# Global registry instance
_global_registry = StyleRegistry()


def register_style(engine: StyleTransferEngine):
    """Register an engine with the global registry"""
    _global_registry.register_engine(engine)


def get_engine(name: str) -> Optional[StyleTransferEngine]:
    """Get an engine from the global registry"""
    return _global_registry.get_engine(name)


def get_registry() -> StyleRegistry:
    """Get the global registry instance"""
    return _global_registry
