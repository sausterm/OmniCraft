"""
Artisan API - FastAPI backend for paint-by-numbers generation.

Provides REST API endpoints for:
- Image upload and processing
- Job status tracking
- Stripe payment integration
- Downloadable product delivery
"""

from .main import app, create_app

__all__ = ["app", "create_app"]
