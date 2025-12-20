"""
Configuration settings for Artisan API.

All settings are loaded from environment variables with sensible defaults.
"""

from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Artisan Paint-by-Numbers API"
    app_version: str = "5.0.0"
    debug: bool = False
    environment: str = "development"  # development, staging, production

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4

    # CORS - Allow all origins for beta testing
    # TODO: Restrict to specific origins for production
    cors_origins: List[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )

    # Database
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/artisan"

    # Redis (for Celery)
    redis_url: str = "redis://localhost:6379/0"

    # AWS S3
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: str = "us-east-1"
    s3_bucket_uploads: str = "artisan-uploads"
    s3_bucket_outputs: str = "artisan-outputs"

    # Stripe
    stripe_api_key: Optional[str] = None
    stripe_webhook_secret: Optional[str] = None
    stripe_success_url: str = "http://localhost:3000/download/{job_id}"
    stripe_cancel_url: str = "http://localhost:3000/checkout?canceled=true"

    # Processing
    max_upload_size_mb: int = 20
    max_image_dimension: int = 4096
    default_model_size: str = "m"  # YOLO model size: n, s, m, l, x
    default_conf_threshold: float = 0.2

    # Rate limiting
    rate_limit_uploads: int = 10  # per minute
    rate_limit_process: int = 5   # per minute

    # Product pricing (in cents)
    price_preview: int = 0
    price_basic: int = 499      # $4.99
    price_standard: int = 999   # $9.99
    price_premium: int = 1999   # $19.99
    price_paint_kit: int = 299  # $2.99
    price_mixing_guide: int = 199  # $1.99

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()


# Product definitions
PRODUCTS = {
    "preview": {
        "name": "Free Preview",
        "price": settings.price_preview,
        "includes": ["low_res_preview", "scene_analysis"],
        "description": "Low-resolution preview of your painting guide"
    },
    "basic": {
        "name": "Basic Package",
        "price": settings.price_basic,
        "includes": ["cumulative_images", "scene_analysis", "progress_overview"],
        "description": "Step-by-step cumulative images and scene analysis"
    },
    "standard": {
        "name": "Standard Package",
        "price": settings.price_standard,
        "includes": [
            "cumulative_images", "context_images", "isolated_images",
            "scene_analysis", "painting_guide", "progress_overview"
        ],
        "description": "Full image set with detailed painting guide"
    },
    "premium": {
        "name": "Premium Package",
        "price": settings.price_premium,
        "includes": [
            "cumulative_images", "context_images", "isolated_images",
            "scene_analysis", "painting_guide", "progress_overview",
            "paint_kit", "mixing_guide"
        ],
        "description": "Everything including paint shopping list and mixing instructions"
    },
    # Add-ons
    "paint_kit": {
        "name": "Paint Kit Add-on",
        "price": settings.price_paint_kit,
        "includes": ["paint_kit"],
        "description": "Shopping list with real paint brands and quantities"
    },
    "mixing_guide": {
        "name": "Mixing Guide Add-on",
        "price": settings.price_mixing_guide,
        "includes": ["mixing_guide"],
        "description": "Step-by-step color mixing instructions"
    },
}
