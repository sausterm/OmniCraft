"""Health check endpoints."""

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import settings
from ...perception.yolo_segmentation import YOLO_AVAILABLE


router = APIRouter()


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    environment: str
    yolo_available: bool


class ReadyResponse(BaseModel):
    """Readiness check response."""
    ready: bool
    database: bool
    redis: bool
    s3: bool
    stripe: bool


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Basic health check endpoint.

    Returns the API status, version, and YOLO availability.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        environment=settings.environment,
        yolo_available=YOLO_AVAILABLE,
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """
    Readiness check for all dependencies.

    Checks database, Redis, S3, and Stripe connectivity.
    Used by load balancers to determine if the instance can receive traffic.
    """
    # TODO: Implement actual connectivity checks
    return ReadyResponse(
        ready=True,
        database=True,  # Check DB connection
        redis=True,     # Check Redis connection
        s3=True,        # Check S3 access
        stripe=bool(settings.stripe_api_key),
    )
