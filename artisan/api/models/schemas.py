"""
Pydantic schemas for API request/response models.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Processing job status."""
    PENDING = "pending"
    UPLOADING = "uploading"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessConfig(BaseModel):
    """Configuration for image processing."""
    model_size: str = Field(
        default="m",
        description="YOLO model size: n (nano), s (small), m (medium), l (large), x (xlarge)"
    )
    conf_threshold: float = Field(
        default=0.2,
        ge=0.1,
        le=0.9,
        description="Confidence threshold for object detection (0.1-0.9)"
    )
    substeps_per_region: int = Field(
        default=4,
        ge=2,
        le=6,
        description="Number of substeps per semantic region (2-6)"
    )


class JobCreate(BaseModel):
    """Request to create a new job."""
    filename: str
    content_type: str
    file_size: int


class JobResponse(BaseModel):
    """Job information response."""
    job_id: str
    status: JobStatus
    filename: str
    created_at: datetime
    updated_at: datetime
    config: Optional[ProcessConfig] = None
    progress: float = 0.0  # 0-100
    error: Optional[str] = None

    # Results (when completed)
    scene_context: Optional[Dict] = None
    num_regions: Optional[int] = None
    num_substeps: Optional[int] = None
    preview_url: Optional[str] = None


class ProcessResponse(BaseModel):
    """Response after starting processing."""
    job_id: str
    status: JobStatus
    message: str
    estimated_time_seconds: Optional[int] = None


class SceneAnalysis(BaseModel):
    """Scene analysis results."""
    time_of_day: str
    weather: str
    setting: str
    lighting: str
    mood: str
    light_direction: str


class RegionInfo(BaseModel):
    """Information about a detected region."""
    name: str
    subject_type: str
    category: str
    coverage: float
    is_focal: bool
    substeps: int


class ProcessingResult(BaseModel):
    """Complete processing results."""
    job_id: str
    scene_context: SceneAnalysis
    regions: List[RegionInfo]
    total_substeps: int
    output_files: Dict[str, int]  # {"cumulative": 26, "context": 25, "isolated": 25}


class ProductInfo(BaseModel):
    """Product information."""
    id: str
    name: str
    price: int  # in cents
    price_formatted: str
    description: str
    includes: List[str]


class ProductResponse(BaseModel):
    """Available products for a job."""
    job_id: str
    products: List[ProductInfo]
    purchased: List[str] = []


class CheckoutRequest(BaseModel):
    """Request to create a checkout session."""
    job_id: str
    product_ids: List[str] = Field(
        ...,
        min_length=1,
        description="List of product IDs to purchase"
    )
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class CheckoutResponse(BaseModel):
    """Checkout session response."""
    checkout_url: str
    session_id: str


class DownloadFile(BaseModel):
    """Information about a downloadable file."""
    name: str
    url: str
    size_bytes: int
    expires_at: datetime


class DownloadResponse(BaseModel):
    """Download URLs for purchased products."""
    job_id: str
    purchased_products: List[str]
    files: List[DownloadFile]
    expires_at: datetime


class WebhookEvent(BaseModel):
    """Stripe webhook event."""
    type: str
    data: Dict
