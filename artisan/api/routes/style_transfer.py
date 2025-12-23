"""
Style Transfer endpoint - apply artistic styles to uploaded images.
"""

import os
import logging
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from PIL import Image

from ..config import settings
from ..models.schemas import JobStatus
from .upload import jobs_db


# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

router = APIRouter()


# Available style presets
STYLE_PRESETS = [
    "pop_art",
    "britto_style",
    "van_gogh",
    "picasso_cubist",
    "anime",
    "watercolor",
    "oil_painting",
    "sketch",
    "custom",
]


class StyleTransferRequest(BaseModel):
    """Request to apply style transfer to an image."""
    job_id: str = Field(..., description="Job ID from upload")
    style: str = Field(
        default="van_gogh",
        description=f"Style preset to apply. Options: {', '.join(STYLE_PRESETS)}"
    )
    custom_prompt: Optional[str] = Field(
        default=None,
        description="Custom style description (only used when style='custom')"
    )
    guidance_scale: float = Field(
        default=7.5,
        ge=1.0,
        le=20.0,
        description="How closely to follow the style (1-20, higher = more stylized)"
    )
    control_strength: float = Field(
        default=1.0,
        ge=0.3,
        le=2.0,
        description="How much to preserve original structure (0.3-2.0, higher = more structure)"
    )
    num_steps: int = Field(
        default=30,
        ge=10,
        le=50,
        description="Number of inference steps (10-50, higher = better quality but slower)"
    )


class StyleTransferResponse(BaseModel):
    """Response after starting style transfer."""
    job_id: str
    status: str
    message: str
    style: str
    estimated_time_seconds: int = Field(
        default=180,
        description="Estimated processing time in seconds"
    )


class StyleInfo(BaseModel):
    """Information about a style preset."""
    id: str
    name: str
    description: str


class StyleListResponse(BaseModel):
    """List of available styles."""
    styles: List[StyleInfo]


@router.get("/styles", response_model=StyleListResponse)
async def list_styles():
    """
    List all available style presets.

    Returns preset names and descriptions for the UI.
    """
    style_info = {
        "pop_art": StyleInfo(
            id="pop_art",
            name="Pop Art",
            description="Bold colors, thick black outlines, inspired by Warhol"
        ),
        "britto_style": StyleInfo(
            id="britto_style",
            name="Romero Britto",
            description="Geometric pop art with hearts, dots, stripes, and vibrant colors"
        ),
        "van_gogh": StyleInfo(
            id="van_gogh",
            name="Van Gogh",
            description="Post-impressionist swirling brushstrokes, expressive colors"
        ),
        "picasso_cubist": StyleInfo(
            id="picasso_cubist",
            name="Picasso Cubism",
            description="Geometric shapes, fragmented forms, multiple perspectives"
        ),
        "anime": StyleInfo(
            id="anime",
            name="Anime",
            description="Japanese animation style, clean lines, cel shading"
        ),
        "watercolor": StyleInfo(
            id="watercolor",
            name="Watercolor",
            description="Soft edges, translucent colors, artistic bleeding effects"
        ),
        "oil_painting": StyleInfo(
            id="oil_painting",
            name="Oil Painting",
            description="Rich colors, visible brushstrokes, classical technique"
        ),
        "sketch": StyleInfo(
            id="sketch",
            name="Pencil Sketch",
            description="Hand-drawn pencil style, detailed linework, cross-hatching"
        ),
        "custom": StyleInfo(
            id="custom",
            name="Custom Style",
            description="Provide your own style description"
        ),
    }

    return StyleListResponse(
        styles=[style_info[s] for s in STYLE_PRESETS]
    )


def run_style_transfer(job_id: str, request: StyleTransferRequest):
    """
    Background task to run style transfer using Replicate API.

    Uses cloud-based Replicate API for fast processing (30-60 seconds)
    instead of local ControlNet which takes 20+ minutes on CPU.
    """
    job = jobs_db.get(job_id)
    if not job:
        logger.error(f"[{job_id}] Job not found in database")
        return

    try:
        logger.info(f"[{job_id}] Starting style transfer (Replicate API)")
        logger.info(f"[{job_id}] Style: {request.style}, guidance: {request.guidance_scale}")

        # Update status
        job["style_status"] = "processing"
        job["updated_at"] = datetime.utcnow()

        # Get input path
        input_path = job["local_path"]
        logger.info(f"[{job_id}] Input file: {input_path}")

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Load image
        image = Image.open(input_path).convert("RGB")
        logger.info(f"[{job_id}] Loaded image: {image.size}")

        # Try Replicate API first (fast, cloud-based)
        try:
            from artisan.transfer.engines.replicate_engine import ReplicateEngine
            logger.info(f"[{job_id}] Using Replicate API for style transfer")
            engine = ReplicateEngine()

            result = engine.apply_style(
                image=image,
                style=request.style,
                custom_prompt=request.custom_prompt,
                guidance_scale=request.guidance_scale,
                num_inference_steps=request.num_steps,
                controlnet_conditioning_scale=request.control_strength,
            )

        except (ImportError, ValueError) as e:
            # Fall back to local ControlNet if Replicate not available
            logger.warning(f"[{job_id}] Replicate unavailable ({e}), falling back to local ControlNet")
            from artisan.transfer.engines.controlnet_engine import ControlNetEngine
            engine = ControlNetEngine()

            kwargs = {
                "num_inference_steps": request.num_steps,
                "guidance_scale": request.guidance_scale,
                "controlnet_conditioning_scale": request.control_strength,
            }

            if request.style == "custom" and request.custom_prompt:
                result = engine.apply_style(
                    image=image,
                    style="custom",
                    custom_prompt=request.custom_prompt,
                    **kwargs
                )
            else:
                result = engine.apply_style(
                    image=image,
                    style=request.style,
                    **kwargs
                )

        # Save styled image
        output_dir = f"artisan/api/outputs/{job_id}"
        os.makedirs(output_dir, exist_ok=True)

        styled_path = os.path.join(output_dir, "styled_image.png")
        result.save(styled_path)
        logger.info(f"[{job_id}] Saved styled image to: {styled_path}")

        # Update job record
        job["styled_image_path"] = styled_path
        job["style_config"] = {
            "style": request.style,
            "custom_prompt": request.custom_prompt,
            "guidance_scale": request.guidance_scale,
            "control_strength": request.control_strength,
            "num_steps": request.num_steps,
            "processing_time": result.processing_time,
        }
        job["style_status"] = "completed"
        job["updated_at"] = datetime.utcnow()

        logger.info(f"[{job_id}] Style transfer completed in {result.processing_time:.1f}s")

    except Exception as e:
        logger.exception(f"[{job_id}] Style transfer failed: {e}")
        job["style_status"] = "failed"
        job["style_error"] = str(e)
        job["updated_at"] = datetime.utcnow()


@router.post("/style-transfer", response_model=StyleTransferResponse)
async def apply_style_transfer(
    request: StyleTransferRequest,
    background_tasks: BackgroundTasks,
):
    """
    Apply artistic style transfer to an uploaded image.

    This starts a background process that typically takes 3-5 minutes.
    Poll /api/job/{job_id} to check status.

    The styled image will be used for paint-by-numbers generation
    instead of the original when you call /api/process.
    """
    # Validate job exists
    if request.job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[request.job_id]

    # Check job has an image
    if not job.get("local_path") or not os.path.exists(job["local_path"]):
        raise HTTPException(status_code=400, detail="No image found for this job")

    # Validate style
    if request.style not in STYLE_PRESETS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid style '{request.style}'. Valid options: {', '.join(STYLE_PRESETS)}"
        )

    # Custom style requires a prompt
    if request.style == "custom" and not request.custom_prompt:
        raise HTTPException(
            status_code=400,
            detail="custom_prompt is required when style is 'custom'"
        )

    # Check if already processing
    if job.get("style_status") == "processing":
        raise HTTPException(
            status_code=409,
            detail="Style transfer already in progress for this job"
        )

    # Start background task
    job["style_status"] = "queued"
    job["updated_at"] = datetime.utcnow()

    background_tasks.add_task(run_style_transfer, request.job_id, request)

    return StyleTransferResponse(
        job_id=request.job_id,
        status="queued",
        message="Style transfer started. This typically takes 30-60 seconds. Poll /api/job/{job_id} to check status.",
        style=request.style,
        estimated_time_seconds=45 + (request.num_steps - 20) * 1,  # Base 45s + extra for more steps
    )


@router.get("/style-transfer/{job_id}")
async def get_style_transfer_status(job_id: str):
    """
    Get the status of a style transfer job.

    Returns the styled image URL when complete.
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    style_status = job.get("style_status", "not_started")
    style_config = job.get("style_config", {})
    styled_path = job.get("styled_image_path")
    style_error = job.get("style_error")

    response = {
        "job_id": job_id,
        "style_status": style_status,
        "style_config": style_config,
    }

    if style_status == "completed" and styled_path:
        # Return URL to styled image
        response["styled_image_url"] = f"/api/preview/{job_id}/styled"
        response["processing_time"] = style_config.get("processing_time")

    if style_status == "failed":
        response["error"] = style_error

    return response


@router.delete("/style-transfer/{job_id}")
async def remove_style_transfer(job_id: str):
    """
    Remove style transfer from a job, reverting to original image.

    The next /api/process call will use the original image.
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    # Remove styled image if it exists
    styled_path = job.get("styled_image_path")
    if styled_path and os.path.exists(styled_path):
        try:
            os.remove(styled_path)
        except Exception as e:
            logger.warning(f"Failed to delete styled image: {e}")

    # Clear style data from job
    job["styled_image_path"] = None
    job["style_config"] = None
    job["style_status"] = None
    job["style_error"] = None
    job["updated_at"] = datetime.utcnow()

    return {"job_id": job_id, "message": "Style transfer removed. Original image will be used."}
