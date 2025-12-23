"""
Upload endpoint for image files.
"""

import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

from ..config import settings
from ..models.schemas import JobResponse, JobStatus


router = APIRouter()


# In-memory job storage (replace with database in production)
jobs_db: dict = {}


class UploadResponse(BaseModel):
    """Response after successful upload."""
    job_id: str
    filename: str
    status: JobStatus
    message: str


def validate_image(file: UploadFile) -> None:
    """Validate uploaded file is a valid image."""
    # Check content type
    allowed_types = ["image/jpeg", "image/png", "image/webp"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Allowed: {allowed_types}"
        )

    # Check file size (will be verified after reading)
    # Note: This is approximate from headers
    if hasattr(file, 'size') and file.size:
        max_size = settings.max_upload_size_mb * 1024 * 1024
        if file.size > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_upload_size_mb}MB"
            )


@router.post("/upload", response_model=UploadResponse)
async def upload_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, or WebP)"),
    background_tasks: BackgroundTasks = None,
):
    """
    Upload an image for processing.

    - **file**: Image file (JPEG, PNG, or WebP, max 20MB)

    Returns a job_id to track the processing status.
    """
    # Validate file
    validate_image(file)

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Read file content
    content = await file.read()
    file_size = len(content)

    # Verify file size
    max_size = settings.max_upload_size_mb * 1024 * 1024
    if file_size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file_size / 1024 / 1024:.1f}MB. Maximum: {settings.max_upload_size_mb}MB"
        )

    # TODO: Upload to S3
    # For now, save locally for development
    import os
    upload_dir = "artisan/api/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # Save with job_id as filename
    ext = file.filename.split('.')[-1] if '.' in file.filename else 'png'
    local_path = os.path.join(upload_dir, f"{job_id}.{ext}")
    with open(local_path, 'wb') as f:
        f.write(content)

    # Create job record
    now = datetime.utcnow()
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": JobStatus.PENDING,
        "filename": file.filename,
        "content_type": file.content_type,
        "file_size": file_size,
        "local_path": local_path,
        "created_at": now,
        "updated_at": now,
        "config": None,
        "progress": 0.0,
        "error": None,
        "results": None,
        "email": None,  # User email for login/receipts
        "user_id": None,  # Linked user account (after login)
        # Style transfer fields
        "styled_image_path": None,  # Path to styled image (if style applied)
        "style_config": None,  # Style settings used
        "style_status": None,  # not_started, queued, processing, completed, failed
        "style_error": None,  # Error message if style transfer failed
    }

    return UploadResponse(
        job_id=job_id,
        filename=file.filename,
        status=JobStatus.PENDING,
        message="Image uploaded successfully. Use /api/process to start processing."
    )


@router.get("/job/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """
    Get job status and information.

    - **job_id**: The job ID returned from upload
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    return JobResponse(
        job_id=job["job_id"],
        status=job["status"],
        filename=job["filename"],
        created_at=job["created_at"],
        updated_at=job["updated_at"],
        config=job.get("config"),
        progress=job.get("progress", 0.0),
        error=job.get("error"),
        scene_context=job.get("results", {}).get("scene_context") if job.get("results") else None,
        num_regions=job.get("results", {}).get("num_regions") if job.get("results") else None,
        num_substeps=job.get("results", {}).get("num_substeps") if job.get("results") else None,
        preview_url=job.get("preview_url"),
        style_status=job.get("style_status"),
        style_config=job.get("style_config"),
        styled_image_url=f"/api/preview/{job_id}/styled" if job.get("styled_image_path") else None,
    )


class UpdateEmailRequest(BaseModel):
    """Request to update job email."""
    email: str


class UpdateEmailResponse(BaseModel):
    """Response after updating email."""
    job_id: str
    email: str
    message: str


@router.patch("/job/{job_id}/email", response_model=UpdateEmailResponse)
async def update_job_email(job_id: str, request: UpdateEmailRequest):
    """
    Update the email associated with a job.

    - **job_id**: The job ID
    - **email**: User's email address
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    # Basic email validation
    import re
    if not re.match(r"[^@]+@[^@]+\.[^@]+", request.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    jobs_db[job_id]["email"] = request.email
    jobs_db[job_id]["updated_at"] = datetime.utcnow()

    return UpdateEmailResponse(
        job_id=job_id,
        email=request.email,
        message="Email updated successfully"
    )


# Export jobs_db for other routes
def get_jobs_db():
    """Get the jobs database (for dependency injection)."""
    return jobs_db
