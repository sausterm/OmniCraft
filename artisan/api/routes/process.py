"""
Processing endpoint - start and monitor image processing.
"""

import os
import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, BackgroundTasks

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

from ..config import settings
from ..models.schemas import (
    ProcessConfig,
    ProcessResponse,
    JobStatus,
    ProcessingResult,
    SceneAnalysis,
    RegionInfo,
)
from .upload import jobs_db


router = APIRouter()


def run_processing(job_id: str, config: ProcessConfig):
    """
    Background task to run the YOLO Bob Ross processing.

    This runs in a separate thread to not block the API.
    In production, this would be a Celery task.
    """
    from artisan.paint.generators.yolo_bob_ross_paint import YOLOBobRossPaint

    job = jobs_db.get(job_id)
    if not job:
        logger.error(f"[{job_id}] Job not found in database")
        return

    try:
        logger.info(f"[{job_id}] Starting processing job")
        logger.info(f"[{job_id}] Config: model_size={config.model_size}, conf={config.conf_threshold}, substeps={config.substeps_per_region}")

        # Update status
        job["status"] = JobStatus.PROCESSING
        job["updated_at"] = datetime.utcnow()
        job["progress"] = 10.0

        # Get input path - use styled image if available
        styled_path = job.get("styled_image_path")
        if styled_path and os.path.exists(styled_path):
            input_path = styled_path
            logger.info(f"[{job_id}] Using styled image: {input_path}")
        else:
            input_path = job["local_path"]
            logger.info(f"[{job_id}] Using original image: {input_path}")

        # Create output directory
        output_dir = f"artisan/api/outputs/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"[{job_id}] Output directory: {output_dir}")

        # Run processing
        job["progress"] = 20.0
        logger.info(f"[{job_id}] Initializing YOLOBobRossPaint...")

        painter = YOLOBobRossPaint(
            input_path,
            model_size=config.model_size,
            conf_threshold=config.conf_threshold,
            substeps_per_region=config.substeps_per_region,
        )

        job["progress"] = 40.0
        logger.info(f"[{job_id}] Running scene analysis and segmentation...")
        painter.process()

        job["progress"] = 70.0
        logger.info(f"[{job_id}] Generating step images and instructions...")
        painter.save_all(output_dir)

        job["progress"] = 90.0

        # Extract results
        scene_ctx = painter.scene_context
        results = {
            "scene_context": {
                "time_of_day": scene_ctx.time_of_day.value,
                "weather": scene_ctx.weather.value,
                "setting": scene_ctx.setting.value,
                "lighting": scene_ctx.lighting.value,
                "mood": scene_ctx.mood.value,
                "light_direction": scene_ctx.light_direction,
            },
            "num_regions": len(painter.painting_layers),
            "num_substeps": sum(len(l.substeps) for l in painter.painting_layers),
            "regions": [
                {
                    "name": layer.name,
                    "category": layer.category,
                    "coverage": layer.coverage,
                    "is_focal": layer.is_focal,
                    "substeps": len(layer.substeps),
                }
                for layer in painter.painting_layers
            ],
            "output_dir": output_dir,
        }

        # Count output files
        results["output_files"] = {
            "cumulative": len(os.listdir(os.path.join(output_dir, "steps", "cumulative"))),
            "context": len(os.listdir(os.path.join(output_dir, "steps", "context"))),
            "isolated": len(os.listdir(os.path.join(output_dir, "steps", "isolated"))),
        }

        # Log output files
        logger.info(f"[{job_id}] Generated {results['output_files']['cumulative']} cumulative images")
        logger.info(f"[{job_id}] Generated {results['output_files']['context']} context images")
        logger.info(f"[{job_id}] Generated {results['output_files']['isolated']} isolated images")
        logger.info(f"[{job_id}] Total regions: {results['num_regions']}, Total substeps: {results['num_substeps']}")

        # Update job
        job["status"] = JobStatus.COMPLETED
        job["progress"] = 100.0
        job["results"] = results
        job["updated_at"] = datetime.utcnow()

        # Generate preview URL (low-res version of progress overview)
        job["preview_url"] = f"/api/preview/{job_id}"

        logger.info(f"[{job_id}] Job completed successfully!")

    except Exception as e:
        logger.error(f"[{job_id}] Job failed with error: {str(e)}", exc_info=True)
        job["status"] = JobStatus.FAILED
        job["error"] = str(e)
        job["updated_at"] = datetime.utcnow()
        raise


@router.post("/process/{job_id}", response_model=ProcessResponse)
async def start_processing(
    job_id: str,
    config: ProcessConfig = ProcessConfig(),
    background_tasks: BackgroundTasks = None,
):
    """
    Start processing an uploaded image.

    - **job_id**: The job ID from upload
    - **config**: Processing configuration (optional)

    Processing runs in the background. Poll /api/job/{job_id} for status.
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    # Check if already processing or completed
    if job["status"] == JobStatus.PROCESSING:
        raise HTTPException(status_code=400, detail="Job is already processing")

    if job["status"] == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job already completed")

    # Store config
    job["config"] = config
    job["status"] = JobStatus.PROCESSING
    job["updated_at"] = datetime.utcnow()

    # Start background processing
    background_tasks.add_task(run_processing, job_id, config)

    # Estimate processing time based on model size
    time_estimates = {"n": 10, "s": 15, "m": 25, "l": 40, "x": 60}
    estimated_time = time_estimates.get(config.model_size, 30)

    return ProcessResponse(
        job_id=job_id,
        status=JobStatus.PROCESSING,
        message="Processing started. Poll /api/job/{job_id} for status.",
        estimated_time_seconds=estimated_time,
    )


@router.get("/status/{job_id}")
async def get_processing_status(job_id: str):
    """
    Get detailed processing status.

    - **job_id**: The job ID
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job.get("progress", 0),
        "error": job.get("error"),
        "updated_at": job["updated_at"],
    }


@router.get("/results/{job_id}", response_model=ProcessingResult)
async def get_results(job_id: str):
    """
    Get processing results for a completed job.

    - **job_id**: The job ID
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Status: {job['status']}"
        )

    results = job["results"]

    return ProcessingResult(
        job_id=job_id,
        scene_context=SceneAnalysis(**results["scene_context"]),
        regions=[RegionInfo(**r, subject_type="unknown") for r in results["regions"]],
        total_substeps=results["num_substeps"],
        output_files=results["output_files"],
    )
