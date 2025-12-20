"""
Download endpoints - serve generated files to paying customers.
"""

import os
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
import zipfile
import io

from ..config import settings, PRODUCTS
from ..models.schemas import (
    JobStatus,
    ProductInfo,
    ProductResponse,
    DownloadResponse,
    DownloadFile,
)
from .upload import jobs_db


router = APIRouter()


# Track purchases (replace with database in production)
purchases_db: dict = {}


def get_product_info(product_id: str) -> ProductInfo:
    """Get formatted product information."""
    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail=f"Product not found: {product_id}")

    product = PRODUCTS[product_id]
    return ProductInfo(
        id=product_id,
        name=product["name"],
        price=product["price"],
        price_formatted=f"${product['price'] / 100:.2f}",
        description=product["description"],
        includes=product["includes"],
    )


@router.get("/products/{job_id}", response_model=ProductResponse)
async def get_products(job_id: str):
    """
    Get available products for a completed job.

    - **job_id**: The job ID
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail="Job not completed. Products available after processing."
        )

    # Get purchased products for this job
    purchased = purchases_db.get(job_id, [])

    # Build product list
    products = [get_product_info(pid) for pid in PRODUCTS.keys()]

    return ProductResponse(
        job_id=job_id,
        products=products,
        purchased=purchased,
    )


@router.get("/image/{job_id}/{image_type}/{index}")
async def get_step_image(job_id: str, image_type: str, index: int):
    """
    Get a specific step image.

    - **job_id**: The job ID
    - **image_type**: cumulative, context, or isolated
    - **index**: Step index (0-based)
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    if image_type not in ["cumulative", "context", "isolated"]:
        raise HTTPException(status_code=400, detail="Invalid image type")

    output_dir = job["results"]["output_dir"]
    image_dir = os.path.join(output_dir, "steps", image_type)

    if not os.path.exists(image_dir):
        raise HTTPException(status_code=404, detail=f"Image directory not found: {image_type}")

    # Get sorted list of images
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    if index < 0 or index >= len(images):
        raise HTTPException(status_code=404, detail=f"Image index out of range: {index}")

    image_path = os.path.join(image_dir, images[index])

    return FileResponse(
        image_path,
        media_type="image/png",
        filename=f"{image_type}_{index}.png"
    )


@router.get("/preview/{job_id}")
async def get_preview(job_id: str):
    """
    Get free low-resolution preview.

    - **job_id**: The job ID
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    # Return progress overview as preview
    output_dir = job["results"]["output_dir"]
    preview_path = os.path.join(output_dir, "progress_overview.png")

    if not os.path.exists(preview_path):
        raise HTTPException(status_code=404, detail="Preview not found")

    return FileResponse(
        preview_path,
        media_type="image/png",
        filename=f"preview_{job_id}.png"
    )


@router.get("/download/{job_id}/{product_id}")
async def download_product(job_id: str, product_id: str):
    """
    Download purchased product files.

    - **job_id**: The job ID
    - **product_id**: The product ID (basic, standard, premium, etc.)
    """
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    # Check if purchased (skip for preview)
    purchased = purchases_db.get(job_id, [])
    if product_id != "preview" and product_id not in purchased:
        raise HTTPException(
            status_code=403,
            detail=f"Product not purchased. Please purchase {product_id} first."
        )

    if product_id not in PRODUCTS:
        raise HTTPException(status_code=404, detail="Product not found")

    # Get included files
    product = PRODUCTS[product_id]
    includes = product["includes"]
    output_dir = job["results"]["output_dir"]

    # Build file list based on what's included
    files_to_include = []

    if "low_res_preview" in includes:
        files_to_include.append(os.path.join(output_dir, "progress_overview.png"))

    if "scene_analysis" in includes:
        files_to_include.append(os.path.join(output_dir, "scene_analysis.json"))

    if "progress_overview" in includes:
        files_to_include.append(os.path.join(output_dir, "progress_overview.png"))

    if "cumulative_images" in includes:
        cumulative_dir = os.path.join(output_dir, "steps", "cumulative")
        if os.path.exists(cumulative_dir):
            for f in os.listdir(cumulative_dir):
                files_to_include.append(os.path.join(cumulative_dir, f))

    if "context_images" in includes:
        context_dir = os.path.join(output_dir, "steps", "context")
        if os.path.exists(context_dir):
            for f in os.listdir(context_dir):
                files_to_include.append(os.path.join(context_dir, f))

    if "isolated_images" in includes:
        isolated_dir = os.path.join(output_dir, "steps", "isolated")
        if os.path.exists(isolated_dir):
            for f in os.listdir(isolated_dir):
                files_to_include.append(os.path.join(isolated_dir, f))

    if "painting_guide" in includes:
        files_to_include.append(os.path.join(output_dir, "painting_guide.json"))

    if "paint_kit" in includes:
        # TODO: Generate paint kit if not exists
        paint_kit_path = os.path.join(output_dir, "paint_kit.json")
        if os.path.exists(paint_kit_path):
            files_to_include.append(paint_kit_path)

    if "mixing_guide" in includes:
        # TODO: Generate mixing guide if not exists
        mixing_guide_path = os.path.join(output_dir, "mixing_guide.json")
        if os.path.exists(mixing_guide_path):
            files_to_include.append(mixing_guide_path)

    # Filter to existing files
    files_to_include = [f for f in files_to_include if os.path.exists(f)]

    if not files_to_include:
        raise HTTPException(status_code=404, detail="No files found for this product")

    # If only one file, return directly
    if len(files_to_include) == 1:
        return FileResponse(
            files_to_include[0],
            filename=os.path.basename(files_to_include[0])
        )

    # Multiple files - create zip
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in files_to_include:
            # Preserve directory structure
            arcname = os.path.relpath(file_path, output_dir)
            zip_file.write(file_path, arcname)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f"attachment; filename=artisan_{job_id}_{product_id}.zip"
        }
    )


# Helper function for payment route
def mark_purchased(job_id: str, product_ids: List[str]):
    """Mark products as purchased for a job."""
    if job_id not in purchases_db:
        purchases_db[job_id] = []

    for pid in product_ids:
        if pid not in purchases_db[job_id]:
            purchases_db[job_id].append(pid)
