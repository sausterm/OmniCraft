"""
Modal deployment for Artisan Paint-by-Numbers API

Deploy with: modal deploy artisan/deploy/modal_app.py
Run locally: modal serve artisan/deploy/modal_app.py

Setup:
1. pip install modal
2. modal token new
3. Set secrets in Modal dashboard (REPLICATE_API_TOKEN, STRIPE_API_KEY, etc.)
4. modal deploy artisan/deploy/modal_app.py
"""

import modal

# Create Modal app
app = modal.App("artisan-api")

# Define the container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1")
    .pip_install(
        # Core
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        # YOLO
        "ultralytics>=8.0.0",
        "torch",
        "torchvision",
        # API
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.22.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        # Style transfer
        "replicate>=0.20.0",
        "requests>=2.28.0",
        # Payment
        "stripe>=5.0.0",
    )
)

# Mount the artisan source code
artisan_mount = modal.Mount.from_local_dir(
    local_path="artisan",
    remote_path="/root/artisan",
    condition=lambda path: not any(
        x in path for x in ["__pycache__", ".pyc", "outputs/", "uploads/", ".git"]
    ),
)


@app.function(
    image=image,
    mounts=[artisan_mount],
    secrets=[modal.Secret.from_name("artisan-secrets")],  # Create in Modal dashboard
    timeout=600,  # 10 minute timeout for processing
    memory=4096,  # 4GB RAM
    cpu=2,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI application."""
    import sys
    sys.path.insert(0, "/root")

    # Import and return the FastAPI app
    from artisan.api.main import app
    return app


# Optional: GPU variant for faster YOLO processing
@app.function(
    image=image,
    mounts=[artisan_mount],
    secrets=[modal.Secret.from_name("artisan-secrets")],
    gpu="T4",  # Cheapest GPU option (~$0.58/hr)
    timeout=600,
    memory=8192,
)
@modal.asgi_app(label="artisan-api-gpu")
def fastapi_app_gpu():
    """GPU-accelerated variant for faster processing."""
    import sys
    sys.path.insert(0, "/root")

    from artisan.api.main import app
    return app


# Health check endpoint for monitoring
@app.function(image=modal.Image.debian_slim())
@modal.web_endpoint(method="GET")
def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "artisan-api"}
