"""
Artisan API - Main FastAPI Application

Production-ready API for paint-by-numbers generation with:
- Image upload and async processing
- Job status tracking
- Stripe payment integration
- S3-based file delivery
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import settings
from .routes import upload, process, download, payment, health, promo


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    print(f"Starting {settings.app_name} v{settings.app_version}")
    print(f"Environment: {settings.environment}")
    print(f"Debug mode: {settings.debug}")

    # Initialize database connection pool
    # await init_db()

    yield

    # Shutdown
    print("Shutting down...")
    # await close_db()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="""
## Artisan Paint-by-Numbers API

AI-powered paint-by-numbers generation with:
- **YOLO semantic segmentation** - Detects objects (dogs, cats, people, etc.)
- **Scene context analysis** - Time of day, weather, lighting, mood
- **Bob Ross methodology** - Back-to-front painting with proper techniques
- **Three view types** - Cumulative, context, and isolated views per step

### Quick Start

1. **Upload** an image → Get a `job_id`
2. **Process** with your settings → Track progress
3. **Preview** free low-res results
4. **Purchase** the package you want
5. **Download** your files

### Products

| Package | Price | Includes |
|---------|-------|----------|
| Preview | Free | Low-res preview |
| Basic | $4.99 | Cumulative images |
| Standard | $9.99 | All images + guide |
| Premium | $19.99 | Everything + paint kit |
        """,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health.router, tags=["Health"])
    app.include_router(upload.router, prefix="/api", tags=["Upload"])
    app.include_router(process.router, prefix="/api", tags=["Process"])
    app.include_router(download.router, prefix="/api", tags=["Download"])
    app.include_router(payment.router, prefix="/api", tags=["Payment"])
    app.include_router(promo.router, prefix="/api", tags=["Promo"])

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else "An error occurred"
            }
        )

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "artisan.api.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
