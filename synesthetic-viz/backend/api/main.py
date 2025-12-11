"""
Synesthetic Audio Visualization System - API
FastAPI backend for audio analysis and AI generation
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Synesthetic Visualization API",
    description="Backend API for audio analysis and AI-enhanced visualization",
    version="0.1.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    version: str
    message: str

class AudioAnalysisResponse(BaseModel):
    filename: str
    duration: Optional[float] = None
    tempo: Optional[float] = None
    key: Optional[str] = None
    features: Dict[str, Any]
    status: str

# Routes
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="running",
        version="0.1.0",
        message="Synesthetic Visualization API - Phase 1"
    )

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "phase": "Research & Architecture",
        "features": {
            "audio_analysis": "available",
            "ai_generation": "not implemented yet"
        }
    }

@app.post("/api/analyze-audio", response_model=AudioAnalysisResponse)
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file and extract features
    
    This endpoint will use LibROSA for comprehensive audio analysis
    including tempo, key, spectral features, etc.
    """
    try:
        logger.info(f"Received audio file: {file.filename}")
        
        # TODO: Implement actual audio analysis using LibROSA
        # For now, return a placeholder response
        
        return AudioAnalysisResponse(
            filename=file.filename,
            duration=None,
            tempo=None,
            key=None,
            features={
                "message": "Audio analysis implementation pending",
                "phase": "Phase 2 - Real-Time Engine Development"
            },
            status="placeholder"
        )
        
    except Exception as e:
        logger.error(f"Error analyzing audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/visualization-presets")
async def get_visualization_presets():
    """Get available visualization presets"""
    return {
        "presets": [
            {
                "id": "particles",
                "name": "Particle System",
                "description": "Audio-reactive particle cloud",
                "status": "implemented"
            },
            {
                "id": "waves",
                "name": "Frequency Waves",
                "description": "Waveform visualization",
                "status": "planned"
            },
            {
                "id": "geometry",
                "name": "Geometric Shapes",
                "description": "Morphing geometric patterns",
                "status": "planned"
            }
        ]
    }

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
