# Development Environment Setup Guide
## Synesthetic Audio Visualization System

**Created:** October 24, 2025  
**Target:** Week 1 Setup

---

## Quick Start: Local Development Setup

### Prerequisites
- [ ] Git installed
- [ ] Node.js 18+ installed
- [ ] Python 3.10+ installed
- [ ] Modern GPU with updated drivers
- [ ] 20GB+ free disk space

---

## Step 1: Initialize Project Structure

```bash
# Create main project directory
mkdir synesthetic-viz
cd synesthetic-viz

# Initialize git
git init
echo "node_modules/" > .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo ".env" >> .gitignore
echo "*.pyc" >> .gitignore

# Create project structure
mkdir -p frontend/src frontend/public
mkdir -p backend/audio_analysis backend/ai_generation
mkdir -p experiments
mkdir -p docs
```

---

## Step 2: Frontend Setup (Three.js + Web Audio)

```bash
cd frontend

# Initialize Node project
npm init -y

# Install core dependencies
npm install three
npm install vite --save-dev  # Fast dev server

# Optional: React setup (if you want UI framework)
# npm install react react-dom
# npm install @vitejs/plugin-react --save-dev

# Create basic structure
cat > vite.config.js << 'EOF'
import { defineConfig } from 'vite'

export default defineConfig({
  server: {
    port: 3000,
  },
})
EOF

# Create index.html
cat > index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Synesthetic Visualization</title>
  <style>
    body { margin: 0; overflow: hidden; }
    canvas { display: block; }
  </style>
</head>
<body>
  <script type="module" src="/src/main.js"></script>
</body>
</html>
EOF

# Create main.js starter
mkdir -p src
cat > src/main.js << 'EOF'
import * as THREE from 'three';

// Basic Three.js setup
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });

renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Add a simple cube
const geometry = new THREE.BoxGeometry();
const material = new THREE.MeshBasicMaterial({ color: 0x00ff00 });
const cube = new THREE.Mesh(geometry, material);
scene.add(cube);

camera.position.z = 5;

// Animation loop
function animate() {
  requestAnimationFrame(animate);
  cube.rotation.x += 0.01;
  cube.rotation.y += 0.01;
  renderer.render(scene, camera);
}

animate();

console.log('Synesthetic Visualization - Frontend Running!');
EOF

# Update package.json scripts
npm pkg set scripts.dev="vite"
npm pkg set scripts.build="vite build"
npm pkg set scripts.preview="vite preview"

# Test it
echo "Frontend setup complete! Run 'npm run dev' to test"
cd ..
```

---

## Step 3: Backend Setup (Python + Audio Analysis)

```bash
cd backend

# Create Python virtual environment
python3 -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Create requirements.txt
cat > requirements.txt << 'EOF'
# Audio Processing
librosa>=0.10.0
numpy>=1.24.0
scipy>=1.11.0

# Web Framework
fastapi>=0.104.0
uvicorn>=0.24.0
python-multipart>=0.0.6

# AI/ML (optional for Phase 1)
# torch>=2.1.0
# diffusers>=0.24.0
# transformers>=4.35.0

# Utilities
python-dotenv>=1.0.0
pydantic>=2.4.0
EOF

# Install dependencies
pip install -r requirements.txt

# Create basic API structure
cat > main.py << 'EOF'
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np

app = FastAPI(title="Synesthetic Viz API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Synesthetic Visualization API", "status": "running"}

@app.post("/analyze-audio")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze uploaded audio file and return features
    """
    # Save uploaded file temporarily
    contents = await file.read()
    
    # TODO: Implement actual audio analysis
    return {
        "filename": file.filename,
        "status": "analyzed",
        "message": "Audio analysis placeholder - implement LibROSA processing"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF

echo "Backend setup complete! Run 'python main.py' to test"
cd ..
```

---

## Step 4: First Proof of Concept - Audio Visualizer

Create a basic audio-reactive visualization:

```bash
cd frontend/src

# Create audio visualizer module
cat > audioVisualizer.js << 'EOF'
export class AudioVisualizer {
  constructor() {
    this.audioContext = null;
    this.analyser = null;
    this.dataArray = null;
    this.bufferLength = null;
  }

  async init() {
    // Request microphone access
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    
    // Create audio context
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    this.analyser = this.audioContext.createAnalyser();
    this.analyser.fftSize = 2048;
    
    this.bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(this.bufferLength);
    
    // Connect microphone to analyser
    const source = this.audioContext.createMediaStreamSource(stream);
    source.connect(this.analyser);
    
    console.log('Audio visualizer initialized');
  }

  getFrequencyData() {
    if (!this.analyser) return null;
    this.analyser.getByteFrequencyData(this.dataArray);
    return this.dataArray;
  }

  getTimeDomainData() {
    if (!this.analyser) return null;
    this.analyser.getByteTimeDomainData(this.dataArray);
    return this.dataArray;
  }

  getAverageFrequency() {
    const data = this.getFrequencyData();
    if (!data) return 0;
    return data.reduce((a, b) => a + b) / data.length;
  }
}
EOF

cd ../..
```

---

## Step 5: VS Code Configuration

Create workspace settings for optimal development:

```bash
# Create .vscode directory
mkdir -p .vscode

cat > .vscode/settings.json << 'EOF'
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "python.formatting.provider": "black",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "[javascript]": {
    "editor.defaultFormatter": "esbenp.prettier-vscode"
  },
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter"
  },
  "files.exclude": {
    "**/__pycache__": true,
    "**/.pytest_cache": true,
    "**/node_modules": true
  }
}
EOF

cat > .vscode/extensions.json << 'EOF'
{
  "recommendations": [
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "ms-python.python",
    "ms-python.black-formatter",
    "slevesque.shader",
    "eamodio.gitlens"
  ]
}
EOF
```

---

## Step 6: Docker Setup (Optional - For Consistent Environments)

```bash
# Frontend Dockerfile
cat > frontend/Dockerfile << 'EOF'
FROM node:18-alpine
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
EXPOSE 3000
CMD ["npm", "run", "dev", "--", "--host"]
EOF

# Backend Dockerfile
cat > backend/Dockerfile << 'EOF'
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "main.py"]
EOF

# Docker Compose
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    environment:
      - PYTHONUNBUFFERED=1
EOF

echo "Docker setup complete! Run 'docker-compose up' to start both services"
```

---

## Testing Your Setup

### Test Frontend:
```bash
cd frontend
npm run dev
# Visit http://localhost:3000
# You should see a rotating green cube
```

### Test Backend:
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python main.py
# Visit http://localhost:8000/docs for API documentation
```

---

## GPU Setup for AI (Week 2-3)

### NVIDIA GPU Setup:
```bash
# Check GPU
nvidia-smi

# Install CUDA Toolkit (if not already installed)
# Visit: https://developer.nvidia.com/cuda-downloads

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Diffusers for Stable Diffusion
pip install diffusers transformers accelerate safetensors

# Test GPU access
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"}')"
```

---

## Useful Development Commands

### Frontend:
```bash
npm run dev          # Start dev server
npm run build        # Build for production
npm run preview      # Preview production build
```

### Backend:
```bash
python main.py                    # Start API server
uvicorn main:app --reload         # Auto-reload on changes
pytest                            # Run tests (when you add them)
```

### Both (with Docker):
```bash
docker-compose up                 # Start all services
docker-compose up --build         # Rebuild and start
docker-compose down               # Stop all services
```

---

## Next Steps After Setup

1. **Create your first POC** (Action Item AI-002):
   - Basic FFT visualization with Three.js
   - Audio input from microphone
   - Real-time frequency data → particle system

2. **Test audio analysis libraries** (Action Item AI-001):
   - Implement LibROSA feature extraction
   - Compare with Web Audio API performance
   - Document findings in RESEARCH_NOTES.docx

3. **Benchmark AI generation** (Action Item AI-003):
   - Set up basic Stable Diffusion inference
   - Measure generation times
   - Test different model variants

---

## Troubleshooting

### Common Issues:

**Port already in use:**
```bash
# Find and kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or change port in vite.config.js
```

**Python package conflicts:**
```bash
# Create fresh virtual environment
deactivate
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**GPU not detected:**
```bash
# Verify CUDA installation
nvcc --version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Resources

- **Three.js Examples:** https://threejs.org/examples/
- **Web Audio API Docs:** https://developer.mozilla.org/en-US/Web_Audio_API
- **LibROSA Tutorial:** https://librosa.org/doc/latest/tutorial.html
- **FastAPI Docs:** https://fastapi.tiangolo.com/
- **Stable Diffusion:** https://github.com/Stability-AI/stablediffusion

---

**Document Owner:** Technical Lead  
**Last Updated:** October 24, 2025
