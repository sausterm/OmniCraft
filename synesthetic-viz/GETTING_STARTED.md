# Getting Started with Synesthetic Audio Visualization

Welcome! This guide will help you get up and running with the project.

## 🎯 What You're Building

A real-time audio-reactive visualization system that:
- Processes live audio with <50ms latency
- Creates dynamic 3D particle visualizations
- Will eventually use AI to generate sophisticated visual narratives

## 📋 Prerequisites

Before you begin, make sure you have:

- **Node.js 18+** - [Download](https://nodejs.org/)
- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Git** - [Download](https://git-scm.com/)
- **Modern browser** - Chrome or Firefox (for WebGL support)
- **Microphone** - For real-time audio input

Optional but recommended:
- **NVIDIA GPU** - For AI features (Phase 3)
- **VS Code** - [Download](https://code.visualstudio.com/)

## 🚀 Quick Start (5 minutes)

### 1. Get the Code

```bash
# If you haven't cloned yet
git clone <your-repo-url>
cd synesthetic-viz
```

### 2. Run Setup Script

```bash
./scripts/setup.sh
```

This will:
- Install frontend dependencies (npm packages)
- Create Python virtual environment
- Install backend dependencies

### 3. Start Development Servers

**Option A: Use the run script (easiest)**
```bash
./scripts/run-dev.sh
```

**Option B: Start manually in two terminals**

Terminal 1 - Frontend:
```bash
cd frontend
npm run dev
```

Terminal 2 - Backend:
```bash
cd backend
source venv/bin/activate  # On Windows: venv\Scripts\activate
python -m uvicorn api.main:app --reload
```

### 4. Open Your Browser

Visit http://localhost:3000

Click "Start Visualization" and allow microphone access when prompted.

**🎉 You should now see a 3D particle system reacting to your microphone input!**

## 📁 Project Structure Overview

```
synesthetic-viz/
├── frontend/              # Web application
│   ├── src/
│   │   ├── audio/         # AudioAnalyzer.js - captures & analyzes audio
│   │   ├── visualizers/   # ParticleVisualizer.js - renders visuals
│   │   └── main.js        # Application entry point
│   └── index.html         # Main HTML file
│
├── backend/               # Python API
│   ├── api/              # FastAPI endpoints
│   ├── audio_analysis/   # Audio feature extraction
│   └── ai_generation/    # AI models (Phase 3)
│
├── docs/                 # Project documentation
│   ├── PROJECT_OVERVIEW.md
│   ├── TECHNICAL_ARCHITECTURE.md
│   └── MEETING_LOG.md
│
└── scripts/              # Setup and run scripts
```

## 🎨 What's Working Now (Phase 1 POC)

✅ **Real-time audio capture** from microphone  
✅ **FFT analysis** - frequency spectrum analysis  
✅ **3D particle system** - 5000 particles  
✅ **Audio-reactive behavior**:
  - Bass frequencies → Red color & pulsation
  - Mid frequencies → Green color
  - Treble frequencies → Blue color
  - Volume → Particle size & rotation speed

## 🔍 Exploring the Code

### Frontend - Audio Processing

`frontend/src/audio/AudioAnalyzer.js`:
- Captures microphone input via Web Audio API
- Performs FFT (Fast Fourier Transform)
- Extracts bass, mids, treble, and volume

Key methods:
- `initMicrophone()` - Get mic access
- `getFeatures()` - Get real-time audio data

### Frontend - Visualization

`frontend/src/visualizers/ParticleVisualizer.js`:
- Creates 3D particle system with Three.js
- Updates particle positions, colors, and sizes
- Renders at 60 fps

Key methods:
- `createParticles()` - Initialize particle system
- `update(audioFeatures)` - Update based on audio
- `render()` - Draw the scene

### Backend - API

`backend/api/main.py`:
- FastAPI server (currently minimal)
- Health check endpoints
- Placeholder for audio analysis

Access API docs: http://localhost:8000/docs

## 🎯 Your First Tasks

Now that it's running, try these experiments:

### Experiment 1: Modify Particle Count
`frontend/src/visualizers/ParticleVisualizer.js`, line 14:
```javascript
this.particleCount = 5000; // Try 1000 or 10000
```

### Experiment 2: Change Color Mapping
Lines 119-121:
```javascript
colors[i3] = bassIntensity;       // Red from bass
colors[i3 + 1] = midsIntensity;   // Green from mids
colors[i3 + 2] = trebleIntensity; // Blue from treble
```
Try swapping these around!

### Experiment 3: Adjust Rotation Speed
Line 99:
```javascript
this.particles.rotation.y += 0.001 + (volume / 255) * 0.01;
```
Increase the multiplier for faster rotation.

## 📚 Next Steps

### Week 1 Goals (Current Phase)

1. **Experiment with the POC**
   - Try different music/sounds
   - Modify visual parameters
   - Note what works and what doesn't

2. **Document Your Findings**
   - Add notes to `docs/RESEARCH_NOTES.docx`
   - Update `docs/MEETING_LOG.md` with decisions

3. **Evaluate Audio Libraries**
   - Test LibROSA for offline analysis
   - Compare with Web Audio API
   - Document in RESEARCH_NOTES.docx

### Coming in Phase 2 (Weeks 4-10)

- Beat detection
- Tempo tracking
- More visualization modes
- File upload support
- Better UI controls

### Coming in Phase 3 (Weeks 11-18)

- AI-generated visuals
- Lyric analysis
- Stable Diffusion integration
- Temporal coherence

## 🐛 Troubleshooting

### "Microphone permission denied"
- Check browser settings for microphone access
- Try a different browser (Chrome recommended)

### "npm: command not found"
- Install Node.js from nodejs.org
- Restart your terminal after installation

### "python3: command not found"
- Install Python from python.org
- Make sure to check "Add to PATH" during installation

### Backend won't start
```bash
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend shows blank screen
- Check browser console for errors (F12)
- Make sure you're on http://localhost:3000
- Try clearing browser cache

## 📖 Learn More

- [Three.js Documentation](https://threejs.org/docs/)
- [Web Audio API Guide](https://developer.mozilla.org/en-US/docs/Web/API/Web_Audio_API)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [LibROSA Documentation](https://librosa.org/doc/latest/index.html)

## 💬 Getting Help

- Check `docs/` for detailed documentation
- Review `CONTRIBUTING.md` for development guidelines
- Check existing GitHub issues
- Ask questions in project discussions

## 🎉 You're Ready!

You now have:
- ✅ A working real-time audio visualizer
- ✅ Understanding of the project structure
- ✅ Ideas for experiments to try
- ✅ Next steps for development

**Start experimenting and have fun!** 🎵✨

---

**Last Updated:** October 24, 2025  
**Current Phase:** Phase 1 - Research & Architecture
