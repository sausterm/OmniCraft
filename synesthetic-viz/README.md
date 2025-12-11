# Synesthetic Audio Visualization System

A groundbreaking audio-reactive visualization platform combining real-time processing with AI-enhanced generative art.

## 🎨 Project Overview

This system creates immersive synesthetic experiences by:
- Processing live audio streams in real-time (<50ms latency)
- Generating dynamic visualizations synchronized to music
- Using AI to create sophisticated visual narratives from audio analysis
- Bridging auditory and visual perception

## 📁 Repository Structure

```
synesthetic-viz/
├── frontend/              # Web application & visualizations
│   ├── src/
│   │   ├── components/    # React/UI components
│   │   ├── visualizers/   # Three.js visualization modules
│   │   ├── audio/         # Audio processing & analysis
│   │   └── utils/         # Helper functions
│   └── public/            # Static assets
│
├── backend/               # Python API & AI services
│   ├── audio_analysis/    # Audio feature extraction
│   ├── ai_generation/     # AI/ML models & generation
│   ├── api/              # FastAPI endpoints
│   └── utils/            # Backend utilities
│
├── experiments/           # POCs and prototypes
│   └── poc/              # Proof of concepts
│
├── tests/                # Test suites
│   ├── frontend/         # Frontend tests
│   └── backend/          # Backend tests
│
├── scripts/              # Build & utility scripts
└── docs/                 # Project documentation
```

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- Python 3.10+
- Modern GPU (NVIDIA recommended for AI features)
- Git

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd synesthetic-viz

# Install frontend dependencies
cd frontend
npm install
cd ..

# Install backend dependencies
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
cd ..
```

### Development

**Start Frontend (Terminal 1):**
```bash
cd frontend
npm run dev
# Visit http://localhost:3000
```

**Start Backend (Terminal 2):**
```bash
cd backend
source venv/bin/activate
python -m uvicorn api.main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

## 🎯 Current Phase: Research & Architecture (Week 1-3)

### Active Tasks
- [ ] Audio analysis library evaluation
- [ ] Basic FFT visualization POC
- [ ] Stable Diffusion speed benchmarks
- [ ] System architecture finalization

See `docs/MEETING_LOG.md` for detailed action items.

## 📚 Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) - Vision, goals, and roadmap
- [Technical Architecture](docs/TECHNICAL_ARCHITECTURE.md) - System design
- [Development Setup](docs/DEV_ENVIRONMENT_SETUP.md) - Detailed setup guide
- [Meeting Log](docs/MEETING_LOG.md) - Decisions and action items

## 🛠️ Technology Stack

### Frontend
- **Rendering:** Three.js + WebGL
- **Audio:** Web Audio API
- **Framework:** Vite (+ React optional)
- **Language:** JavaScript/TypeScript

### Backend
- **Audio Analysis:** LibROSA, Essentia
- **AI/ML:** Stable Diffusion, ControlNet, AnimateDiff
- **Framework:** FastAPI
- **Language:** Python 3.10+

## 🎵 Features

### Real-Time Mode
- Live audio stream processing
- FFT and frequency analysis
- Beat detection and tempo tracking
- Dynamic parameter mapping
- 60fps rendering at <50ms latency

### AI-Enhanced Mode
- Deep audio structure analysis
- Lyric sentiment analysis
- AI-generated visual narratives
- Temporal coherence across frames
- Style transfer and composition

## 📊 Project Status

**Phase:** 1 - Research & Architecture  
**Started:** October 24, 2025  
**Target Completion:** November 14, 2025

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

[To be determined]

## 🙏 Acknowledgments

- Three.js community
- Stable Diffusion / Stability AI
- LibROSA developers
- Web Audio API specification

---

**Last Updated:** October 24, 2025  
**Version:** 0.1.0-alpha
