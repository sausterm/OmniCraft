# Contributing to Synesthetic Audio Visualization System

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd synesthetic-viz
   ```

2. **Run the setup script**
   ```bash
   ./scripts/setup.sh
   ```

3. **Start development servers**
   ```bash
   ./scripts/run-dev.sh
   ```

## Project Structure

```
synesthetic-viz/
├── frontend/          # Three.js + Web Audio visualization
├── backend/           # Python API + audio analysis
├── experiments/       # POCs and experiments
├── tests/            # Test suites
└── docs/             # Documentation
```

## Development Workflow

### Frontend Development

1. All visualization code goes in `frontend/src/visualizers/`
2. Audio processing code goes in `frontend/src/audio/`
3. UI components go in `frontend/src/components/`
4. Follow ESLint rules (run `npm run lint`)

### Backend Development

1. API endpoints go in `backend/api/`
2. Audio analysis code goes in `backend/audio_analysis/`
3. AI generation code goes in `backend/ai_generation/`
4. Format code with Black (run `black .`)

### Testing

```bash
# Frontend tests
cd frontend
npm test

# Backend tests
cd backend
source venv/bin/activate
pytest
```

## Coding Standards

### JavaScript/TypeScript
- Use ES6+ features
- Prefer const over let
- Use meaningful variable names
- Add JSDoc comments for functions
- Keep functions small and focused

### Python
- Follow PEP 8 style guide
- Use type hints
- Add docstrings to classes and functions
- Keep functions small and focused
- Use meaningful variable names

### Git Workflow

1. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit
   ```bash
   git add .
   git commit -m "feat: add new visualization mode"
   ```

3. Push and create a pull request
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

Follow conventional commits:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation changes
- `style:` - Code style changes (formatting, etc.)
- `refactor:` - Code refactoring
- `test:` - Adding tests
- `chore:` - Maintenance tasks

Examples:
```
feat: add frequency wave visualizer
fix: resolve audio latency issue
docs: update API documentation
```

## Current Development Phase

**Phase 1: Research & Architecture (Weeks 1-3)**

Priority tasks:
1. Audio analysis library evaluation
2. Basic visualization POCs
3. AI model speed testing
4. Architecture documentation

See `docs/MEETING_LOG.md` for current action items.

## Questions or Issues?

- Check existing issues on GitHub
- Review documentation in `docs/`
- Ask questions in discussions

## License

[To be determined]
