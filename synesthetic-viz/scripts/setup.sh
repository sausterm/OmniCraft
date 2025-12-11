#!/bin/bash

# Synesthetic Audio Visualization System - Setup Script
# Run this script to set up the entire development environment

set -e  # Exit on error

echo "=========================================="
echo "Synesthetic Visualization - Setup"
echo "=========================================="
echo ""

# Check prerequisites
echo "Checking prerequisites..."

# Check Node.js
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js 18+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi
echo "✓ Node.js $(node --version)"

# Check npm
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed."
    exit 1
fi
echo "✓ npm $(npm --version)"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.10+ first."
    exit 1
fi
echo "✓ Python $(python3 --version)"

echo ""
echo "=========================================="
echo "Setting up Frontend..."
echo "=========================================="

cd frontend

# Install frontend dependencies
echo "Installing Node.js dependencies..."
npm install

echo "✓ Frontend setup complete!"
echo ""

cd ..

echo "=========================================="
echo "Setting up Backend..."
echo "=========================================="

cd backend

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install Python dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✓ Backend setup complete!"
echo ""

cd ..

echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "🚀 To start development:"
echo ""
echo "Terminal 1 (Frontend):"
echo "  cd frontend"
echo "  npm run dev"
echo "  -> http://localhost:3000"
echo ""
echo "Terminal 2 (Backend):"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  python -m uvicorn api.main:app --reload"
echo "  -> http://localhost:8000/docs"
echo ""
echo "Or use the run script:"
echo "  ./scripts/run-dev.sh"
echo ""
