# Artisan Deployment Guide

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────────┐
│  Vercel         │     │  Backend (choose 1+) │
│  (Frontend)     │────▶│  • Local + ngrok     │
│                 │     │  • Modal (cloud)     │
│  Next.js App    │     │  • Railway (cloud)   │
└─────────────────┘     └──────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────┐
                        │  Replicate API   │
                        │  (Style Transfer)│
                        └──────────────────┘
```

## Quick Start

### Option 1: Modal (Recommended for GPU)

Modal is serverless - you only pay when processing. $30/month free credits.

```bash
# Install Modal CLI
pip install modal

# Authenticate
modal token new

# Create secrets in Modal dashboard (modal.com):
# - REPLICATE_API_TOKEN
# - STRIPE_API_KEY (optional)

# Deploy
cd /path/to/OmniCraft
modal deploy artisan/deploy/modal_app.py

# Your URL will be: https://your-username--artisan-api-fastapi-app.modal.run
```

### Option 2: Railway (Simple CPU)

Railway is simpler and works well for YOLO on CPU (~$20/mo).

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login
railway login

# Initialize project
cd /path/to/OmniCraft
railway init

# Set environment variables
railway variables set REPLICATE_API_TOKEN=your_token
railway variables set DEBUG=false
railway variables set ENVIRONMENT=production

# Deploy
railway up
```

### Option 3: Local + ngrok (Development)

Keep using your local machine with ngrok tunnel.

```bash
# Start the API
cd /path/to/OmniCraft
./artisan/venv/bin/python -m uvicorn artisan.api.main:app --reload

# In another terminal, start ngrok
ngrok http 8000
```

## Vercel Environment Variables

Set these in Vercel dashboard (vercel.com → Project → Settings → Environment Variables):

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Primary backend (ngrok) | `https://abc123.ngrok-free.app` |
| `NEXT_PUBLIC_FALLBACK_API_URL` | Cloud fallback | `https://you--artisan-api.modal.run` |
| `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | Stripe public key | `pk_live_...` |
| `AUTH_SECRET` | NextAuth secret | Random 32+ char string |

## Dual Backend Setup

The frontend automatically switches between backends:

1. **Primary (ngrok)**: Used when your local machine is running
2. **Fallback (Modal/Railway)**: Used when primary is unavailable

Health checks run every 30 seconds. If primary fails, traffic routes to fallback.

## Cost Comparison

| Option | Monthly Cost | GPU | Notes |
|--------|--------------|-----|-------|
| Local + ngrok | Free | Your machine | Requires machine running |
| Modal (CPU) | ~$5-15 | No | Pay per second |
| Modal (GPU) | ~$15-50 | T4 | Pay per second |
| Railway | ~$20 | No | Always-on |
| Render | ~$25 | No | Always-on |

## Style Transfer

Style transfer uses Replicate API (cloud GPU), so no local GPU needed:

```bash
# Get API token from replicate.com
export REPLICATE_API_TOKEN=r8_xxxxx
```

Replicate charges ~$0.01-0.05 per style transfer.
