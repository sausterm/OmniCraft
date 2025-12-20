# OmniCraft Deployment Guide

This guide covers deploying the OmniCraft web application with a Vercel frontend and self-hosted backend.

## Architecture Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│  Vercel CDN     │      │  ngrok Tunnel    │      │  Local Server   │
│  (Next.js)      │──────│  (HTTPS proxy)   │──────│  (FastAPI)      │
│  Frontend       │      │  Public HTTPS    │      │  localhost:8000 │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

## Prerequisites

- Node.js 18+ (for frontend)
- Python 3.10+ (for backend)
- ngrok account (free tier works)
- Vercel account (free tier works)
- GPU recommended for YOLO inference

## Backend Setup

### 1. Install Dependencies

```bash
cd artisan
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
```

### 2. Configure ngrok

```bash
# Install ngrok (macOS)
brew install ngrok

# Add your authtoken
ngrok config add-authtoken YOUR_AUTH_TOKEN
```

### 3. Start the Backend

```bash
cd artisan

# Set environment variables and start
PYTHONPATH=/path/to/OmniCraft \
CORS_ORIGINS='["*"]' \
./venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 4. Start ngrok Tunnel

In a separate terminal:

```bash
ngrok http 8000
```

Note the HTTPS URL (e.g., `https://xxxx-xxxx-xxxx.ngrok-free.dev`).

## Frontend Setup

### 1. Install Dependencies

```bash
cd web
npm install
```

### 2. Configure Environment

Create `.env.local`:

```env
NEXT_PUBLIC_API_URL=https://your-ngrok-url.ngrok-free.dev
```

### 3. Deploy to Vercel

```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel

# Set production environment variable
vercel env add NEXT_PUBLIC_API_URL production
# Enter your ngrok URL when prompted
```

## API Routes Configuration

The frontend proxies certain routes to the backend. See `web/next.config.js`:

| Route | Destination |
|-------|-------------|
| `/api/upload` | Backend |
| `/api/process/*` | Backend |
| `/api/status/*` | Backend |
| `/api/results/*` | Backend |
| `/api/products/*` | Backend |
| `/api/preview/*` | Backend |
| `/api/checkout` | Backend |
| `/api/image/*` | **Next.js proxy** (adds ngrok header) |

## Image Proxy

Images are proxied through Next.js to add the `ngrok-skip-browser-warning` header that `<img>` tags cannot add directly.

See: `web/app/api/image/[jobId]/[type]/[index]/route.ts`

## Monitoring

### Backend Logs

The backend outputs structured logs:

```
2024-12-19 20:30:15 [INFO] api.routes.process: [abc123] Starting processing job
2024-12-19 20:30:15 [INFO] api.routes.process: [abc123] Config: model_size=m, conf=0.3
2024-12-19 20:30:16 [INFO] yolo_bob_ross_paint: Creating step images...
2024-12-19 20:30:16 [INFO] yolo_bob_ross_paint:   Step 1/15: Sky - Dark Values (blocking)
```

### Health Check

```bash
curl https://your-ngrok-url.ngrok-free.dev/health
# {"status":"healthy","version":"5.0.0","yolo_available":true}
```

## Troubleshooting

### Images not loading

1. Check that ngrok tunnel is running
2. Verify `NEXT_PUBLIC_API_URL` is set correctly in Vercel
3. Check browser console for CORS errors
4. Ensure `/api/image/*` is NOT in next.config.js rewrites

### CORS errors

Set `CORS_ORIGINS='["*"]'` when starting the backend, or specify your Vercel domain.

### Backend import errors

Ensure `PYTHONPATH` includes the OmniCraft directory:

```bash
PYTHONPATH=/path/to/OmniCraft ./venv/bin/python -m uvicorn api.main:app ...
```

## Production Considerations

For production deployment, consider:

1. **Persistent ngrok URL**: Use ngrok paid tier for static subdomain
2. **Process manager**: Use systemd, pm2, or supervisor to keep backend running
3. **SSL**: ngrok provides HTTPS, but for custom domain use Cloudflare Tunnel
4. **Database**: Current implementation uses in-memory storage; add Redis/PostgreSQL for persistence
5. **File storage**: Add S3/GCS for storing generated images
