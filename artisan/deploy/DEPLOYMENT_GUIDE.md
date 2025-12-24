# Artisan Deployment Guide

Deploy the Artisan paint-by-numbers app to Vercel (frontend) + Modal (backend).

## Prerequisites

1. **Accounts needed:**
   - [Modal](https://modal.com) - Backend hosting (serverless)
   - [Vercel](https://vercel.com) - Frontend hosting
   - [Replicate](https://replicate.com) - AI style transfer
   - [Stripe](https://stripe.com) - Payments (optional)
   - [Resend](https://resend.com) - Email magic links (optional)

2. **Install Modal CLI:**
   ```bash
   pip install modal
   modal token new
   ```

---

## Step 1: Deploy Backend to Modal

### 1.1 Create Modal Secrets

Go to [modal.com/secrets](https://modal.com/secrets) and create a new secret group named `artisan-secrets`:

| Key | Where to get it | Required |
|-----|-----------------|----------|
| `REPLICATE_API_TOKEN` | [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens) | **Yes** (for style transfer) |
| `STRIPE_API_KEY` | [dashboard.stripe.com/apikeys](https://dashboard.stripe.com/apikeys) (sk_test_... or sk_live_...) | Optional |
| `STRIPE_WEBHOOK_SECRET` | Stripe dashboard > Webhooks | Optional |

### 1.2 Deploy to Modal

From the OmniCraft root directory:

```bash
cd /path/to/OmniCraft
modal deploy artisan/deploy/modal_app.py
```

You'll see output like:
```
✓ Created objects.
├── 🔗 https://sloan--artisan-api-fastapi-app.modal.run
└── 🔗 https://sloan--artisan-api-health.modal.run
```

**Save the first URL** - this is your `NEXT_PUBLIC_API_URL`.

### 1.3 Test Backend

```bash
# Health check
curl https://YOUR-USERNAME--artisan-api-fastapi-app.modal.run/health

# Should return: {"status":"ok","version":"5.0.0",...}
```

---

## Step 2: Deploy Frontend to Vercel

### 2.1 Connect Repository

1. Go to [vercel.com/new](https://vercel.com/new)
2. Import your OmniCraft repository
3. Set **Root Directory** to `web`
4. Framework: Next.js (auto-detected)

### 2.2 Configure Environment Variables

In Vercel Project Settings > Environment Variables, add:

| Variable | Value | Environment |
|----------|-------|-------------|
| `NEXT_PUBLIC_API_URL` | `https://YOUR-USERNAME--artisan-api-fastapi-app.modal.run` | Production |
| `AUTH_SECRET` | Generate with `openssl rand -base64 32` | Production |
| `AUTH_RESEND_KEY` | Your Resend API key (for email auth) | Production |
| `NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY` | `pk_live_...` or `pk_test_...` | Production |

### 2.3 Deploy

Click Deploy. Vercel will build and deploy your frontend.

---

## Step 3: Configure CORS (If Needed)

If you get CORS errors, update `artisan/api/config.py`:

```python
cors_origins: List[str] = Field(
    default=["https://your-app.vercel.app", "https://your-custom-domain.com"],
    description="Allowed CORS origins"
)
```

Then redeploy Modal:
```bash
modal deploy artisan/deploy/modal_app.py
```

---

## Testing the Full Flow

1. **Upload an image:**
   - Go to your Vercel URL
   - Upload an image
   - Should get a job ID

2. **Test style transfer:**
   - Select a style (Van Gogh, Pop Art, etc.)
   - Wait 30-60 seconds for Replicate to process
   - Styled image should appear

3. **Test processing:**
   - Click "Generate Guide"
   - Wait for YOLO segmentation (10-30 seconds)
   - Should see step-by-step painting instructions

---

## Troubleshooting

### "Style transfer failed"
- Check that `REPLICATE_API_TOKEN` is set in Modal secrets
- Verify token at [replicate.com/account/api-tokens](https://replicate.com/account/api-tokens)

### CORS errors
- Add your Vercel domain to `cors_origins` in config.py
- Redeploy Modal

### 502/504 Gateway Timeout
- Modal functions have a 10-minute timeout
- Large images may take longer - try resizing

### "Job not found"
- Modal uses in-memory storage; jobs are lost on cold starts
- This is expected for the current setup

---

## Cost Estimates

| Service | Cost |
|---------|------|
| Modal | ~$0.10-0.50/hour of compute (pay per use) |
| Replicate | ~$0.005-0.01 per style transfer |
| Vercel | Free tier is sufficient |
| Stripe | 2.9% + $0.30 per transaction |

For low volume (~100 users/day), expect ~$10-30/month total.

---

## Quick Reference

```bash
# Deploy backend
modal deploy artisan/deploy/modal_app.py

# Test backend locally
modal serve artisan/deploy/modal_app.py

# View Modal logs
modal app logs artisan-api

# Redeploy frontend (from Vercel dashboard or CLI)
vercel --prod
```
