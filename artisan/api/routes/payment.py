"""
Payment endpoints - Stripe checkout integration.
"""

from typing import List
from fastapi import APIRouter, HTTPException, Request, Header

from ..config import settings, PRODUCTS
from ..models.schemas import (
    JobStatus,
    CheckoutRequest,
    CheckoutResponse,
)
from ..services.email import email_service
from .upload import jobs_db
from .download import mark_purchased


router = APIRouter()


# Check if Stripe is configured
STRIPE_AVAILABLE = bool(settings.stripe_api_key)

if STRIPE_AVAILABLE:
    import stripe
    stripe.api_key = settings.stripe_api_key


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(request: CheckoutRequest):
    """
    Create a Stripe checkout session.

    - **job_id**: The job ID for the images
    - **product_ids**: List of products to purchase

    Returns a URL to redirect the user to Stripe's hosted checkout.
    """
    # Validate job first (before Stripe check - free/promo orders don't need Stripe)
    if request.job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[request.job_id]

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")

    # Store email with job
    import re
    if not re.match(r"[^@]+@[^@]+\.[^@]+", request.email):
        raise HTTPException(status_code=400, detail="Invalid email format")

    jobs_db[request.job_id]["email"] = request.email

    # Validate products
    line_items = []
    for product_id in request.product_ids:
        if product_id not in PRODUCTS:
            raise HTTPException(status_code=400, detail=f"Invalid product: {product_id}")

        product = PRODUCTS[product_id]

        # Skip free products
        if product["price"] == 0:
            continue

        line_items.append({
            "price_data": {
                "currency": "usd",
                "product_data": {
                    "name": product["name"],
                    "description": product["description"],
                },
                "unit_amount": product["price"],
            },
            "quantity": 1,
        })

    if not line_items:
        # All free products - mark as purchased and return
        mark_purchased(request.job_id, request.product_ids)

        # Send confirmation email
        email_service.send_purchase_confirmation(
            to_email=request.email,
            job_id=request.job_id,
            products=request.product_ids,
            is_promo=True,
        )

        return CheckoutResponse(
            checkout_url=request.success_url or settings.stripe_success_url.format(job_id=request.job_id),
            session_id="free",
        )

    # Only check Stripe when we have paid items
    if not STRIPE_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Payment system not configured. Contact support."
        )

    # Build URLs
    success_url = request.success_url or settings.stripe_success_url.format(job_id=request.job_id)
    cancel_url = request.cancel_url or settings.stripe_cancel_url

    try:
        # Create Stripe checkout session
        session = stripe.checkout.Session.create(
            payment_method_types=["card"],
            line_items=line_items,
            mode="payment",
            customer_email=request.email,  # Pre-fill email for Stripe receipt
            success_url=success_url + "?session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
            metadata={
                "job_id": request.job_id,
                "product_ids": ",".join(request.product_ids),
                "email": request.email,
            },
        )

        return CheckoutResponse(
            checkout_url=session.url,
            session_id=session.id,
        )

    except stripe.error.StripeError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Payment error: {str(e)}"
        )


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: str = Header(None, alias="stripe-signature"),
):
    """
    Handle Stripe webhook events.

    This endpoint receives events from Stripe when:
    - Payment succeeds
    - Payment fails
    - Subscription changes (if applicable)
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Payment system not configured")

    if not settings.stripe_webhook_secret:
        raise HTTPException(status_code=503, detail="Webhook not configured")

    # Get raw body
    payload = await request.body()

    try:
        # Verify webhook signature
        event = stripe.Webhook.construct_event(
            payload,
            stripe_signature,
            settings.stripe_webhook_secret,
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    # Handle the event
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]

        # Extract metadata
        job_id = session.get("metadata", {}).get("job_id")
        product_ids_str = session.get("metadata", {}).get("product_ids", "")
        product_ids = product_ids_str.split(",") if product_ids_str else []
        email = session.get("metadata", {}).get("email") or session.get("customer_email")

        if job_id and product_ids:
            # Mark products as purchased
            mark_purchased(job_id, product_ids)
            print(f"Payment successful for job {job_id}: {product_ids}")

            # Send confirmation email
            if email:
                email_service.send_purchase_confirmation(
                    to_email=email,
                    job_id=job_id,
                    products=product_ids,
                    is_promo=False,
                )

    elif event["type"] == "payment_intent.payment_failed":
        intent = event["data"]["object"]
        print(f"Payment failed: {intent.get('id')}")

    return {"status": "ok"}


@router.get("/verify/{session_id}")
async def verify_payment(session_id: str):
    """
    Verify a payment session status.

    - **session_id**: The Stripe checkout session ID

    Returns payment status and purchased products.
    """
    if not STRIPE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Payment system not configured")

    if session_id == "free":
        return {"status": "paid", "products": []}

    try:
        session = stripe.checkout.Session.retrieve(session_id)

        return {
            "status": session.payment_status,
            "job_id": session.metadata.get("job_id"),
            "products": session.metadata.get("product_ids", "").split(","),
        }

    except stripe.error.StripeError as e:
        raise HTTPException(status_code=400, detail=str(e))
