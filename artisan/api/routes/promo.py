"""
Promo code endpoints - One-time use codes for friends/family beta testing.
"""

import secrets
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory promo code storage
# In production, use a database
promo_codes_db: Dict[str, dict] = {}


class PromoCode(BaseModel):
    code: str
    created_at: str
    used: bool
    used_at: Optional[str] = None
    used_by_job: Optional[str] = None
    note: Optional[str] = None
    tier: str = "premium"  # premium, basic, unlimited


class PromoCodeCreate(BaseModel):
    count: int = 1
    note: Optional[str] = None
    tier: str = "premium"


class PromoCodeValidate(BaseModel):
    code: str
    job_id: str


class PromoCodeResponse(BaseModel):
    valid: bool
    message: str
    tier: Optional[str] = None


def generate_promo_code() -> str:
    """Generate a friendly, easy-to-share promo code."""
    # Format: ARTISAN-XXXX-XXXX (easy to read/type)
    part1 = secrets.token_hex(2).upper()
    part2 = secrets.token_hex(2).upper()
    return f"ARTISAN-{part1}-{part2}"


@router.post("/promo/generate")
async def generate_promo_codes(request: PromoCodeCreate) -> Dict[str, Any]:
    """
    Generate one-time use promo codes.

    - **count**: Number of codes to generate (1-50)
    - **note**: Optional note (e.g., "For Mom", "Beta tester")
    - **tier**: Access tier (premium, basic, unlimited)

    Returns list of generated codes.
    """
    if request.count < 1 or request.count > 50:
        raise HTTPException(status_code=400, detail="Count must be 1-50")

    codes = []
    for _ in range(request.count):
        code = generate_promo_code()

        # Ensure unique
        while code in promo_codes_db:
            code = generate_promo_code()

        promo_codes_db[code] = {
            "code": code,
            "created_at": datetime.utcnow().isoformat(),
            "used": False,
            "used_at": None,
            "used_by_job": None,
            "note": request.note,
            "tier": request.tier,
        }
        codes.append(code)
        logger.info(f"Generated promo code: {code} (tier={request.tier}, note={request.note})")

    return {"codes": codes, "tier": request.tier}


@router.post("/promo/validate", response_model=PromoCodeResponse)
async def validate_promo_code(request: PromoCodeValidate) -> PromoCodeResponse:
    """
    Validate and redeem a promo code.

    - **code**: The promo code to validate
    - **job_id**: The job to apply it to

    If valid, marks the code as used and returns success.
    """
    code = request.code.upper().strip()

    if code not in promo_codes_db:
        logger.warning(f"Invalid promo code attempt: {code}")
        return PromoCodeResponse(
            valid=False,
            message="Invalid promo code. Please check and try again."
        )

    promo = promo_codes_db[code]

    if promo["used"]:
        logger.warning(f"Already used promo code attempt: {code}")
        return PromoCodeResponse(
            valid=False,
            message="This promo code has already been used."
        )

    # Mark as used
    promo["used"] = True
    promo["used_at"] = datetime.utcnow().isoformat()
    promo["used_by_job"] = request.job_id

    logger.info(f"Promo code redeemed: {code} for job {request.job_id}")

    return PromoCodeResponse(
        valid=True,
        message=f"Promo code applied! You have full {promo['tier']} access.",
        tier=promo["tier"]
    )


@router.get("/promo/list")
async def list_promo_codes() -> Dict[str, Any]:
    """
    List all promo codes (admin endpoint).

    Returns all codes with their status.
    """
    codes = list(promo_codes_db.values())

    # Sort: unused first, then by creation date
    codes.sort(key=lambda x: (x["used"], x["created_at"]))

    return {
        "codes": codes,
        "total": len(codes),
        "used": sum(1 for c in codes if c["used"]),
        "available": sum(1 for c in codes if not c["used"]),
    }


@router.delete("/promo/{code}")
async def delete_promo_code(code: str) -> Dict[str, str]:
    """
    Delete a promo code (admin endpoint).
    """
    code = code.upper().strip()

    if code not in promo_codes_db:
        raise HTTPException(status_code=404, detail="Promo code not found")

    del promo_codes_db[code]
    logger.info(f"Deleted promo code: {code}")

    return {"status": "deleted", "code": code}


# Pre-generate some codes for immediate use
def init_promo_codes():
    """Initialize some promo codes on startup for testing."""
    initial_codes = [
        {"note": "Sloan's testing", "tier": "premium"},
        {"note": "Family member 1", "tier": "premium"},
        {"note": "Family member 2", "tier": "premium"},
        {"note": "Friend 1", "tier": "premium"},
        {"note": "Friend 2", "tier": "premium"},
    ]

    for config in initial_codes:
        code = generate_promo_code()
        promo_codes_db[code] = {
            "code": code,
            "created_at": datetime.utcnow().isoformat(),
            "used": False,
            "used_at": None,
            "used_by_job": None,
            "note": config["note"],
            "tier": config["tier"],
        }
        print(f"  Promo code: {code} ({config['note']})")


# Initialize on module load
init_promo_codes()
