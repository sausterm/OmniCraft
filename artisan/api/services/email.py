"""
Email service using Resend for sending transactional emails.
"""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Check if Resend is configured
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
RESEND_AVAILABLE = bool(RESEND_API_KEY)

if RESEND_AVAILABLE:
    import resend
    resend.api_key = RESEND_API_KEY


class EmailService:
    """Service for sending emails via Resend."""

    def __init__(self):
        self.from_email = os.getenv("FROM_EMAIL", "Artisan <onboarding@resend.dev>")
        self.app_url = os.getenv("APP_URL", "http://localhost:3000")

    def send_purchase_confirmation(
        self,
        to_email: str,
        job_id: str,
        products: list[str],
        is_promo: bool = False,
    ) -> bool:
        """
        Send purchase confirmation with download links.

        Returns True if sent successfully, False otherwise.
        """
        if not RESEND_AVAILABLE:
            logger.warning("Email not configured - skipping confirmation email")
            return False

        download_url = f"{self.app_url}/download/{job_id}"

        # Build product list HTML
        product_list = "".join([
            f"<li>{p.replace('_', ' ').title()}</li>"
            for p in products
        ])

        subject = "Your Paint-by-Numbers Kit is Ready!" if not is_promo else "Your Free Premium Kit is Ready!"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #059669; margin: 0;">Artisan</h1>
        <p style="color: #6b7280; margin: 5px 0;">Paint-by-Numbers Generator</p>
    </div>

    <div style="background: linear-gradient(135deg, #f0fdf4 0%, #ecfeff 100%); border-radius: 16px; padding: 30px; margin-bottom: 20px;">
        <h2 style="margin-top: 0; color: #111827;">{"Your Free Premium Kit is Ready!" if is_promo else "Thank You for Your Purchase!"}</h2>

        <p style="color: #4b5563;">Your personalized paint-by-numbers kit has been created and is ready for download.</p>

        <div style="background: white; border-radius: 8px; padding: 15px; margin: 20px 0;">
            <h3 style="margin-top: 0; color: #111827;">What's Included:</h3>
            <ul style="color: #4b5563; padding-left: 20px;">
                {product_list}
            </ul>
        </div>

        <a href="{download_url}"
           style="display: inline-block; background: #059669; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; margin-top: 10px;">
            Download Your Kit
        </a>
    </div>

    <div style="background: #fef3c7; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
        <h3 style="margin-top: 0; color: #92400e;">Bob Ross Says...</h3>
        <p style="color: #92400e; font-style: italic; margin-bottom: 0;">
            "There are no mistakes, only happy little accidents. You're going to do great!"
        </p>
    </div>

    <div style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 30px;">
        <p>Need help? Reply to this email or visit our support page.</p>
        <p>Artisan Paint-by-Numbers &bull; Made with love</p>
    </div>
</body>
</html>
        """

        try:
            result = resend.Emails.send({
                "from": self.from_email,
                "to": [to_email],
                "subject": subject,
                "html": html_content,
            })
            logger.info(f"Purchase confirmation sent to {to_email}: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to send email to {to_email}: {e}")
            return False

    def send_magic_link(
        self,
        to_email: str,
        token: str,
        expires_in_hours: int = 24,
    ) -> bool:
        """
        Send a magic link for passwordless login.

        Returns True if sent successfully, False otherwise.
        """
        if not RESEND_AVAILABLE:
            logger.warning("Email not configured - skipping magic link email")
            return False

        login_url = f"{self.app_url}/api/auth/callback/email?token={token}&email={to_email}"

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="text-align: center; margin-bottom: 30px;">
        <h1 style="color: #059669; margin: 0;">Artisan</h1>
        <p style="color: #6b7280; margin: 5px 0;">Paint-by-Numbers Generator</p>
    </div>

    <div style="background: #f3f4f6; border-radius: 16px; padding: 30px; text-align: center;">
        <h2 style="margin-top: 0; color: #111827;">Sign in to Artisan</h2>

        <p style="color: #4b5563;">Click the button below to sign in to your account. This link will expire in {expires_in_hours} hours.</p>

        <a href="{login_url}"
           style="display: inline-block; background: #059669; color: white; padding: 14px 28px; border-radius: 8px; text-decoration: none; font-weight: 600; margin: 20px 0;">
            Sign In
        </a>

        <p style="color: #9ca3af; font-size: 14px;">
            If you didn't request this email, you can safely ignore it.
        </p>
    </div>

    <div style="text-align: center; color: #9ca3af; font-size: 12px; margin-top: 30px;">
        <p>Artisan Paint-by-Numbers &bull; Made with love</p>
    </div>
</body>
</html>
        """

        try:
            result = resend.Emails.send({
                "from": self.from_email,
                "to": [to_email],
                "subject": "Sign in to Artisan",
                "html": html_content,
            })
            logger.info(f"Magic link sent to {to_email}: {result}")
            return True
        except Exception as e:
            logger.error(f"Failed to send magic link to {to_email}: {e}")
            return False


# Singleton instance
email_service = EmailService()
