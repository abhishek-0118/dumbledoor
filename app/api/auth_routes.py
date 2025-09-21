"""Authentication API routes."""

from fastapi import APIRouter, HTTPException, status, Depends, Query, Request
from fastapi.responses import RedirectResponse
from typing import Dict, Any
from ..auth.google_oauth import google_oauth
from ..auth.dependencies import get_current_user
from ..services.user_service import user_service
from ..models.user import User, UserResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.get("/google/login")
async def google_login():
    """Initiate Google OAuth login."""
    auth_url = google_oauth.get_authorization_url()
    return {"auth_url": auth_url}

@router.get("/google")
async def google_callback(
    code: str = Query(..., description="Authorization code from Google"),
    error: str = Query(None, description="Error from Google OAuth")
):
    """Handle Google OAuth callback."""
    if error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google OAuth error: {error}"
        )
    
    try:
        # Exchange code for token
        token_data = await google_oauth.exchange_code_for_token(code)
        access_token = token_data.get("access_token")
        
        if not access_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to get access token"
            )
        
        # Get user info from Google
        google_user_data = await google_oauth.get_user_info(access_token)
        
        # Create or get user
        user = await user_service.create_user_from_google_data(google_user_data)
        
        # Generate JWT token
        jwt_token = await user_service.generate_user_token(user)
        
        # Redirect to frontend main screen with token
        frontend_url = "http://localhost:3000"
        redirect_url = f"{frontend_url}/?token={jwt_token}"
        
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Authentication failed: {str(e)}"
        )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information."""
    return UserResponse(
        id=str(current_user.id),
        email=current_user.email,
        name=current_user.name,
        picture=current_user.picture,
        settings=current_user.settings,
        created_at=current_user.created_at,
        last_login=current_user.last_login
    )

@router.post("/logout")
async def logout():
    """Logout user (client-side token removal)."""
    return {"message": "Logged out successfully"}

@router.get("/status")
async def auth_status(current_user: User = Depends(get_current_user)):
    """Check authentication status."""
    return {
        "authenticated": True,
        "user_id": str(current_user.id),
        "email": current_user.email,
        "queries_used": current_user.settings.queries_used,
        "query_limit": current_user.settings.query_limit
    }
