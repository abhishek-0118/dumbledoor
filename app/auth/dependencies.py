"""Authentication dependencies for FastAPI."""

from fastapi import Depends, HTTPException, status, Query, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from ..db.mongodb import get_database
from ..models.user import User
from .jwt_auth import verify_token, get_user_id_from_token
from bson import ObjectId

security = HTTPBearer()

async def get_token_from_request(
    request: Request,
    token: Optional[str] = Query(None, description="Authentication token for EventSource"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[str]:
    """Extract token from either Authorization header or query parameter."""
    # First try to get from Authorization header
    if credentials:
        return credentials.credentials
    
    # Fallback to query parameter for EventSource requests
    if token:
        return token
    
    return None

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current authenticated user."""
    try:
        # Verify JWT token
        token = credentials.credentials
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database with async connection
        from ..services.user_service import user_service
        user = await user_service.get_user_by_id(user_id)
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return user
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_optional_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """Get current user if authenticated, otherwise None."""
    if credentials is None:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None

async def get_current_user_flexible(
    token: Optional[str] = Depends(get_token_from_request)
) -> Optional[User]:
    """Get current user with flexible token authentication (header or query param)."""
    if not token:
        return None
    
    try:
        # Verify JWT token
        payload = verify_token(token)
        user_id = payload.get("sub")
        
        if user_id is None:
            return None
        
        # Get user from database with async connection
        from ..services.user_service import user_service
        user = await user_service.get_user_by_id(user_id)
        
        return user
        
    except Exception as e:
        # Log error but don't raise for flexible auth
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Token authentication failed: {e}")
        return None

async def get_current_user_required(
    user: Optional[User] = Depends(get_current_user_flexible)
) -> User:
    """Get current user but require authentication."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authenticated"
        )
    return user

async def check_query_limit(current_user: User = Depends(get_current_user)) -> User:
    """Check if user has exceeded query limit."""
    if current_user.settings.queries_used >= current_user.settings.query_limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Query limit exceeded. You have used {current_user.settings.queries_used}/{current_user.settings.query_limit} queries."
        )
    
    return current_user
