import os
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import secrets
import jwt
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from ..models import User, UserCreate, UserSession, PyObjectId
from ..db.mongodb import mongodb

logger = logging.getLogger("app.auth.google_oauth")

# Security
security = HTTPBearer()


class GoogleOAuthManager:
    """Google OAuth authentication manager"""
    
    def __init__(self):
        self.client_id = os.getenv("GOOGLE_CLIENT_ID")
        self.client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        self.redirect_uri = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
        self.jwt_secret = os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32))
        self.jwt_algorithm = "HS256"
        self.jwt_expire_hours = int(os.getenv("JWT_EXPIRE_HOURS", "24"))
        
        if not self.client_id or not self.client_secret:
            logger.warning("Google OAuth credentials not found. Set GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET environment variables.")
    
    def get_authorization_url(self, state: Optional[str] = None) -> str:
        """Get the Google OAuth authorization URL"""
        if not self.client_id:
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
        
        # Create OAuth flow
        flow = Flow.from_client_config(
            {
                "web": {
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": [self.redirect_uri]
                }
            },
            scopes=[
                "openid",
                "email",
                "profile"
            ]
        )
        flow.redirect_uri = self.redirect_uri
        
        authorization_url, _ = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            state=state
        )
        
        return authorization_url
    
    async def handle_oauth_callback(self, code: str, state: Optional[str] = None) -> Dict[str, Any]:
        """Handle the OAuth callback and create/update user"""
        if not self.client_id:
            raise HTTPException(status_code=500, detail="Google OAuth not configured")
        
        try:
            # Create OAuth flow
            flow = Flow.from_client_config(
                {
                    "web": {
                        "client_id": self.client_id,
                        "client_secret": self.client_secret,
                        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "redirect_uris": [self.redirect_uri]
                    }
                },
                scopes=[
                    "openid",
                    "email", 
                    "profile"
                ]
            )
            flow.redirect_uri = self.redirect_uri
            
            # Exchange code for tokens
            flow.fetch_token(code=code)
            
            # Get user info from ID token
            id_info = id_token.verify_oauth2_token(
                flow.credentials.id_token,
                google_requests.Request(),
                self.client_id
            )
            
            # Create or update user
            user_data = UserCreate(
                google_id=id_info['sub'],
                email=id_info['email'],
                name=id_info.get('name', ''),
                picture_url=id_info.get('picture'),
                given_name=id_info.get('given_name'),
                family_name=id_info.get('family_name'),
                locale=id_info.get('locale'),
                verified_email=id_info.get('email_verified', False)
            )
            
            user = await self._create_or_update_user(user_data, flow.credentials)
            
            # Create session and JWT token
            session_token = await self._create_user_session(user.id)
            jwt_token = self._create_jwt_token(user)
            
            return {
                "user": user.dict(),
                "access_token": jwt_token,
                "session_token": session_token,
                "token_type": "bearer"
            }
            
        except Exception as e:
            logger.error(f"OAuth callback failed: {e}")
            raise HTTPException(status_code=400, detail=f"OAuth authentication failed: {str(e)}")
    
    async def _create_or_update_user(self, user_data: UserCreate, credentials) -> User:
        """Create a new user or update existing user"""
        existing_user = await mongodb.users.find_one({"google_id": user_data.google_id})
        
        if existing_user:
            # Update existing user
            update_data = {
                "name": user_data.name,
                "picture_url": user_data.picture_url,
                "given_name": user_data.given_name,
                "family_name": user_data.family_name,
                "locale": user_data.locale,
                "verified_email": user_data.verified_email,
                "access_token": credentials.token,
                "refresh_token": credentials.refresh_token,
                "token_expires_at": credentials.expiry,
                "updated_at": datetime.utcnow(),
                "last_login": datetime.utcnow()
            }
            
            await mongodb.users.update_one(
                {"google_id": user_data.google_id},
                {"$set": update_data}
            )
            
            # Fetch updated user
            updated_user = await mongodb.users.find_one({"google_id": user_data.google_id})
            return User(**updated_user)
        else:
            # Create new user
            new_user = User(
                google_id=user_data.google_id,
                email=user_data.email,
                name=user_data.name,
                picture_url=user_data.picture_url,
                given_name=user_data.given_name,
                family_name=user_data.family_name,
                locale=user_data.locale,
                verified_email=user_data.verified_email,
                access_token=credentials.token,
                refresh_token=credentials.refresh_token,
                token_expires_at=credentials.expiry,
                last_login=datetime.utcnow()
            )
            
            result = await mongodb.users.insert_one(new_user.dict(by_alias=True))
            new_user.id = result.inserted_id
            
            logger.info(f"Created new user: {new_user.email}")
            return new_user
    
    async def _create_user_session(self, user_id: PyObjectId) -> str:
        """Create a user session"""
        session_token = secrets.token_urlsafe(32)
        expires_at = datetime.utcnow() + timedelta(hours=self.jwt_expire_hours)
        
        session = UserSession(
            user_id=user_id,
            session_token=session_token,
            expires_at=expires_at
        )
        
        await mongodb.user_sessions.insert_one(session.dict(by_alias=True))
        return session_token
    
    def _create_jwt_token(self, user: User) -> str:
        """Create a JWT token for the user"""
        payload = {
            "user_id": str(user.id),
            "email": user.email,
            "name": user.name,
            "exp": datetime.utcnow() + timedelta(hours=self.jwt_expire_hours),
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    async def verify_jwt_token(self, token: str) -> Optional[User]:
        """Verify a JWT token and return the user"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            user_id = payload.get("user_id")
            
            if not user_id:
                return None
            
            user_doc = await mongodb.users.find_one({"_id": PyObjectId(user_id)})
            if not user_doc:
                return None
            
            return User(**user_doc)
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid JWT token")
            return None
        except Exception as e:
            logger.error(f"JWT verification failed: {e}")
            return None
    
    async def verify_session_token(self, session_token: str) -> Optional[User]:
        """Verify a session token and return the user"""
        try:
            session_doc = await mongodb.user_sessions.find_one({
                "session_token": session_token,
                "is_active": True,
                "expires_at": {"$gt": datetime.utcnow()}
            })
            
            if not session_doc:
                return None
            
            user_doc = await mongodb.users.find_one({"_id": session_doc["user_id"]})
            if not user_doc:
                return None
            
            return User(**user_doc)
            
        except Exception as e:
            logger.error(f"Session verification failed: {e}")
            return None
    
    async def revoke_session(self, session_token: str) -> bool:
        """Revoke a user session"""
        try:
            result = await mongodb.user_sessions.update_one(
                {"session_token": session_token},
                {"$set": {"is_active": False}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Session revocation failed: {e}")
            return False


# Global OAuth manager instance
oauth_manager = GoogleOAuthManager()


async def get_current_user(credentials: HTTPAuthorizationCredentials = security) -> User:
    """FastAPI dependency to get current authenticated user"""
    token = credentials.credentials
    user = await oauth_manager.verify_jwt_token(token)
    
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_optional_user(request: Request) -> Optional[User]:
    """FastAPI dependency to get current user if authenticated (optional)"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    return await oauth_manager.verify_jwt_token(token)
