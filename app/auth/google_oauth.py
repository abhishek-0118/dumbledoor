"""Google OAuth integration."""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from authlib.integrations.httpx_client import AsyncOAuth2Client
from fastapi import HTTPException, status
import httpx

def load_oauth_config() -> Dict[str, Any]:
    """Load OAuth configuration from JSON file."""
    config_path = Path(__file__).parent.parent.parent / "oauth-client.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"OAuth configuration file not found at {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config["web"]
    except (json.JSONDecodeError, KeyError) as e:
        raise ValueError(f"Invalid OAuth configuration file: {e}")

# Load Google OAuth Configuration from JSON file
_oauth_config = load_oauth_config()
GOOGLE_CLIENT_ID = _oauth_config["client_id"]
GOOGLE_CLIENT_SECRET = _oauth_config["client_secret"]
GOOGLE_REDIRECT_URI = _oauth_config["redirect_uris"][0]  

class GoogleOAuth:
    def __init__(self):
        self.client_id = GOOGLE_CLIENT_ID
        self.client_secret = GOOGLE_CLIENT_SECRET
        self.redirect_uri = GOOGLE_REDIRECT_URI
        self.authorization_url = "https://accounts.google.com/o/oauth2/auth"
        self.token_url = "https://oauth2.googleapis.com/token"
        self.userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"

    def get_authorization_url(self, state: str = None) -> str:
        """Generate Google OAuth authorization URL."""
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": "openid email profile",
            "response_type": "code",
            "access_type": "offline",
            # Remove "prompt": "consent" to avoid showing consent screen every time
            # This will only show consent on first login or when permissions change
        }
        
        if state:
            params["state"] = state
        
        query_string = "&".join([f"{k}={v}" for k, v in params.items()])
        return f"{self.authorization_url}?{query_string}"

    async def exchange_code_for_token(self, code: str) -> Dict[str, Any]:
        """Exchange authorization code for access token."""
        # Check if client secret is configured
        if not self.client_secret:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth client secret not configured. Please check oauth-client.json file."
            )
        
        async with httpx.AsyncClient() as client:
            data = {
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": self.redirect_uri,
            }
            
            response = await client.post(self.token_url, data=data)
            
            if response.status_code != 200:
                try:
                    error_details = response.json()
                    error_msg = error_details.get("error_description", "Unknown error")
                except:
                    error_msg = response.text
                
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Failed to exchange code for token: {error_msg}"
                )
            
            return response.json()

    async def get_user_info(self, access_token: str) -> Dict[str, Any]:
        """Get user information from Google API."""
        async with httpx.AsyncClient() as client:
            headers = {"Authorization": f"Bearer {access_token}"}
            response = await client.get(self.userinfo_url, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to get user information"
                )
            
            return response.json()

    async def verify_google_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify Google ID token."""
        async with httpx.AsyncClient() as client:
            url = f"https://oauth2.googleapis.com/tokeninfo?id_token={token}"
            response = await client.get(url)
            
            if response.status_code != 200:
                return None
            
            token_info = response.json()
            
            # Verify the token is for our application
            if token_info.get("aud") != self.client_id:
                return None
            
            return token_info

# Global instance
google_oauth = GoogleOAuth()
