"""JWT authentication utilities."""

import os
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from fastapi import HTTPException, status
from passlib.context import CryptContext

from ..constants.system import ENV_VARS

# Configuration
SECRET_KEY = os.getenv(ENV_VARS["JWT_SECRET"], "your-secret-key-here-change-in-production")
ALGORITHM = ENV_VARS["JWT_ALGORITHM"]

ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv(ENV_VARS["ACCESS_TOKEN_EXPIRE_MINUTES"], str(ENV_VARS["DEFAULT_TOKEN_EXPIRE"])))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_user_id_from_token(token: str) -> str:
    """Extract user ID from JWT token."""
    payload = verify_token(token)
    user_id = payload.get("sub")
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id

def create_user_token(user_id: str, email: str, name: str) -> str:
    """Create JWT token for user."""
    token_data = {
        "sub": user_id,
        "email": email,
        "name": name,
        "iat": datetime.utcnow(),
    }
    return create_access_token(token_data)
