"""User model and Pydantic schemas."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, field=None):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")
        return field_schema

class UserSettings(BaseModel):
    """User settings model."""
    query_limit: int = Field(default=10, description="Maximum queries per session")
    queries_used: int = Field(default=0, description="Queries used in current session")
    reset_date: datetime = Field(default_factory=datetime.utcnow, description="Last reset date")

class User(BaseModel):
    """User model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    google_id: str = Field(..., description="Google OAuth ID")
    email: str = Field(..., description="User email")
    name: str = Field(..., description="User full name")
    picture: Optional[str] = Field(None, description="User profile picture URL")
    settings: UserSettings = Field(default_factory=UserSettings)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = Field(None)

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class UserCreate(BaseModel):
    """User creation schema."""
    google_id: str
    email: str
    name: str
    picture: Optional[str] = None

class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    name: str
    picture: Optional[str]
    settings: UserSettings
    created_at: datetime
    last_login: Optional[datetime]

class UserUpdate(BaseModel):
    """User update schema."""
    name: Optional[str] = None
    picture: Optional[str] = None
    settings: Optional[UserSettings] = None
