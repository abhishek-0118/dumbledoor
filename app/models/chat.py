"""Chat model and Pydantic schemas."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from .user import PyObjectId

class Message(BaseModel):
    """Individual message in a chat."""
    id: str = Field(default_factory=lambda: str(ObjectId()))
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional message metadata")

class Chat(BaseModel):
    """Chat model."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: str = Field(..., description="User ID who owns this chat")
    title: str = Field(..., description="Chat title/heading")
    messages: List[Message] = Field(default_factory=list, description="Chat messages")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional chat metadata")

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

class ChatCreate(BaseModel):
    """Chat creation schema."""
    title: str
    first_message: Optional[str] = None

class ChatResponse(BaseModel):
    """Chat response schema."""
    id: str
    title: str
    messages: List[Message]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]]

class ChatSummary(BaseModel):
    """Chat summary for sidebar listing."""
    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    message_count: int

class MessageCreate(BaseModel):
    """Message creation schema."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")
    metadata: Optional[Dict[str, Any]] = None

class ChatUpdate(BaseModel):
    """Chat update schema."""
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
