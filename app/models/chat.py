from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from bson import ObjectId
from enum import Enum

from .user import PyObjectId


class MessageRole(str, Enum):
    """Message roles in a conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Individual chat message"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    session_id: PyObjectId
    role: MessageRole
    content: str
    
    # Context and metadata
    query_analysis: Optional[Dict[str, Any]] = None
    sources: Optional[List[Dict[str, Any]]] = None
    context_summary: Optional[Dict[str, Any]] = None
    total_sources_found: Optional[int] = None
    
    # Token usage and costs
    token_usage: Optional[Dict[str, Any]] = None
    estimated_cost: Optional[float] = None
    
    # Response metadata
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    response_time_ms: Optional[int] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Status
    is_deleted: bool = False

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ChatSession(BaseModel):
    """Chat session containing multiple messages"""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    user_id: PyObjectId
    
    # Session metadata
    title: str = "New Chat"
    description: Optional[str] = None
    
    # Repository context
    repo_context: Optional[str] = None  # Specific repo being discussed
    repositories: List[str] = Field(default_factory=list)  # All repos accessed
    
    # Conversation settings
    settings: Dict[str, Any] = Field(default_factory=lambda: {
        "k": 20,
        "alpha": 0.3,
        "architectural": False,
        "include_context": True,
        "detailed_response": True
    })
    
    # Session statistics
    message_count: int = 0
    total_tokens_used: Optional[int] = None
    total_cost: Optional[float] = None
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
    
    # Status
    is_active: bool = True
    is_deleted: bool = False

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


class ConversationBuffer(BaseModel):
    """Conversation buffer for maintaining context across messages"""
    session_id: PyObjectId
    
    # Memory configuration
    max_token_limit: int = 4000
    max_messages: int = 20
    
    # Buffer content
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    summary: Optional[str] = None
    
    # Context tracking
    current_repo: Optional[str] = None
    active_files: List[str] = Field(default_factory=list)
    key_concepts: List[str] = Field(default_factory=list)
    
    # Buffer metadata
    total_tokens: int = 0
    last_summarized_at: Optional[datetime] = None
    buffer_full: bool = False
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}


# Request/Response models for API

class ChatRequest(BaseModel):
    """Request model for chat endpoint"""
    message: str
    session_id: Optional[str] = None
    repo: Optional[str] = None
    k: int = 20
    alpha: float = 0.3
    architectural: bool = False
    include_context: bool = True
    detailed_response: bool = True


class ChatResponse(BaseModel):
    """Response model for chat endpoint"""
    session_id: str
    message_id: str
    response: str
    sources: List[Dict[str, Any]]
    context_summary: Dict[str, Any]
    query_analysis: Dict[str, Any]
    total_sources_found: int
    token_usage: Optional[Dict[str, Any]] = None
    estimated_cost: Optional[float] = None


class SessionCreate(BaseModel):
    """Model for creating a new chat session"""
    title: Optional[str] = "New Chat"
    description: Optional[str] = None
    repo_context: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class SessionUpdate(BaseModel):
    """Model for updating a chat session"""
    title: Optional[str] = None
    description: Optional[str] = None
    repo_context: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class SessionSummary(BaseModel):
    """Summary model for listing sessions"""
    id: str
    title: str
    description: Optional[str]
    message_count: int
    last_activity: datetime
    repo_context: Optional[str]
    total_cost: Optional[float]
    created_at: datetime
