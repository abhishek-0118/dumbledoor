"""Chat API routes."""

from fastapi import APIRouter, HTTPException, status, Depends, Query
from typing import List, Optional
from ..auth.dependencies import get_current_user, check_query_limit
from ..services.chat_service import chat_service
from ..services.user_service import user_service
from ..models.user import User
from ..models.chat import (
    Chat, ChatCreate, ChatResponse, ChatSummary, 
    MessageCreate, ChatUpdate
)

router = APIRouter(prefix="/chats", tags=["Chats"])

@router.post("/", response_model=ChatResponse)
async def create_chat(
    chat_data: ChatCreate,
    current_user: User = Depends(get_current_user)
):
    """Create a new chat."""
    chat = await chat_service.create_chat(str(current_user.id), chat_data)
    
    return ChatResponse(
        id=str(chat.id),
        title=chat.title,
        messages=chat.messages,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        metadata=chat.metadata
    )

@router.get("/", response_model=List[ChatSummary])
async def get_user_chats(
    limit: int = Query(50, description="Maximum number of chats to return"),
    search: Optional[str] = Query(None, description="Search query for chat titles/content"),
    current_user: User = Depends(get_current_user)
):
    """Get user's chat summaries for sidebar."""
    if search:
        chats = await chat_service.search_user_chats(str(current_user.id), search, limit)
    else:
        chats = await chat_service.get_user_chats(str(current_user.id), limit)
    
    return chats

@router.get("/{chat_id}", response_model=ChatResponse)
async def get_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get a specific chat with all messages."""
    chat = await chat_service.get_chat_by_id(chat_id, str(current_user.id))
    
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    return ChatResponse(
        id=str(chat.id),
        title=chat.title,
        messages=chat.messages,
        created_at=chat.created_at,
        updated_at=chat.updated_at,
        metadata=chat.metadata
    )

@router.post("/{chat_id}/messages", response_model=ChatResponse)
async def add_message_to_chat(
    chat_id: str,
    message_data: MessageCreate,
    current_user: User = Depends(get_current_user)  # Use regular auth check instead of query limit check
):
    """Add a message to an existing chat."""
    # Check if chat exists and belongs to user
    chat = await chat_service.get_chat_by_id(chat_id, str(current_user.id))
    if not chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    # Add message to chat
    updated_chat = await chat_service.add_message_to_chat(
        chat_id, str(current_user.id), message_data
    )
    
    if not updated_chat:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to add message to chat"
        )
    
    # Note: Query count is incremented in the streaming endpoint, not here
    # This endpoint is just for saving messages to the database
    
    return ChatResponse(
        id=str(updated_chat.id),
        title=updated_chat.title,
        messages=updated_chat.messages,
        created_at=updated_chat.created_at,
        updated_at=updated_chat.updated_at,
        metadata=updated_chat.metadata
    )

@router.put("/{chat_id}", response_model=ChatResponse)
async def update_chat(
    chat_id: str,
    update_data: ChatUpdate,
    current_user: User = Depends(get_current_user)
):
    """Update chat information (e.g., title)."""
    updated_chat = await chat_service.update_chat(
        chat_id, str(current_user.id), update_data
    )
    
    if not updated_chat:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found or update failed"
        )
    
    return ChatResponse(
        id=str(updated_chat.id),
        title=updated_chat.title,
        messages=updated_chat.messages,
        created_at=updated_chat.created_at,
        updated_at=updated_chat.updated_at,
        metadata=updated_chat.metadata
    )

@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user)
):
    """Delete a chat."""
    success = await chat_service.delete_chat(chat_id, str(current_user.id))
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Chat not found"
        )
    
    return {"message": "Chat deleted successfully"}
