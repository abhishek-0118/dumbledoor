import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from bson import ObjectId

from ..models import (
    ChatSession, ChatMessage, ConversationBuffer, MessageRole, 
    SessionCreate, SessionUpdate, SessionSummary, PyObjectId, User
)
from ..db.mongodb import mongodb
from .conversation_buffer import ConversationBufferManager

logger = logging.getLogger("app.core.chat_history")


class ChatHistoryManager:
    """Manages chat sessions and message history with MongoDB"""
    
    def __init__(self, conversation_manager: ConversationBufferManager):
        self.conversation_manager = conversation_manager
    
    async def create_session(
        self, 
        user_id: PyObjectId, 
        session_data: SessionCreate
    ) -> ChatSession:
        """Create a new chat session"""
        
        session = ChatSession(
            user_id=user_id,
            title=session_data.title or "New Chat",
            description=session_data.description,
            repo_context=session_data.repo_context,
            settings=session_data.settings or {}
        )
        
        # Insert into database
        result = await mongodb.chat_sessions.insert_one(session.dict(by_alias=True))
        session.id = result.inserted_id
        
        # Create conversation buffer for this session
        await self.conversation_manager.get_or_create_buffer(session.id)
        
        logger.info(f"Created new chat session {session.id} for user {user_id}")
        return session
    
    async def get_user_sessions(
        self, 
        user_id: PyObjectId, 
        limit: int = 50,
        offset: int = 0,
        include_deleted: bool = False
    ) -> List[SessionSummary]:
        """Get all sessions for a user"""
        
        query = {"user_id": user_id}
        if not include_deleted:
            query["is_deleted"] = {"$ne": True}
        
        sessions_cursor = mongodb.chat_sessions.find(query).sort(
            "last_activity", -1
        ).skip(offset).limit(limit)
        
        sessions = []
        async for session_doc in sessions_cursor:
            session = ChatSession(**session_doc)
            sessions.append(SessionSummary(
                id=str(session.id),
                title=session.title,
                description=session.description,
                message_count=session.message_count,
                last_activity=session.last_activity,
                repo_context=session.repo_context,
                total_cost=session.total_cost,
                created_at=session.created_at
            ))
        
        return sessions
    
    async def get_session(
        self, 
        session_id: PyObjectId, 
        user_id: Optional[PyObjectId] = None
    ) -> Optional[ChatSession]:
        """Get a specific session"""
        
        query = {"_id": session_id}
        if user_id:
            query["user_id"] = user_id
        
        session_doc = await mongodb.chat_sessions.find_one(query)
        if not session_doc:
            return None
        
        return ChatSession(**session_doc)
    
    async def update_session(
        self, 
        session_id: PyObjectId, 
        user_id: PyObjectId,
        update_data: SessionUpdate
    ) -> Optional[ChatSession]:
        """Update a session"""
        
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        update_dict["updated_at"] = datetime.utcnow()
        
        result = await mongodb.chat_sessions.update_one(
            {"_id": session_id, "user_id": user_id},
            {"$set": update_dict}
        )
        
        if result.modified_count == 0:
            return None
        
        return await self.get_session(session_id, user_id)
    
    async def delete_session(
        self, 
        session_id: PyObjectId, 
        user_id: PyObjectId,
        hard_delete: bool = False
    ) -> bool:
        """Delete a session (soft delete by default)"""
        
        if hard_delete:
            # Hard delete: remove session and all messages
            await mongodb.chat_messages.delete_many({"session_id": session_id})
            await mongodb.conversation_buffers.delete_one({"session_id": session_id})
            result = await mongodb.chat_sessions.delete_one({
                "_id": session_id, 
                "user_id": user_id
            })
        else:
            # Soft delete: mark as deleted
            result = await mongodb.chat_sessions.update_one(
                {"_id": session_id, "user_id": user_id},
                {"$set": {"is_deleted": True, "updated_at": datetime.utcnow()}}
            )
        
        return result.modified_count > 0 or result.deleted_count > 0
    
    async def add_message(
        self,
        session_id: PyObjectId,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChatMessage:
        """Add a message to a session"""
        
        message = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            query_analysis=metadata.get("query_analysis") if metadata else None,
            sources=metadata.get("sources") if metadata else None,
            context_summary=metadata.get("context_summary") if metadata else None,
            total_sources_found=metadata.get("total_sources_found") if metadata else None,
            token_usage=metadata.get("token_usage") if metadata else None,
            estimated_cost=metadata.get("estimated_cost") if metadata else None,
            model_used=metadata.get("model_used") if metadata else None,
            provider_used=metadata.get("provider_used") if metadata else None,
            response_time_ms=metadata.get("response_time_ms") if metadata else None
        )
        
        # Insert message into database
        result = await mongodb.chat_messages.insert_one(message.dict(by_alias=True))
        message.id = result.inserted_id
        
        # Update session statistics
        await self._update_session_stats(session_id, message)
        
        # Add to conversation buffer
        await self.conversation_manager.add_message_to_buffer(
            session_id, role, content, metadata
        )
        
        logger.debug(f"Added {role.value} message to session {session_id}")
        return message
    
    async def get_session_messages(
        self,
        session_id: PyObjectId,
        limit: int = 100,
        offset: int = 0,
        include_deleted: bool = False
    ) -> List[ChatMessage]:
        """Get messages for a session"""
        
        query = {"session_id": session_id}
        if not include_deleted:
            query["is_deleted"] = {"$ne": True}
        
        messages_cursor = mongodb.chat_messages.find(query).sort(
            "created_at", 1
        ).skip(offset).limit(limit)
        
        messages = []
        async for msg_doc in messages_cursor:
            messages.append(ChatMessage(**msg_doc))
        
        return messages
    
    async def delete_message(
        self,
        message_id: PyObjectId,
        session_id: PyObjectId,
        hard_delete: bool = False
    ) -> bool:
        """Delete a message"""
        
        if hard_delete:
            result = await mongodb.chat_messages.delete_one({
                "_id": message_id,
                "session_id": session_id
            })
        else:
            result = await mongodb.chat_messages.update_one(
                {"_id": message_id, "session_id": session_id},
                {"$set": {"is_deleted": True}}
            )
        
        if result.modified_count > 0 or result.deleted_count > 0:
            # Update session message count
            await self._recalculate_session_stats(session_id)
            return True
        
        return False
    
    async def _update_session_stats(self, session_id: PyObjectId, message: ChatMessage):
        """Update session statistics after adding a message"""
        
        update_data = {
            "$inc": {"message_count": 1},
            "$set": {"last_activity": datetime.utcnow(), "updated_at": datetime.utcnow()}
        }
        
        # Update token usage and cost if available
        if message.token_usage:
            total_tokens = message.token_usage.get("total_tokens", 0)
            if total_tokens > 0:
                update_data["$inc"]["total_tokens_used"] = total_tokens
        
        if message.estimated_cost:
            update_data["$inc"]["total_cost"] = message.estimated_cost
        
        # Track repositories accessed
        if message.sources:
            repos = set()
            for source in message.sources:
                if isinstance(source, dict) and source.get("repo"):
                    repos.add(source["repo"])
            
            if repos:
                update_data["$addToSet"] = {"repositories": {"$each": list(repos)}}
        
        await mongodb.chat_sessions.update_one(
            {"_id": session_id},
            update_data
        )
    
    async def _recalculate_session_stats(self, session_id: PyObjectId):
        """Recalculate session statistics"""
        
        # Count active messages
        message_count = await mongodb.chat_messages.count_documents({
            "session_id": session_id,
            "is_deleted": {"$ne": True}
        })
        
        # Calculate total tokens and cost
        pipeline = [
            {"$match": {"session_id": session_id, "is_deleted": {"$ne": True}}},
            {"$group": {
                "_id": None,
                "total_tokens": {"$sum": "$token_usage.total_tokens"},
                "total_cost": {"$sum": "$estimated_cost"}
            }}
        ]
        
        result = await mongodb.chat_messages.aggregate(pipeline).to_list(1)
        total_tokens = result[0]["total_tokens"] if result else 0
        total_cost = result[0]["total_cost"] if result else 0.0
        
        # Update session
        await mongodb.chat_sessions.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "message_count": message_count,
                    "total_tokens_used": total_tokens,
                    "total_cost": total_cost,
                    "updated_at": datetime.utcnow()
                }
            }
        )
    
    async def search_sessions(
        self,
        user_id: PyObjectId,
        query: str,
        limit: int = 20
    ) -> List[SessionSummary]:
        """Search sessions by title, description, or message content"""
        
        # Text search in sessions
        session_query = {
            "user_id": user_id,
            "is_deleted": {"$ne": True},
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"description": {"$regex": query, "$options": "i"}}
            ]
        }
        
        sessions = []
        sessions_cursor = mongodb.chat_sessions.find(session_query).sort(
            "last_activity", -1
        ).limit(limit)
        
        async for session_doc in sessions_cursor:
            session = ChatSession(**session_doc)
            sessions.append(SessionSummary(
                id=str(session.id),
                title=session.title,
                description=session.description,
                message_count=session.message_count,
                last_activity=session.last_activity,
                repo_context=session.repo_context,
                total_cost=session.total_cost,
                created_at=session.created_at
            ))
        
        return sessions
    
    async def get_session_context(self, session_id: PyObjectId) -> str:
        """Get conversation context for a session"""
        return await self.conversation_manager.get_conversation_context(session_id)
    
    async def get_buffer_stats(self, session_id: PyObjectId) -> Dict[str, Any]:
        """Get conversation buffer statistics"""
        return await self.conversation_manager.get_buffer_stats(session_id)
