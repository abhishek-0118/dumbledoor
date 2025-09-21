"""Chat service for database operations."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from bson import ObjectId
from ..db.mongodb import get_database
from ..models.chat import Chat, ChatCreate, ChatSummary, Message, MessageCreate, ChatUpdate

class ChatService:
    def __init__(self):
        self._db = None

    async def get_db(self):
        """Get database instance asynchronously, ensuring connection."""
        if self._db is None:
            from ..db.mongodb import mongodb, connect_to_mongo
            
            # Check if already connected
            if mongodb.database is None:
                try:
                    await connect_to_mongo()
                except Exception as e:
                    raise RuntimeError(f"Failed to connect to database: {e}")
            
            self._db = mongodb.database
            
            if self._db is None:
                raise RuntimeError("Database connection failed. Check MongoDB status and configuration.")
        
        return self._db

    async def create_chat(self, user_id: str, chat_data: ChatCreate) -> Chat:
        """Create a new chat."""
        db = await self.get_db()
        
        chat_dict = {
            "user_id": user_id,
            "title": chat_data.title,
            "messages": [],
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "metadata": {}
        }
        
        # Add first message if provided
        if chat_data.first_message:
            first_msg = Message(
                role="user",
                content=chat_data.first_message,
                timestamp=datetime.utcnow()
            )
            chat_dict["messages"] = [first_msg.dict()]
        
        result = await db.chats.insert_one(chat_dict)
        chat_dict["_id"] = result.inserted_id
        
        return Chat(**chat_dict)

    async def get_chat_by_id(self, chat_id: str, user_id: str) -> Optional[Chat]:
        """Get chat by ID and user ID."""
        db = await self.get_db()
        
        chat_data = await db.chats.find_one({
            "_id": ObjectId(chat_id),
            "user_id": user_id
        })
        
        if chat_data:
            return Chat(**chat_data)
        return None

    async def get_user_chats(self, user_id: str, limit: int = 50) -> List[ChatSummary]:
        """Get user's chat summaries for sidebar."""
        db = await self.get_db()
        
        cursor = db.chats.find(
            {"user_id": user_id},
            {
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
                "messages": 1
            }
        ).sort("updated_at", -1).limit(limit)
        
        chats = []
        async for chat_data in cursor:
            chat_summary = ChatSummary(
                id=str(chat_data["_id"]),
                title=chat_data["title"],
                created_at=chat_data["created_at"],
                updated_at=chat_data["updated_at"],
                message_count=len(chat_data.get("messages", []))
            )
            chats.append(chat_summary)
        
        return chats

    async def add_message_to_chat(
        self, 
        chat_id: str, 
        user_id: str, 
        message_data: MessageCreate
    ) -> Optional[Chat]:
        """Add a message to an existing chat."""
        db = await self.get_db()
        
        message = Message(**message_data.dict())
        
        result = await db.chats.update_one(
            {"_id": ObjectId(chat_id), "user_id": user_id},
            {
                "$push": {"messages": message.dict()},
                "$set": {"updated_at": datetime.utcnow()}
            }
        )
        
        if result.modified_count:
            return await self.get_chat_by_id(chat_id, user_id)
        return None

    async def update_chat(
        self, 
        chat_id: str, 
        user_id: str, 
        update_data: ChatUpdate
    ) -> Optional[Chat]:
        """Update chat information."""
        db = await self.get_db()
        
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        update_dict["updated_at"] = datetime.utcnow()
        
        result = await db.chats.update_one(
            {"_id": ObjectId(chat_id), "user_id": user_id},
            {"$set": update_dict}
        )
        
        if result.modified_count:
            return await self.get_chat_by_id(chat_id, user_id)
        return None

    async def delete_chat(self, chat_id: str, user_id: str) -> bool:
        """Delete a chat."""
        db = await self.get_db()
        
        result = await db.chats.delete_one({
            "_id": ObjectId(chat_id),
            "user_id": user_id
        })
        return result.deleted_count > 0

    async def get_chat_messages(
        self, 
        chat_id: str, 
        user_id: str, 
        limit: int = 100
    ) -> List[Message]:
        """Get messages from a chat."""
        db = await self.get_db()
        
        chat_data = await db.chats.find_one(
            {"_id": ObjectId(chat_id), "user_id": user_id},
            {"messages": {"$slice": -limit}}
        )
        
        if chat_data and "messages" in chat_data:
            return [Message(**msg) for msg in chat_data["messages"]]
        return []

    async def search_user_chats(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 20
    ) -> List[ChatSummary]:
        """Search user's chats by title or content."""
        db = await self.get_db()
        
        # Search in title and message content
        search_filter = {
            "user_id": user_id,
            "$or": [
                {"title": {"$regex": query, "$options": "i"}},
                {"messages.content": {"$regex": query, "$options": "i"}}
            ]
        }
        
        cursor = db.chats.find(
            search_filter,
            {
                "title": 1,
                "created_at": 1,
                "updated_at": 1,
                "messages": 1
            }
        ).sort("updated_at", -1).limit(limit)
        
        chats = []
        async for chat_data in cursor:
            chat_summary = ChatSummary(
                id=str(chat_data["_id"]),
                title=chat_data["title"],
                created_at=chat_data["created_at"],
                updated_at=chat_data["updated_at"],
                message_count=len(chat_data.get("messages", []))
            )
            chats.append(chat_summary)
        
        return chats

    async def generate_chat_title(self, first_message: str, max_length: int = 50) -> str:
        """Generate a chat title from the first message."""
        # Simple title generation - take first few words
        words = first_message.split()[:6]
        title = " ".join(words)
        
        if len(title) > max_length:
            title = title[:max_length-3] + "..."
        
        return title or "New Chat"

# Global service instance
chat_service = ChatService()
