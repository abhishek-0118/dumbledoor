import logging
import os
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import IndexModel, ASCENDING, DESCENDING
from contextlib import asynccontextmanager

logger = logging.getLogger("app.db.mongodb")


class MongoDB:
    """MongoDB connection and database management"""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.connected = False
    
    async def connect(self, connection_string: str, database_name: str):
        """Connect to MongoDB"""
        try:
            self.client = AsyncIOMotorClient(connection_string)
            # Test the connection
            await self.client.admin.command('ping')
            self.database = self.client[database_name]
            self.connected = True
            logger.info(f"Connected to MongoDB database: {database_name}")
            
            # Create indexes
            await self._create_indexes()
            
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self.client:
            self.client.close()
            self.connected = False
            logger.info("Disconnected from MongoDB")

    async def _create_indexes(self):
        """Create database indexes for optimal performance"""
        try:
            # User collection indexes
            users_collection = self.database.users
            user_indexes = [
                IndexModel([("google_id", ASCENDING)], unique=True),
                IndexModel([("email", ASCENDING)], unique=True),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("last_login", DESCENDING)]),
            ]
            await users_collection.create_indexes(user_indexes)
            
            # Chat sessions collection indexes
            sessions_collection = self.database.chat_sessions
            session_indexes = [
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("updated_at", DESCENDING)]),
                IndexModel([("last_activity", DESCENDING)]),
                IndexModel([("is_active", ASCENDING)]),
                IndexModel([("is_deleted", ASCENDING)]),
                IndexModel([("user_id", ASCENDING), ("is_active", ASCENDING), ("is_deleted", ASCENDING)]),
            ]
            await sessions_collection.create_indexes(session_indexes)
            
            # Chat messages collection indexes
            messages_collection = self.database.chat_messages
            message_indexes = [
                IndexModel([("session_id", ASCENDING)]),
                IndexModel([("created_at", DESCENDING)]),
                IndexModel([("role", ASCENDING)]),
                IndexModel([("is_deleted", ASCENDING)]),
                IndexModel([("session_id", ASCENDING), ("created_at", ASCENDING)]),
                IndexModel([("session_id", ASCENDING), ("is_deleted", ASCENDING)]),
            ]
            await messages_collection.create_indexes(message_indexes)
            
            # Conversation buffers collection indexes
            buffers_collection = self.database.conversation_buffers
            buffer_indexes = [
                IndexModel([("session_id", ASCENDING)], unique=True),
                IndexModel([("updated_at", DESCENDING)]),
            ]
            await buffers_collection.create_indexes(buffer_indexes)
            
            # User sessions collection indexes
            user_sessions_collection = self.database.user_sessions
            user_session_indexes = [
                IndexModel([("user_id", ASCENDING)]),
                IndexModel([("session_token", ASCENDING)], unique=True),
                IndexModel([("expires_at", ASCENDING)]),
                IndexModel([("is_active", ASCENDING)]),
                IndexModel([("user_id", ASCENDING), ("is_active", ASCENDING)]),
            ]
            await user_sessions_collection.create_indexes(user_session_indexes)
            
            logger.info("Database indexes created successfully")
            
        except Exception as e:
            logger.error(f"Failed to create indexes: {e}")
            # Don't raise here as the application can still work without indexes

    def get_collection(self, collection_name: str) -> AsyncIOMotorCollection:
        """Get a collection from the database"""
        if not self.connected or not self.database:
            raise RuntimeError("Database not connected")
        return self.database[collection_name]

    @property
    def users(self) -> AsyncIOMotorCollection:
        """Get users collection"""
        return self.get_collection("users")

    @property
    def chat_sessions(self) -> AsyncIOMotorCollection:
        """Get chat sessions collection"""
        return self.get_collection("chat_sessions")

    @property
    def chat_messages(self) -> AsyncIOMotorCollection:
        """Get chat messages collection"""
        return self.get_collection("chat_messages")

    @property
    def conversation_buffers(self) -> AsyncIOMotorCollection:
        """Get conversation buffers collection"""
        return self.get_collection("conversation_buffers")

    @property
    def user_sessions(self) -> AsyncIOMotorCollection:
        """Get user sessions collection"""
        return self.get_collection("user_sessions")


# Global MongoDB instance
mongodb = MongoDB()


async def get_database() -> AsyncIOMotorDatabase:
    """Get the database instance"""
    if not mongodb.connected:
        raise RuntimeError("Database not connected")
    return mongodb.database


@asynccontextmanager
async def get_db_transaction():
    """Get a database transaction context"""
    if not mongodb.connected:
        raise RuntimeError("Database not connected")
    
    async with await mongodb.client.start_session() as session:
        async with session.start_transaction():
            yield session


def get_mongodb_connection_string() -> str:
    """Get MongoDB connection string from environment variables"""
    # Try different environment variable names
    connection_string = (
        os.getenv("MONGODB_CONNECTION_STRING") or
        os.getenv("MONGODB_URI") or
        os.getenv("MONGO_URI") or
        "mongodb://localhost:27017"  # Default local MongoDB
    )
    return connection_string


def get_database_name() -> str:
    """Get database name from environment variables"""
    return os.getenv("MONGODB_DATABASE", "starkfoundation")


async def init_mongodb():
    """Initialize MongoDB connection"""
    connection_string = get_mongodb_connection_string()
    database_name = get_database_name()
    
    logger.info(f"Initializing MongoDB connection to: {database_name}")
    await mongodb.connect(connection_string, database_name)


async def close_mongodb():
    """Close MongoDB connection"""
    await mongodb.disconnect()
