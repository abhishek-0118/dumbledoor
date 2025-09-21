"""MongoDB connection and configuration."""

import os
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from typing import Optional

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    database = None

# MongoDB connection
mongodb = MongoDB()

async def connect_to_mongo():
    """Create database connection."""
    mongodb.client = AsyncIOMotorClient(
        os.getenv("MONGODB_URL", "mongodb://localhost:27017")
    )
    mongodb.database = mongodb.client[os.getenv("DATABASE_NAME", "jarvis_db")]
    
    # Test connection
    try:
        await mongodb.client.admin.command('ping')
        print("Successfully connected to MongoDB!")
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection."""
    if mongodb.client:
        mongodb.client.close()

def get_database():
    """Get database instance."""
    if mongodb.database is None:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("Database not connected. Server startup may have failed.")
            else:
                # Synchronous fallback (shouldn't normally happen)
                loop.run_until_complete(connect_to_mongo())
        except Exception as e:
            raise RuntimeError(f"Database connection failed: {e}")
    return mongodb.database
