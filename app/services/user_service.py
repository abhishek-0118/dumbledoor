"""User service for database operations."""

from datetime import datetime
from typing import Optional, Dict, Any
from bson import ObjectId
from ..db.mongodb import get_database
from ..models.user import User, UserCreate, UserUpdate, UserSettings
from ..auth.jwt_auth import create_user_token

class UserService:
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

    @property
    def db(self):
        """Get database instance (synchronous, for backward compatibility)."""
        if self._db is None:
            self._db = get_database()
            if self._db is None:
                raise RuntimeError("Database not connected. Make sure MongoDB is running and the server has started properly.")
        return self._db

    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user."""
        db = await self.get_db()
        
        # Check if user already exists
        existing_user = await db.users.find_one({"google_id": user_data.google_id})
        if existing_user:
            return User(**existing_user)
        
        # Create new user
        user_dict = user_data.dict()
        user_dict["settings"] = UserSettings().dict()
        user_dict["created_at"] = datetime.utcnow()
        user_dict["updated_at"] = datetime.utcnow()
        
        result = await db.users.insert_one(user_dict)
        user_dict["_id"] = result.inserted_id
        
        return User(**user_dict)

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID."""
        db = await self.get_db()
        user_data = await db.users.find_one({"_id": ObjectId(user_id)})
        if user_data:
            return User(**user_data)
        return None

    async def get_user_by_google_id(self, google_id: str) -> Optional[User]:
        """Get user by Google ID."""
        db = await self.get_db()
        user_data = await db.users.find_one({"google_id": google_id})
        if user_data:
            return User(**user_data)
        return None

    async def update_user(self, user_id: str, update_data: UserUpdate) -> Optional[User]:
        """Update user information."""
        db = await self.get_db()
        update_dict = {k: v for k, v in update_data.dict().items() if v is not None}
        update_dict["updated_at"] = datetime.utcnow()
        
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": update_dict}
        )
        
        if result.modified_count:
            return await self.get_user_by_id(user_id)
        return None

    async def update_last_login(self, user_id: str) -> None:
        """Update user's last login timestamp."""
        db = await self.get_db()
        await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"last_login": datetime.utcnow()}}
        )

    async def increment_query_count(self, user_id: str) -> bool:
        """Increment user's query count."""
        db = await self.get_db()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$inc": {"settings.queries_used": 1}}
        )
        return result.modified_count > 0

    async def reset_query_count(self, user_id: str) -> bool:
        """Reset user's query count."""
        db = await self.get_db()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {
                "$set": {
                    "settings.queries_used": 0,
                    "settings.reset_date": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    async def update_query_limit(self, user_id: str, new_limit: int) -> bool:
        """Update user's query limit."""
        db = await self.get_db()
        result = await db.users.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {"settings.query_limit": new_limit}}
        )
        return result.modified_count > 0

    async def create_user_from_google_data(self, google_user_data: Dict[str, Any]) -> User:
        """Create user from Google OAuth data."""
        user_create = UserCreate(
            google_id=google_user_data["id"],
            email=google_user_data["email"],
            name=google_user_data["name"],
            picture=google_user_data.get("picture")
        )
        
        user = await self.create_user(user_create)
        await self.update_last_login(str(user.id))
        
        return user

    async def generate_user_token(self, user: User) -> str:
        """Generate JWT token for user."""
        return create_user_token(str(user.id), user.email, user.name)

# Global service instance - will be initialized lazily
user_service = UserService()
