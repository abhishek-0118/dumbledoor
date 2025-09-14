import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import tiktoken
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationSummaryBufferMemory
from langchain.memory.chat_message_histories import BaseChatMessageHistory

from ..models import ConversationBuffer, ChatMessage, MessageRole, PyObjectId
from ..db.mongodb import mongodb
from ..core.chat import create_chat_llm

logger = logging.getLogger("app.core.conversation_buffer")


class MongoDBChatMessageHistory(BaseChatMessageHistory):
    """Custom chat message history implementation using MongoDB"""
    
    def __init__(self, session_id: PyObjectId):
        self.session_id = session_id
        self._messages: List[BaseMessage] = []
        self._loaded = False
    
    async def load_messages(self):
        """Load messages from MongoDB"""
        if self._loaded:
            return
        
        messages_cursor = mongodb.chat_messages.find(
            {"session_id": self.session_id, "is_deleted": False}
        ).sort("created_at", 1)
        
        self._messages = []
        async for msg_doc in messages_cursor:
            message = ChatMessage(**msg_doc)
            if message.role == MessageRole.USER:
                self._messages.append(HumanMessage(content=message.content))
            elif message.role == MessageRole.ASSISTANT:
                self._messages.append(AIMessage(content=message.content))
        
        self._loaded = True
    
    @property
    def messages(self) -> List[BaseMessage]:
        """Get all messages"""
        return self._messages
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history"""
        self._messages.append(message)
    
    def clear(self) -> None:
        """Clear all messages"""
        self._messages = []


class ConversationBufferManager:
    """Manages conversation buffers for maintaining context across chat sessions"""
    
    def __init__(self, chat_config):
        self.chat_config = chat_config
        self.default_max_token_limit = 4000
        self.default_max_messages = 20
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    async def get_or_create_buffer(
        self, 
        session_id: PyObjectId,
        max_token_limit: Optional[int] = None,
        max_messages: Optional[int] = None
    ) -> ConversationBuffer:
        """Get existing buffer or create new one for a session"""
        
        # Try to find existing buffer
        buffer_doc = await mongodb.conversation_buffers.find_one({"session_id": session_id})
        
        if buffer_doc:
            buffer = ConversationBuffer(**buffer_doc)
            logger.debug(f"Found existing buffer for session {session_id}")
        else:
            # Create new buffer
            buffer = ConversationBuffer(
                session_id=session_id,
                max_token_limit=max_token_limit or self.default_max_token_limit,
                max_messages=max_messages or self.default_max_messages
            )
            
            await mongodb.conversation_buffers.insert_one(buffer.dict(by_alias=True))
            logger.info(f"Created new conversation buffer for session {session_id}")
        
        return buffer
    
    async def add_message_to_buffer(
        self,
        session_id: PyObjectId,
        role: MessageRole,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConversationBuffer:
        """Add a message to the conversation buffer"""
        
        buffer = await self.get_or_create_buffer(session_id)
        
        # Create message dict for buffer
        message_dict = {
            "role": role.value,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {}
        }
        
        # Add message to buffer
        buffer.messages.append(message_dict)
        
        # Calculate token count for the new message
        message_tokens = len(self.encoding.encode(content))
        buffer.total_tokens += message_tokens
        
        # Check if buffer needs trimming
        buffer = await self._trim_buffer_if_needed(buffer)
        
        # Update buffer in database
        await self._update_buffer_in_db(buffer)
        
        return buffer
    
    async def _trim_buffer_if_needed(self, buffer: ConversationBuffer) -> ConversationBuffer:
        """Trim buffer if it exceeds limits"""
        
        # Check message count limit
        if len(buffer.messages) > buffer.max_messages:
            messages_to_remove = len(buffer.messages) - buffer.max_messages
            buffer.messages = buffer.messages[messages_to_remove:]
            logger.debug(f"Trimmed {messages_to_remove} messages from buffer")
        
        # Check token limit
        if buffer.total_tokens > buffer.max_token_limit:
            # Recalculate actual token count
            actual_tokens = sum(
                len(self.encoding.encode(msg["content"])) 
                for msg in buffer.messages
            )
            buffer.total_tokens = actual_tokens
            
            # If still over limit, summarize older messages
            if buffer.total_tokens > buffer.max_token_limit:
                buffer = await self._summarize_and_trim(buffer)
        
        return buffer
    
    async def _summarize_and_trim(self, buffer: ConversationBuffer) -> ConversationBuffer:
        """Summarize older messages and keep recent ones"""
        
        try:
            # Keep the last few messages and summarize the rest
            keep_messages = 8  # Keep last 8 messages
            
            if len(buffer.messages) <= keep_messages:
                return buffer
            
            # Messages to summarize
            messages_to_summarize = buffer.messages[:-keep_messages]
            messages_to_keep = buffer.messages[-keep_messages:]
            
            # Create conversation text for summarization
            conversation_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in messages_to_summarize
            ])
            
            # Create LLM for summarization
            llm = create_chat_llm(self.chat_config, self.chat_config.provider)
            
            # Create summary prompt
            summary_prompt = f"""Please provide a concise summary of the following conversation, focusing on key topics, decisions, and context that would be useful for continuing the conversation:

{conversation_text}

Summary:"""
            
            # Generate summary
            summary = await llm.apredict(summary_prompt)
            
            # Update buffer
            buffer.summary = summary
            buffer.messages = messages_to_keep
            buffer.last_summarized_at = datetime.utcnow()
            
            # Recalculate token count
            buffer.total_tokens = sum(
                len(self.encoding.encode(msg["content"])) 
                for msg in buffer.messages
            )
            
            if buffer.summary:
                buffer.total_tokens += len(self.encoding.encode(buffer.summary))
            
            logger.info(f"Summarized {len(messages_to_summarize)} messages for session {buffer.session_id}")
            
        except Exception as e:
            logger.error(f"Failed to summarize conversation: {e}")
            # Fallback: just trim messages without summarization
            buffer.messages = buffer.messages[-keep_messages:]
            buffer.total_tokens = sum(
                len(self.encoding.encode(msg["content"])) 
                for msg in buffer.messages
            )
        
        return buffer
    
    async def get_conversation_context(self, session_id: PyObjectId) -> str:
        """Get conversation context for use in prompts"""
        
        buffer = await self.get_or_create_buffer(session_id)
        
        context_parts = []
        
        # Add summary if available
        if buffer.summary:
            context_parts.append(f"Previous conversation summary:\n{buffer.summary}\n")
        
        # Add recent messages
        if buffer.messages:
            context_parts.append("Recent conversation:")
            for msg in buffer.messages[-10:]:  # Last 10 messages
                role = msg["role"].upper()
                content = msg["content"][:500]  # Truncate long messages
                context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
    async def get_conversation_memory(self, session_id: PyObjectId) -> ConversationSummaryBufferMemory:
        """Get a LangChain conversation memory object"""
        
        # Create custom message history
        message_history = MongoDBChatMessageHistory(session_id)
        await message_history.load_messages()
        
        # Create LLM for summarization
        llm = create_chat_llm(self.chat_config, self.chat_config.provider)
        
        # Create memory
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            chat_memory=message_history,
            max_token_limit=self.default_max_token_limit,
            return_messages=True
        )
        
        return memory
    
    async def update_conversation_concepts(
        self,
        session_id: PyObjectId,
        new_concepts: List[str],
        active_files: List[str],
        current_repo: Optional[str] = None
    ):
        """Update conversation concepts and active files"""
        
        buffer = await self.get_or_create_buffer(session_id)
        
        # Update key concepts (keep unique, limit to 20)
        all_concepts = set(buffer.key_concepts + new_concepts)
        buffer.key_concepts = list(all_concepts)[:20]
        
        # Update active files (keep unique, limit to 15)
        all_files = set(buffer.active_files + active_files)
        buffer.active_files = list(all_files)[:15]
        
        # Update current repo
        if current_repo:
            buffer.current_repo = current_repo
        
        # Update buffer in database
        await self._update_buffer_in_db(buffer)
    
    async def _update_buffer_in_db(self, buffer: ConversationBuffer):
        """Update buffer in database"""
        buffer.updated_at = datetime.utcnow()
        
        await mongodb.conversation_buffers.update_one(
            {"session_id": buffer.session_id},
            {"$set": buffer.dict(exclude={"id"})},
            upsert=True
        )
    
    async def clear_buffer(self, session_id: PyObjectId):
        """Clear conversation buffer"""
        await mongodb.conversation_buffers.delete_one({"session_id": session_id})
        logger.info(f"Cleared conversation buffer for session {session_id}")
    
    async def get_buffer_stats(self, session_id: PyObjectId) -> Dict[str, Any]:
        """Get buffer statistics"""
        buffer = await self.get_or_create_buffer(session_id)
        
        return {
            "message_count": len(buffer.messages),
            "total_tokens": buffer.total_tokens,
            "max_token_limit": buffer.max_token_limit,
            "has_summary": bool(buffer.summary),
            "key_concepts": buffer.key_concepts,
            "active_files": buffer.active_files,
            "current_repo": buffer.current_repo,
            "last_updated": buffer.updated_at,
            "buffer_utilization": min(buffer.total_tokens / buffer.max_token_limit, 1.0)
        }
